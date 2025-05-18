import torch.optim
import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_prediction_model, initialize_metric,\
    initialize_dataset, get_specific_args
import torch
import time
import math
import os
import timm.scheduler
import train_eval.utils as u
from utils import device, get_wd_params


class Trainer:
    """
    Trainer class for running train-val loops
    """
    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path=None, just_weights=False, writer=None, num_epochs=-1):
        """
        Initialize trainer object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        :param just_weights: Load just weights from checkpoint
        :param writer: Tensorboard summary writer
        :param num_epochs: Number of epochs to run training for
        """

        self.num_epochs = num_epochs
        
        # Initialize datasets:
        ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
        spec_args = get_specific_args(cfg['dataset'], data_root, cfg['version'] if 'version' in cfg.keys() else None)
        train_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg['train_set_args']] + spec_args)
        val_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg['val_set_args']] + spec_args)
        datasets = {'train': train_set, 'val': val_set}

        # Initialize dataloaders
        if os.environ["LANEFORMERSEED"] != 'None':
            seed = int(os.environ["LANEFORMERSEED"])
            tr_g = torch.Generator()
            tr_g.manual_seed(seed)
            val_g = torch.Generator()
            val_g.manual_seed(seed)
        else:
            tr_g, val_g = None, None

        self.tr_dl = torch_data.DataLoader(datasets['train'], cfg['batch_size'], shuffle=True,
                                           num_workers=cfg['num_workers'], pin_memory=True,
                                           drop_last=True, persistent_workers=cfg['num_workers'] > 0,
                                           worker_init_fn=u.seed_worker, generator=tr_g)
        self.val_dl = torch_data.DataLoader(datasets['val'], cfg.get('val_batch_size', cfg['batch_size']), shuffle=False,
                                            num_workers=cfg['num_workers'], pin_memory=True,
                                            drop_last=False, persistent_workers=cfg['num_workers'] > 0,
                                            worker_init_fn=u.seed_worker, generator=val_g)
        self.len_tr_dl = len(self.tr_dl)
        
        # Initialize model
        self.model = initialize_prediction_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                                 cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])

        # Initialize optimizer and scheduler
        optimizer_mapping = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'adamw': torch.optim.AdamW}
        scheduler_mapping = {'step': timm.scheduler.StepLRScheduler, 'cosine': timm.scheduler.CosineLRScheduler}
        scheduler_specific_args = {'step': {'t_in_epochs': True}, 'cosine': {'t_initial': self.len_tr_dl * self.num_epochs, 't_in_epochs': False}}
        
        self.scheduler_interval = cfg['optim_args']['scheduler_interval']
        
        optimizer = cfg['optim_args']['optimizer']
        scheduler = cfg['optim_args']['scheduler']
        
        param_groups = get_wd_params(self.model)
        
        self.optimizer = optimizer_mapping[optimizer](param_groups, **cfg['optim_args']['optimizer_kwargs'])
        self.scheduler = scheduler_mapping[scheduler](self.optimizer, **scheduler_specific_args[scheduler], **cfg['optim_args']['scheduler_kwargs'])

        # Initialize epochs
        self.current_epoch = 0

        # Initialize losses
        self.losses = [initialize_metric(cfg['losses'][i], cfg['loss_args'][i]) for i in range(len(cfg['losses']))]
        self.loss_weights = cfg['loss_weights']

        # Initialize metrics
        self.train_metrics = [initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i], train_set.helper)
                              for i in range(len(cfg['tr_metrics']))]
        self.val_metrics = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i], val_set.helper)
                            for i in range(len(cfg['val_metrics']))]
        self.val_metric = math.inf
        self.min_val_metric = math.inf

        # Print metrics after these many minibatches to keep track of training
        self.log_period = len(self.tr_dl)//cfg['log_freq']

        # Initialize tensorboard writer
        self.writer = writer
        self.tb_iters = 0

        # Load checkpoint if checkpoint path is provided
        if checkpoint_path is not None:
            print()
            print("Loading checkpoint from " + checkpoint_path + " ...", end=" ")
            self.load_checkpoint(checkpoint_path, just_weights=just_weights)
            print("Done")

        # Generate anchors if using an anchor based trajectory decoder
        if hasattr(self.model.decoder, 'need_anchors') or (hasattr(self.model.decoder, 'anchors') and torch.as_tensor(self.model.decoder.anchors == 0).all()):
            if 'anchors_checkpoint' in cfg:
                anchors = torch.load(cfg['anchors_checkpoint'])
                self.model.decoder.generate_anchors(anchors=anchors)
            else:
                self.model.decoder.generate_anchors(ds=self.tr_dl.dataset)

        self.model = self.model.float().to(device)

    def train(self, output_dir: str):
        """
        Main function to train model
        :param output_dir: Output directory to store tensorboard logs and checkpoints
        :return:
        """

        # Run training, validation for given number of epochs
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            
            torch.cuda.empty_cache()
            
            # Set current epoch
            self.current_epoch = epoch
            print()
            print('Epoch (' + str(self.current_epoch + 1) + '/' + str(start_epoch + self.num_epochs) + ')')

            # Train
            train_loss, train_epoch_metrics = self.run_epoch('train', self.tr_dl)
            self.print_metrics(train_loss, train_epoch_metrics, self.tr_dl, mode='train')

            # Validate
            with torch.no_grad():
                _, val_epoch_metrics = self.run_epoch('val', self.val_dl)
            self.print_metrics(None, val_epoch_metrics, self.val_dl, mode='val')

            # Scheduler step
            if self.scheduler_interval == 'epoch':
                self.scheduler.step(self.current_epoch)

            # Update validation metric
            self.val_metric = val_epoch_metrics[self.val_metrics[0].name] / val_epoch_metrics['minibatch_count']

            # save best checkpoint when applicable
            if self.val_metric < self.min_val_metric:
                self.min_val_metric = self.val_metric
                self.save_checkpoint(os.path.join(output_dir, 'checkpoints', 'best.tar'))

            # Save checkpoint
            # self.save_checkpoint(os.path.join(output_dir, 'checkpoints', str(self.current_epoch) + '.tar'))

    def run_epoch(self, mode: str, dl: torch_data.DataLoader):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'val'
        :param dl: dataloader object
        """
        if mode == 'val':
            self.model.eval()
        else:
            self.model.train()
            
        loss_total = 0.

        # Initialize epoch metrics
        epoch_metrics = self.initialize_metrics_for_epoch(mode)
        
        

        # Main loop
        st_time = time.time()
        for i, data in enumerate(dl):

            # Load data
            data = u.send_to_device(u.convert_double_to_float(data))

            # Forward pass
            predictions = self.model(data['inputs'])

            # Compute loss and backprop if training
            if mode == 'train':
                loss = self.compute_loss(predictions, data['ground_truth'])
                self.back_prop(loss)
                if self.scheduler_interval == 'step':
                    self.scheduler.step_update(self.current_epoch * self.len_tr_dl + i)
                loss_val = loss.item()
                loss_total += loss_val
            else:
                loss_val = None

            # Keep time
            minibatch_time = time.time() - st_time
            st_time = time.time()

            # Aggregate metrics
            minibatch_metrics, epoch_metrics = self.aggregate_metrics(epoch_metrics, minibatch_time,
                                                                      predictions, data['ground_truth'], mode)

            # Log minibatch metrics to tensorboard during training
            if mode == 'train':
                self.log_tensorboard_train(loss_val, minibatch_metrics)

            # Display metrics at a predefined frequency
            if i % self.log_period == self.log_period - 1:
                self.print_metrics(loss_val, epoch_metrics, dl, mode)

        # Log val metrics for the complete epoch to tensorboard
        if mode == 'val':
            self.log_tensorboard_val(epoch_metrics)

        return loss_total / self.len_tr_dl, epoch_metrics

    def compute_loss(self, model_outputs: Dict, ground_truth: Dict) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        loss_vals = [loss.compute(model_outputs, ground_truth) for loss in self.losses]
        total_loss = torch.as_tensor(0, device=device).float()
        for n in range(len(loss_vals)):
            total_loss += self.loss_weights[n] * loss_vals[n]

        return total_loss

    def back_prop(self, loss: torch.Tensor, grad_clip_thresh=10):
        """
        Backpropagate loss
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
        self.optimizer.step()

    def initialize_metrics_for_epoch(self, mode: str):
        """
        Initialize metrics for epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        epoch_metrics = {'minibatch_count': 0, 'time_elapsed': 0}
        for metric in metrics:
            epoch_metrics[metric.name] = 0

        return epoch_metrics

    @torch.no_grad()
    def aggregate_metrics(self, epoch_metrics: Dict, minibatch_time: float, model_outputs: Dict, ground_truth: Dict,
                          mode: str):
        """
        Aggregates metrics by minibatch for the entire epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        minibatch_metrics = {}
        for metric in metrics:
            minibatch_metrics[metric.name] = metric.compute(model_outputs, ground_truth).item()

        epoch_metrics['minibatch_count'] += 1
        epoch_metrics['time_elapsed'] += minibatch_time
        for metric in metrics:
            epoch_metrics[metric.name] += minibatch_metrics[metric.name]

        return minibatch_metrics, epoch_metrics

    def print_metrics(self, loss_val: float, epoch_metrics: Dict, dl: torch_data.DataLoader, mode: str):
        """
        Prints aggregated metrics
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        minibatches_left = len(dl) - epoch_metrics['minibatch_count']
        eta = (epoch_metrics['time_elapsed']/epoch_metrics['minibatch_count']) * minibatches_left
        epoch_progress = int(epoch_metrics['minibatch_count']/len(dl) * 100)
        print('\rTraining:' if mode == 'train' else '\rValidating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, str(epoch_progress), '%', end=", ")
        print('ETA:', int(eta), end="s, ")
        print('Metrics', end=": { ")
        if loss_val is not None:
            print('loss:', format(loss_val, '0.4f'), end=", ")
        for metric in metrics:
            metric_val = epoch_metrics[metric.name]/epoch_metrics['minibatch_count']
            print(metric.name + ':', format(metric_val, '0.2f'), end=", ")
        print('\b\b }', end="\n" if eta == 0 else "")

    def load_checkpoint(self, checkpoint_path, just_weights=False):
        """
        Loads checkpoint from given path
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not just_weights:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.val_metric = checkpoint['val_metric']
            self.min_val_metric = checkpoint['min_val_metric']

    def save_checkpoint(self, checkpoint_path):
        """
        Saves checkpoint to given path
        """
        torch.save({
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metric': self.val_metric,
            'min_val_metric': self.min_val_metric
        }, checkpoint_path)

    def log_tensorboard_train(self, loss_val: float, minibatch_metrics: Dict):
        """
        Logs minibatch metrics during training
        """
        self.writer.add_scalar('train/loss', loss_val, self.tb_iters)
        for metric_name, metric_val in minibatch_metrics.items():
            self.writer.add_scalar('train/' + metric_name, metric_val, self.tb_iters)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.tb_iters)
        self.tb_iters += 1

    def log_tensorboard_val(self, epoch_metrics):
        """
        Logs epoch metrics for validation set
        """
        for metric_name, metric_val in epoch_metrics.items():
            if metric_name != 'minibatch_count' and metric_name != 'time_elapsed':
                metric_val /= epoch_metrics['minibatch_count']
                self.writer.add_scalar('val/' + metric_name, metric_val, self.tb_iters)
