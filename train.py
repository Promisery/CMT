import argparse
import yaml
from train_eval.trainer import Trainer
from train_eval.utils import seed_everything
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import glob

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
    parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
    parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to save checkpoints and logs", required=True)
    parser.add_argument("-n", "--num_epochs", help="Number of epochs to run training for", required=True)
    parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=False)
    parser.add_argument("-s", "--seed", type=int, help="Random seed for everything", default=2024)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Make directories
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    if not os.path.isdir(os.path.join(args.output_dir, 'tensorboard_logs')):
        os.makedirs(os.path.join(args.output_dir, 'tensorboard_logs'), exist_ok=True)

    # Backup all scripts
    def backup(src, dst, exts=None, subdirs=[]):
        dst = os.path.join(dst, 'backup')
        if not os.path.isdir(dst):
            os.makedirs(dst, exist_ok=True)
        exts = ['*'] if exts is None else exts
        files = []
        for ext in exts:
            files.extend(list(glob.iglob(os.path.join(src, F"*.{ext}"))))
            for subdir in subdirs:
                files.extend(list(glob.iglob(os.path.join(src, subdir, '**', F"*.{ext}"), recursive=True))) 
        
        for file in files:
            if os.path.isfile(file):
                file_dst = os.path.dirname(os.path.join(dst, os.path.relpath(file)))
                os.makedirs(file_dst, exist_ok=True)
                shutil.copy2(file, file_dst)
    backup(os.curdir, args.output_dir, exts=['py', 'yml'], subdirs=['configs', 'datasets', 'metrics', 'models', 'train_eval'])

    # Load config
    with open(args.config, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    shutil.copy2(args.config, os.path.join(args.output_dir, 'config.yml'))
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))


    # Train
    trainer = Trainer(cfg, args.data_root, args.data_dir, checkpoint_path=args.checkpoint, writer=writer, num_epochs=int(args.num_epochs))
    trainer.train(output_dir=args.output_dir)


    # Close tensorboard writer
    writer.close()
