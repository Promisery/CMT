from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import traj_to_coord_and_cov


class MMNLL(Metric):
    """
    Negative log likelyhood loss for multimodal trajectories.
    https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus
    """
    def __init__(self, args: Dict):
        self.name = 'mm_nll'

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute MinADEK
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        traj = predictions['traj']
        log_probs = predictions['log_probs']
        probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        
        # Useful params
        batch_size = probs.shape[0]
        num_modes = traj.shape[1]
        sequence_length = traj.shape[2]
        
        traj, covariance_matrices = traj_to_coord_and_cov(traj)

        precision_matrices = torch.inverse(covariance_matrices)
        gt = torch.unsqueeze(traj_gt, 1)
        avails = (1 - ground_truth['masks']) if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.ones(batch_size, sequence_length, device=traj.device)
        avails = avails[:, None, :, None]
        coordinates_delta = (gt - traj).unsqueeze(-1)
        errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
        errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))
        errors = log_probs + torch.sum(errors, dim=[2, 3])
        errors = -torch.logsumexp(errors, dim=-1, keepdim=True)
        return torch.mean(errors)
