from metrics.metric import Metric
from typing import Dict, Union
import torch
from einops import rearrange


class TrajDiversity(Metric):
    """
    Multimodal trajectory prediction diversity.
    """
    def __init__(self, args: Dict):
        self.name = 'traj_diversity'
        self.scale = args['scale']

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Compute Diversity of Multimodal Trajectory prediction.

        :param predictions: Dictionary with 'traj': predicted trajectories
        """
        # Unpack arguments
        # batch_size, self.num_samples, self.op_len, 2
        traj = predictions['traj'][..., :2]
        traj = rearrange(traj, 'b s t d -> b s (t d)')
        loss = torch.mean(torch.exp(-(torch.cdist(traj, traj, compute_mode='use_mm_for_euclid_dist') / self.scale)))

        return loss
