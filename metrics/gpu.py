from metrics.metric import Metric
from typing import Dict, Union
import torch


class Float:
    def __init__(self, f):
        self.f = f
        
    def item(self):
        return self.f


class GPUStats(Metric):
    """
    Log gpu memory usage.
    Implementation modified from pytorch_lightning/accetrators/cuda/get_nvidia_gpu_stats
    """

    def __init__(self, args: Dict):
        self.name = 'gpu_memory_usage'
        
    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Get GPU usage
        :param predictions: Not used
        :param ground_truth: Not used
        :return:
        """
        result = Float(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024)
        torch.cuda.reset_peak_memory_stats()
        return result