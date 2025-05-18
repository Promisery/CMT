import numpy as np
import torch
from typing import Tuple
import math
from utils import device


def mse(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes MSE for a set of trajectories with respect to ground truth.

    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return: errs: errors, shape [batch_size, num_modes]
    """

    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    return err


def max_dist(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes max distance of a set of trajectories with respect to ground truth trajectory.

    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return dist: shape [batch_size, num_modes]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(dist, exponent=2)
    dist = torch.sum(dist, dim=3)
    dist = torch.pow(dist, exponent=0.5)
    dist[masks_rpt.bool()] = -math.inf
    dist, _ = torch.max(dist, dim=2)

    return dist


def min_mse(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes MSE for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """

    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    err, inds = torch.min(err, dim=1)

    return err, inds


def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    err, inds = torch.min(err, dim=1)

    return err, inds


def min_fde(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes final displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    lengths = torch.sum(1-masks, dim=1).long()
    inds = lengths.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, num_modes, 1, 2) - 1

    traj_last = torch.gather(traj[..., :2], dim=2, index=inds).squeeze(2)
    traj_gt_last = torch.gather(traj_gt_rpt, dim=2, index=inds).squeeze(2)

    err = traj_gt_last - traj_last[..., 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=2)
    err = torch.pow(err, exponent=0.5)
    err, inds = torch.min(err, dim=1)

    return err, inds


def miss_rate(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor, dist_thresh: float = 2) -> torch.Tensor:
    """
    Computes miss rate for mini batch of trajectories, with respect to ground truth and given distance threshold
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :param dist_thresh: distance threshold for computing miss rate.
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(dist, exponent=2)
    dist = torch.sum(dist, dim=3)
    dist = torch.pow(dist, exponent=0.5)
    dist[masks_rpt.bool()] = -math.inf
    dist, _ = torch.max(dist, dim=2)
    dist, _ = torch.min(dist, dim=1)
    m_r = torch.sum(torch.as_tensor(dist > dist_thresh)) / len(dist)

    return m_r


# TODO: DEBUG THIS FUNCTION (?)
def traj_nll(pred_dist: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor):
    """
    Computes negative log likelihood of ground truth trajectory under a predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the prediction horizon

    :param pred_dist: parameters of a bivariate Gaussian distribution, shape [batch_size, sequence_length, 5]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return:
    """
    mu_x = pred_dist[:, :, 0]
    mu_y = pred_dist[:, :, 1]
    x = traj_gt[:, :, 0]
    y = traj_gt[:, :, 1]

    sig_x = pred_dist[:, :, 2]
    sig_y = pred_dist[:, :, 3]
    rho = pred_dist[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    nll = torch.pow(ohr, 2) * 0.5 * \
        (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) +
         torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2) -
         2 * rho * torch.pow(sig_x, 1) * torch.pow(sig_y, 1) * (x - mu_x) * (y - mu_y))\
        - torch.log(sig_x * sig_y * ohr) + 1.8379

    nll[nll.isnan()] = 0
    nll[nll.isinf()] = 0

    nll = torch.sum(nll * (1 - masks), dim=1) / torch.sum((1 - masks), dim=1)
    # Note: Normalizing with torch.sum((1 - masks), dim=1) makes values somewhat comparable for trajectories of
    # different lengths

    return nll


def traj_to_coord_and_cov(traj):
    sig_x = traj[..., 2]
    sig_y = traj[..., 3]
    rho = traj[..., 4]
    traj = traj[..., :2]
    
    covariance_matrices = torch.stack([
        torch.stack([torch.pow(sig_x, 2), rho * sig_x * sig_y], dim=-1),
        torch.stack([rho * sig_x * sig_y, torch.pow(sig_y, 2)], dim=-1),
    ], dim=-1)
    return traj, covariance_matrices


@torch.no_grad()
def _compute_coefficients(covariance_matrices_centroids, traj, traj_centroids, prob_centroids):
    
    # logdet on cuda is slow
    covariance_matrices_centroids = covariance_matrices_centroids.cpu()
    traj = traj.cpu()
    traj_centroids = traj_centroids.cpu()
    prob_centroids = prob_centroids.cpu()
    precision_matrices_centroids = torch.inverse(covariance_matrices_centroids)
    diff = traj_centroids[:, :, 3::4, :].unsqueeze(2) - \
        traj[:, :, 3::4, :].unsqueeze(1)
    A = diff.unsqueeze(-2)
    B = diff.unsqueeze(-1)
    C = precision_matrices_centroids[:, :, 3::4, :, :].unsqueeze(2)
    qform = (A @ C @ B)[..., 0, 0]
    logdetCovM = torch.logdet(covariance_matrices_centroids[:, :, 3::4, :, :].unsqueeze(2))
    pMatrix = torch.exp((
        - logdetCovM * 0.5 -np.log(2 * np.pi) - 0.5 * qform).sum(dim=-1)) + 1e-8
    pMatrix = (pMatrix * prob_centroids.unsqueeze(2)) / ((
        pMatrix * prob_centroids.unsqueeze(2)).sum(dim=1, keepdims=True) + 1e-8)
    return pMatrix.to(device)
