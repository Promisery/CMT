import torch
from torch.distributions import Categorical
from datasets.interface import SingleAgentDataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import psutil
import ray
from scipy.spatial.distance import cdist
from tqdm import tqdm
from einops import rearrange, repeat
from utils import device


# Initialize ray:
if not ray.is_initialized():
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus, log_to_driver=False)


def gmm_anchors(k, ds: SingleAgentDataset):
    """
    Extracts anchors for multipath/covernet using gaussian mixture model on train set trajectories
    """
    prototype_traj = ds[0]['ground_truth']['traj']
    traj_len = prototype_traj.shape[0]
    traj_dim = prototype_traj.shape[1]
    ds_size = len(ds)
    trajectories = np.zeros((ds_size, traj_len, traj_dim), dtype=np.float32)
    for i, data in enumerate(tqdm(ds, desc='Extracting anchors')):
        trajectories[i] = data['ground_truth']['traj']
    clustering = GaussianMixture(n_components=k, max_iter=1000, random_state=0, covariance_type='diag').fit(trajectories.reshape((ds_size, -1)))
    anchors = np.concatenate([clustering.means_.reshape(k, traj_len, traj_dim), np.log(np.sqrt(clustering.covariances_)).reshape(k, traj_len, traj_dim)], axis=-1) #np.zeros((k, traj_len, traj_dim))
    # for i in range(k):
    #     anchors[i] = np.mean(trajectories[clustering.labels_ == i], axis=0)
    anchors = torch.from_numpy(anchors).float().to(device)
    return anchors


def bivariate_gaussian_activation_with_anchors(ip: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    
    anchor_mu_x = anchors[..., 0:1]
    anchor_mu_y = anchors[..., 1:2]
    anchor_sig_x = anchors[..., 2:3]
    anchor_sig_y = anchors[..., 3:4]
    
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    
    anchor_sig_x = torch.exp(anchor_sig_x)
    anchor_sig_y = torch.exp(anchor_sig_y)
    
    rho = torch.tanh(rho)
    
    out = torch.cat([mu_x + anchor_mu_x, mu_y + anchor_mu_y, sig_x * anchor_sig_x, sig_y * anchor_sig_y, rho], dim=-1)
    return out


@ray.remote
def kmeans_cluster_and_rank(k: int, data: np.ndarray):
    """
    Combines the clustering and ranking steps so that ray.remote gets called just once
    """

    def cluster(n_clusters: int, x: np.ndarray):
        """
        Cluster using Scikit learn
        """
        clustering_op = KMeans(n_clusters=n_clusters, n_init=1, max_iter=100, init='k-means++', random_state=0).fit(x)
        return clustering_op.cluster_centers_, clustering_op.labels_, clustering_op.cluster_centers_

    def rank_clusters(cluster_counts, cluster_centers):
        """
        Rank the K clustered trajectories using Ward's criterion. Start with K cluster centers and cluster counts.
        Find the two clusters to merge based on Ward's criterion. Smaller of the two will get assigned rank K.
        Merge the two clusters. Repeat process to assign ranks K-1, K-2, ..., 2.
        """

        num_clusters = len(cluster_counts)
        cluster_ids = np.arange(num_clusters)
        ranks = np.ones(num_clusters)

        for i in range(num_clusters, 0, -1):
            # Compute Ward distances:
            centroid_dists = cdist(cluster_centers, cluster_centers)
            n1 = cluster_counts.reshape(1, -1).repeat(len(cluster_counts), axis=0)
            n2 = n1.transpose()
            wts = n1 * n2 / (n1 + n2)
            dists = wts * centroid_dists + np.diag(np.inf * np.ones(len(cluster_counts)))

            # Get clusters with min Ward distance and select cluster with fewer counts
            c1, c2 = np.unravel_index(dists.argmin(), dists.shape)
            c = c1 if cluster_counts[c1] <= cluster_counts[c2] else c2
            c_ = c2 if cluster_counts[c1] <= cluster_counts[c2] else c1

            # Assign rank i to selected cluster
            ranks[cluster_ids[c]] = i

            # Merge clusters and update identity of merged cluster
            cluster_centers[c_] = (cluster_counts[c_] * cluster_centers[c_] + cluster_counts[c] * cluster_centers[c]) /\
                                  (cluster_counts[c_] + cluster_counts[c])
            cluster_counts[c_] += cluster_counts[c]

            # Discard merged cluster
            cluster_ids = np.delete(cluster_ids, c)
            cluster_centers = np.delete(cluster_centers, c, axis=0)
            cluster_counts = np.delete(cluster_counts, c)

        return ranks

    cluster_centers, cluster_lbls, cluster_ctrs = cluster(k, data)
    cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
    cluster_ranks = rank_clusters(cluster_cnts.copy(), cluster_ctrs.copy())
    return {'centers': cluster_centers, 'ranks': cluster_ranks}


def gmrc_cluster_traj(k: int, traj: torch.Tensor):
    """
    clusters sampled trajectories to output K modes for Gaussian Mixture Reduction.
    :param k: number of clusters
    :param traj: set of sampled trajectories, shape [batch_size, num_samples, traj_len, 2]
    :param cluster_method: Method used for clustering, should be one of 'kmeans' or 'gmm'
    :return: traj_clustered:  set of clustered trajectories, shape [batch_size, k, traj_len, 2]
             scores: scores for clustered trajectories (basically 1/rank), shape [batch_size, k]
    """

    # Initialize output tensors
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]

    data = traj.reshape(batch_size, num_samples, -1).detach().cpu().numpy()
    
    # Cluster and rank
    cluster_ops = ray.get([kmeans_cluster_and_rank.remote(k, data_slice) for data_slice in data])
    cluster_ranks = [cluster_op['ranks'] for cluster_op in cluster_ops]
    cluster_centers = [cluster_op['centers'] for cluster_op in cluster_ops]

    # Compute mean (clustered) traj and scores
    traj_clustered = rearrange(torch.as_tensor(np.array(cluster_centers), device=device), 'b n (t d) -> b n t d', t=traj_len)
    scores = 1 / torch.as_tensor(np.array(cluster_ranks), device=device)
    scores = scores / torch.sum(scores, dim=1)[0]

    return traj_clustered, scores


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


def gmrc(traj: torch.Tensor, log_probs: torch.Tensor, k: int, n_sample: int, **kwargs):
    
    probs = torch.exp(log_probs)
    
    traj, covariance_matrices = traj_to_coord_and_cov(traj)
    b, n, t, d = traj.shape
    
    traj_mode_dist = Categorical(probs)
    traj_selected = traj_mode_dist.sample((n_sample,))
    traj_selected = repeat(traj_selected, 's b -> s b 1 t d', t=t, d=d)
    
    scale_tril = torch.linalg.cholesky(covariance_matrices)
    eps = repeat(torch.randn(n_sample, b, n, d, device=device), 's b n d -> s b n t d', t=t)
    traj_sample = traj + torch.matmul(scale_tril, eps.unsqueeze(-1)).squeeze(-1)
        
    traj_sample = traj_sample.gather(2, traj_selected)
    traj_sample = rearrange(traj_sample, 's b 1 t d -> b s t d')
    
    traj, prob = gmrc_cluster_traj(k, traj_sample)
    
    return traj, prob, None