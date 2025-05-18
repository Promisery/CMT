from metrics.metric import Metric
from typing import Dict, Union, Tuple
import numpy as np
import torch
from scipy import interpolate
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction


class Float:
    def __init__(self, f):
        self.f = f
        
    def item(self):
        return self.f


class OffroadRate(Metric):
    """
    Offroad rate for the trajectories.
    """

    def __init__(self, args: Dict = None, helper = None):
        self.name = 'offroad_rate'
        if args is not None:
            self.nuscenes_version = args.get('nuscenes_version', 'v1.0-trainval')
            self.nuscenes_data_root = args.get('nuscenes_data_root', 'nuscenes')
        else:
            self.nuscenes_version = 'v1.0-trainval'
            self.nuscenes_data_root = 'nuscenes'
        
        if helper is None:
            ns = NuScenes(self.nuscenes_version, dataroot=self.nuscenes_data_root)
            helper = PredictHelper(ns)
        
        self.helper = helper
        self.drivable_area_polygons = self.load_drivable_area_masks(helper)
        self.pixels_per_meter = 10
        self.number_of_points = 200

    @staticmethod
    def load_drivable_area_masks(helper: PredictHelper) -> Dict[str, np.ndarray]:
        """
        Loads the polygon representation of the drivable area for each map.
        :param helper: Instance of PredictHelper.
        :return: Mapping from map_name to drivable area polygon.
        """

        maps: Dict[str, NuScenesMap] = load_all_maps(helper)

        masks = {}
        for map_name, map_api in maps.items():

            masks[map_name] = map_api.get_map_mask(patch_box=None, patch_angle=0, layer_names=['drivable_area'],
                                                   canvas_size=None)[0]

        return masks
    
    @staticmethod
    def interpolate_path(mode: np.ndarray, number_of_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Interpolate trajectory with a cubic spline if there are enough points. """

        # interpolate.splprep needs unique points.
        # We use a loop as opposed to np.unique because
        # the order of the points must be the same
        seen = set()
        ordered_array = []
        for row in mode:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                ordered_array.append(row_tuple)

        new_array = np.array(ordered_array)

        unique_points = np.atleast_2d(new_array)

        if unique_points.shape[0] <= 3:
            return unique_points[:, 0], unique_points[:, 1]
        else:
            knots, _ = interpolate.splprep([unique_points[:, 0], unique_points[:, 1]], k=3, s=0.1)
            x_interpolated, y_interpolated = interpolate.splev(np.linspace(0, 1, number_of_points), knots)
            return x_interpolated, y_interpolated
    
    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute miss rate
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        traj = predictions['traj']
        probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        
        traj = predictions['traj'][..., :2]
        probs = predictions['probs']

        # Load instance and sample tokens for batch
        instance_tokens = predictions['instance_token']
        sample_tokens = predictions['sample_token']

        preds = []
        
        # Create prediction object and add to list of predictions
        for n in range(traj.shape[0]):

            traj_local = traj[n].detach().cpu().numpy()
            probs_n = probs[n].detach().cpu().numpy()
            starting_annotation = self.helper.get_sample_annotation(instance_tokens[n], sample_tokens[n])
            traj_global = np.zeros_like(traj_local)
            for m in range(traj_local.shape[0]):
                traj_global[m] = convert_local_coords_to_global(traj_local[m],
                                                                starting_annotation['translation'],
                                                                starting_annotation['rotation'])

            preds.append(Prediction(instance=instance_tokens[n], sample=sample_tokens[n],
                                    prediction=traj_global, probabilities=probs_n))

        n_violations = 0
        cnt = 0

        for prediction in preds:
            map_name = self.helper.get_map_name_from_sample_token(prediction.sample)
            drivable_area = self.drivable_area_polygons[map_name]
            max_row, max_col = drivable_area.shape
            for mode in prediction.prediction:

                # Fit a cubic spline to the trajectory and interpolate with 200 points
                x_interpolated, y_interpolated = self.interpolate_path(mode, self.number_of_points)

                # x coordinate -> col, y coordinate -> row
                # Mask has already been flipped over y-axis
                index_row = (y_interpolated * self.pixels_per_meter).astype("int")
                index_col = (x_interpolated * self.pixels_per_meter).astype("int")

                row_out_of_bounds = np.any(index_row >= max_row) or np.any(index_row < 0)
                col_out_of_bounds = np.any(index_col >= max_col) or np.any(index_col < 0)
                out_of_bounds = row_out_of_bounds or col_out_of_bounds
                
                if out_of_bounds or not np.all(drivable_area[index_row, index_col]):
                    n_violations += 1
                cnt += 1

        return Float(n_violations / cnt)
