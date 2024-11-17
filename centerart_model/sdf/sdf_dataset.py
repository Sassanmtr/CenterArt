import os
import torch
import numpy as np
from collections import OrderedDict
from centerart_model.gdf.gdf_utils import generate_gt_v


def data_reader(idx, idx_path, data_config):
    # Joint code
    joint_data = list(data_config.values())[idx]
    joint_state = float(joint_data["joint_state"])
    joint_upper_limit = float(joint_data["joint_upper_limit"])
    joint_lower_limit = float(joint_data["joint_lower_limit"])
    obj_index = joint_data["object_index"]
    joint_code = np.array(
        [(joint_state - joint_lower_limit) / (joint_upper_limit - joint_lower_limit)]
    )
    # SDF data
    sdf_values_path = os.path.join(idx_path, "sdf_values.npy")
    sdf_points_path = os.path.join(idx_path, "sdf_points.npy")
    sdf_values = np.load(sdf_values_path)
    sdf_points = np.load(sdf_points_path)
    # Grasp data
    grasps_path = os.path.join(idx_path, "grasps.npy")
    grasps = np.load(grasps_path)
    grasp_vecs = generate_gt_v(sdf_points, grasps)
    return obj_index, sdf_points, sdf_values, grasp_vecs, joint_code


class BaseDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self, points_per_obj, dataset_dir, dataset_config, split_config, mode="train"
    ) -> None:
        super().__init__()
        if mode not in ["train", "valid"]:
            raise ValueError("mode must be either 'train' or 'valid'")
        self.mode = mode
        self.points_per_obj = points_per_obj
        self.dataset_dir = dataset_dir
        self.dataset_config = OrderedDict(dataset_config)
        self.split_config = split_config
        return

    def __len__(self):
        return len(os.listdir(self.dataset_dir))


class SGDFDataset(BaseDataset):  # type: ignore
    def __init__(
        self, points_per_obj, dataset_dir, dataset_config, split_config, mode="train"
    ):
        super().__init__(
            points_per_obj, dataset_dir, dataset_config, split_config, mode
        )
        return

    def __getitem__(self, idx):
        data_file = self.split_config[self.mode][idx]
        # data_file = list(self.dataset_config.keys())[idx]
        idx_path = os.path.join(self.dataset_dir, data_file)
        keys_list = list(self.dataset_config.keys())
        new_index = keys_list.index(data_file)
        obj_index, points, sdf, grasps, joint_code = data_reader(
            new_index, idx_path, self.dataset_config
        )
        points_th = torch.tensor(points, dtype=torch.float32)
        sdf_th = torch.tensor(sdf, dtype=torch.float32)
        grasps = np.array(grasps)
        grasps_th = torch.tensor(grasps, dtype=torch.float32)
        joint_code_th = torch.tensor(joint_code, dtype=torch.float32)
        return obj_index, points_th, sdf_th, grasps_th, joint_code_th

    def get_num_objects(self):
        return len(os.listdir(self.dataset_dir))
