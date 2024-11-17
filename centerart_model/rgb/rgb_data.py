import pathlib
import json
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
from centerart_model.utils.configs import Directories, DEVICE
from centerart_model.utils.camera import CameraConventions
from centerart_model.sapien.scenes import GTData
import centerart_model.utils.data_utils as data_utils
import centerart_model.utils.se3_utils as se3_utils
from centerart_model.sgdf.training_deep_sgdf import load_embeddings


@dataclass
class RgbdPaths:
    rgb: List[pathlib.Path]
    depth_gt: List[pathlib.Path]
    depth_noisy: List[pathlib.Path]
    binary_masks: List[pathlib.Path]
    heatmap: List[pathlib.Path]
    segmentation: List[pathlib.Path]
    poses: List[pathlib.Path]
    camera_poses: List[pathlib.Path]
    info: pathlib.Path
    counter_range: List[pathlib.Path]


@dataclass
class RgbdPathsSingle:
    rgb: pathlib.Path
    depth: pathlib.Path
    binary_masks: pathlib.Path
    heatmap: pathlib.Path
    poses: pathlib.Path
    camera_poses: List[pathlib.Path]
    info: pathlib.Path


@dataclass
class RgbdDataNp:
    rgb: np.ndarray
    depth: np.ndarray
    heatmap: np.ndarray
    binary_masks: np.ndarray
    poses: np.ndarray
    info: List[dict]

    @classmethod
    def from_torch(
        cls, rgb_th: torch.Tensor, depth_th: torch.Tensor, heatmap_th: torch.Tensor
    ):
        rgb = data_utils.img_torch_to_np(rgb_th)
        depth = depth_th.squeeze(0).detach().cpu().numpy()
        heatmap = data_utils.img_torch_to_np(heatmap_th)
        return cls(rgb, depth, heatmap, None, None, None)


def get_rgbd_paths(mode: str, scene_idx: int, num_cams: int) -> RgbdPaths:
    root_path = Directories.RGBD / mode
    file_names = [f"{scene_idx:08d}_{cam_idx:04d}" for cam_idx in range(num_cams)]
    rgbd_paths = RgbdPaths(
        rgb=[root_path / "rgb" / f"{file_names[idx]}.png" for idx in range(num_cams)],
        depth_gt=[
            root_path / "depth_gt" / f"{file_names[idx]}.png" for idx in range(num_cams)
        ],
        depth_noisy=[
            root_path / "depth_noisy" / f"{file_names[idx]}.png"
            for idx in range(num_cams)
        ],
        poses=[
            root_path / "pose" / f"{file_names[idx]}.npy" for idx in range(num_cams)
        ],
        camera_poses=[
            root_path / "cam_pose" / f"{file_names[idx]}.npy" for idx in range(num_cams)
        ],
        counter_range=[
            root_path / "counter_range" / f"{file_names[idx]}.npy"
            for idx in range(num_cams)
        ],
        binary_masks=[
            root_path / "segm" / f"{file_names[idx]}_bm.json" for idx in range(num_cams)
        ],
        heatmap=[
            root_path / "segm" / f"{file_names[idx]}_heatmap.png"
            for idx in range(num_cams)
        ],
        segmentation=[
            root_path / "sem_seg" / f"{file_names[idx]}.png" for idx in range(num_cams)
        ],
        info=root_path / "segm" / f"{scene_idx:08d}_info.json",
    )
    return rgbd_paths


def check_exists(rgbd_paths: RgbdPaths) -> bool:
    rgb = all([p.exists() for p in rgbd_paths.rgb])
    depth_gt = all([p.exists() for p in rgbd_paths.depth_gt])
    depth_noisy = all([p.exists() for p in rgbd_paths.depth_noisy])
    poses = all([p.exists() for p in rgbd_paths.poses])
    camera_poses = all([p.exists() for p in rgbd_paths.camera_poses])
    binary_masks = all([p.exists() for p in rgbd_paths.binary_masks])
    heatmap = all([p.exists() for p in rgbd_paths.heatmap])
    counter_range = all([p.exists() for p in rgbd_paths.counter_range])
    info = rgbd_paths.info.exists()
    return (
        rgb
        and depth_gt
        and depth_noisy
        and poses
        and camera_poses
        and binary_masks
        and heatmap
        and info
        and counter_range
    )


def write_rgbd_data(
    rgbd_paths: RgbdPaths, gt_data_list: List[GTData], objs_info: List[dict]
):
    data_utils.save_dict_as_json(objs_info, rgbd_paths.info)
    for cam_idx, gt_data in enumerate(gt_data_list):
        data_utils.save_rgb(gt_data.rgb, rgbd_paths.rgb[cam_idx])
        data_utils.save_depth(gt_data.depth_gt, rgbd_paths.depth_gt[cam_idx])
        data_utils.save_depth(gt_data.depth_noisy, rgbd_paths.depth_noisy[cam_idx])
        data_utils.save_semantic(gt_data.segmentation, rgbd_paths.segmentation[cam_idx])
        np.save(rgbd_paths.poses[cam_idx], gt_data.camTposes, allow_pickle=False)
        np.save(
            rgbd_paths.camera_poses[cam_idx], gt_data.camera_pose, allow_pickle=False
        )
        np.save(
            rgbd_paths.counter_range[cam_idx],
            gt_data.counter_range,
            allow_pickle=False,
        )
        data_utils.save_binary_masks(
            gt_data.binary_masks, rgbd_paths.binary_masks[cam_idx]
        )
        data_utils.save_rgb(gt_data.heatmap, rgbd_paths.heatmap[cam_idx])
    return


class RGBDReader:
    def __init__(self, mode: str = "train") -> None:
        self.rgb_paths = self.get_rgb_paths(mode)
        self.rgb_paths_to_idx = {
            rgb_path: idx for idx, rgb_path in enumerate(self.rgb_paths)
        }
        return

    @staticmethod
    def get_rgb_paths(mode: str) -> List[pathlib.Path]:
        assert mode in ["train", "valid", "test"]
        rgb_paths = sorted((Directories.RGBD / mode / "rgb").iterdir())
        return rgb_paths

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> RgbdPathsSingle:
        rgb_path = self.rgb_paths[idx]
        scene_idx = int(rgb_path.stem.split("_")[0])
        info_path = rgb_path.parents[1] / "segm" / f"{scene_idx:08d}_info.json"
        paths = RgbdPathsSingle(
            rgb=rgb_path,
            depth=pathlib.Path(str(rgb_path).replace("/rgb/", "/depth/")),
            poses=pathlib.Path(
                str(rgb_path).replace("/rgb/", "/pose/").replace(".png", ".npy")
            ),
            camera_poses=pathlib.Path(
                str(rgb_path).replace("/rgb/", "/cam_pose/").replace(".png", ".npy"),
            ),
            binary_masks=pathlib.Path(
                str(rgb_path).replace("/rgb/", "/segm/").replace(".png", "_bm.json")
            ),
            heatmap=pathlib.Path(
                str(rgb_path).replace("/rgb/", "/segm/").replace(".png", "_heatmap.png")
            ),
            info=info_path,
        )
        return paths

    def get_data_np(self, idx: int) -> RgbdDataNp:
        paths = self[idx]
        rgb = data_utils.load_rgb_from_file(paths.rgb)
        depth = data_utils.load_depth_from_file(paths.depth)
        poses = np.load(paths.poses, allow_pickle=False)
        binary_masks = data_utils.load_binary_masks(paths.binary_masks)
        heatmap = data_utils.load_rgb_from_file(paths.heatmap)
        info = json.load(open(paths.info))
        # Transform obj poses from sapien camera frame to opencv camera frame
        poses = CameraConventions.opencv_T_robotics @ poses
        return RgbdDataNp(rgb, depth, heatmap, binary_masks, poses, info)

    def get_data_np_test(self, idx: int) -> RgbdDataNp:
        paths = self[idx]
        rgb = data_utils.load_rgb_from_file(paths.rgb)
        depth = data_utils.load_depth_from_file(paths.depth)
        poses = np.load(paths.poses, allow_pickle=False)
        binary_masks = data_utils.load_binary_masks(paths.binary_masks)
        heatmap = data_utils.load_rgb_from_file(paths.heatmap)
        info = json.load(open(paths.info))
        # Transform obj poses from sapien camera frame to opencv camera frame
        poses = CameraConventions.opencv_T_robotics @ poses
        return paths.rgb, RgbdDataNp(rgb, depth, heatmap, binary_masks, poses, info)

    def get_random(self) -> RgbdPathsSingle:
        return self[np.random.randint(len(self))]

    def get_idx_from_path(self, rgb_path: pathlib.Path) -> int:
        return self.rgb_paths_to_idx[rgb_path]


class RGBDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, embedding_ckpt: str, mode="train"):
        if mode not in ["train", "valid"]:
            raise ValueError("mode must be either 'train' or 'valid'")
        self.mode = mode
        # self.sgdf_paths_loader = SgdfPathsLoader(mode)
        self.rgbd_reader = RGBDReader(mode)
        # Needs to be on CPU for dataloading
        self.embeddings_matrix = load_embeddings(embedding_ckpt).cpu()
        self.code_length = self.embeddings_matrix.shape[1]
        return

    def __len__(self):
        return len(self.rgbd_reader)

    def __getitem__(self, idx):
        data_np = self.rgbd_reader.get_data_np(idx)
        pose_target, shape_target, joint_target, invariance_map = self.make_targets(
            data_np.poses, data_np.binary_masks, data_np.info
        )
        # Torch shape order: [channels, height, width]
        rgb = data_utils.img_np_to_torch(data_np.rgb)
        depth = torch.from_numpy(data_np.depth).unsqueeze(0)
        heatmap_target = data_utils.img_np_to_torch(data_np.heatmap)
        pose_target = torch.from_numpy(pose_target).permute(2, 0, 1)
        shape_target = torch.from_numpy(shape_target).permute(2, 0, 1)
        joint_target = torch.from_numpy(joint_target).permute(2, 0, 1)
        invariance_map = torch.from_numpy(invariance_map)
        return (
            rgb,
            depth,
            heatmap_target,
            pose_target,
            shape_target,
            joint_target,
            invariance_map,
        )

    def make_targets(
        self, poses: np.ndarray, binary_masks: np.ndarray, info: List[dict]
    ) -> Tuple[np.ndarray]:
        height = binary_masks.shape[1]
        width = binary_masks.shape[2]
        pose_target = np.zeros((height, width, 12), dtype=np.float32)
        invariance_map = np.zeros((height, width), dtype=np.uint8)
        shape_target = np.zeros((height, width, self.code_length), dtype=np.float32)
        joint_target = np.zeros((height, width, 1), dtype=np.float32)
        for obj_idx, obj_info in enumerate(info):
            sgdf_idx = obj_info["emb_index"]
            pose = poses[obj_idx]
            pose_flat = se3_utils.pose_4x4_to_flat(pose)
            shape = self.embeddings_matrix[sgdf_idx]
            joint = obj_info["joint_state"]
            invariance = 0  # TODO
            pose_target[binary_masks[obj_idx] == 1] = pose_flat
            shape_target[binary_masks[obj_idx] == 1] = shape
            joint_target[binary_masks[obj_idx] == 1] = joint
            invariance_map[binary_masks[obj_idx] == 1] = invariance
        # downsample
        pose_target = pose_target[::8, ::8, :]
        shape_target = shape_target[::8, ::8, :]
        joint_target = joint_target[::8, ::8, :]
        invariance_map = invariance_map[::8, ::8]
        return pose_target, shape_target, joint_target, invariance_map

    def get_data(self, idx, to_device):
        rgb, depth, heatmap_target, pose_target, shape_target, invariance_map = self[
            idx
        ]
        if to_device:
            rgb = data_utils.make_single_batch(rgb, 3).to(DEVICE)
            depth = data_utils.make_single_batch(depth, 1).to(DEVICE)
            heatmap_target = heatmap_target.to(DEVICE)
            pose_target = pose_target.to(DEVICE)
            shape_target = shape_target.to(DEVICE)
            invariance_map = invariance_map.to(DEVICE)
        return rgb, depth, heatmap_target, pose_target, shape_target, invariance_map
