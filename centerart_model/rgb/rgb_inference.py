import torch
import numpy as np
import open3d as o3d
from typing import List, Tuple
from dataclasses import dataclass
import centerart_model.rgb.heatmaps as heatmaps
import centerart_model.rgb.pose_utils as pose_utils
import centerart_model.sgdf.sgdf_inference as sgdf_inference
from centerart_model.rgb.training_centerart import load_rgb_model
from centerart_model.utils.configs import DEVICE
from centerart_model.umpnet.ump_net import UMPModel
from centerart_model.umpnet.ump_utils import get_position_action


@dataclass
class ObjPredictionTh:
    pose: torch.Tensor
    embedding: torch.Tensor
    joint_code: torch.Tensor
    bmask: np.ndarray


@dataclass
class FullObjPred:
    pose: np.ndarray
    bmask: np.ndarray
    pc_o3d: o3d.geometry.PointCloud
    grasp_poses: np.ndarray
    grasp_confidences: np.ndarray

    @classmethod
    def from_net_predictions(
        cls, rgb_pred: ObjPredictionTh, sgdf_pred: sgdf_inference.SGDFPrediction
    ):
        pose = rgb_pred.pose.detach().cpu().numpy()
        bmask = rgb_pred.bmask
        pc_o3d = sgdf_pred.pc_o3d.transform(pose)
        grasp_poses = pose @ sgdf_pred.grasp_poses
        grasp_confidences = sgdf_pred.grasp_confidences
        return cls(pose, bmask, pc_o3d, grasp_poses, grasp_confidences)


@dataclass
class PostprObjPred:
    pc_o3d: o3d.geometry.PointCloud
    grasp_poses: np.ndarray


def extract_obj_predictions(
    heatmap_th: torch.Tensor,
    posemap_th: torch.Tensor,
    shapemap_th: torch.Tensor,
    joint_code_th: torch.Tensor,
) -> List[ObjPredictionTh]:
    heatmap_np = heatmap_th.detach().cpu().numpy()
    peaks = heatmaps.extract_peaks_from_heatmap(heatmap_np)
    binary_masks = heatmaps.binary_masks_from_heatmap(heatmap_np, peaks, threshold=0.25)
    obj_predictions = []
    for i, peak in enumerate(peaks):
        peak_downsampled = peak // 8
        pose = posemap_th[:, peak_downsampled[0], peak_downsampled[1]]
        pose = pose_utils.pose_flat_to_matrix_th(pose)
        shape = shapemap_th[:, peak_downsampled[0], peak_downsampled[1]]
        joint_code = joint_code_th[:, peak_downsampled[0], peak_downsampled[1]]
        binary_mask = binary_masks[i]
        obj_predictions.append(ObjPredictionTh(pose, shape, joint_code, binary_mask))
    return obj_predictions


class RGBInference:
    def __init__(self, rgb_model: str, data_dict: dict):
        self.lit_rgb_model, rgb_specs = load_rgb_model(rgb_model)
        self.sgdf_inference_net = sgdf_inference.SGDFInference(
            rgb_specs["EmbeddingCkptPath"], data_dict
        )
        self.lit_rgb_model.eval()
        return

    def np_to_torch(
        self, rgb_uint8_np: np.ndarray, depth_np: np.ndarray
    ) -> Tuple[torch.Tensor]:
        rgb_np = rgb_uint8_np.astype(np.float32) / 255
        rgb_np = rgb_np.transpose((2, 0, 1))
        depth_np = depth_np.transpose((2, 0, 1))
        rgb_th = torch.from_numpy(rgb_np).to(DEVICE).unsqueeze(0)
        depth_th = torch.from_numpy(depth_np).to(DEVICE).unsqueeze(0)
        return rgb_th, depth_th

    def predict(
        self, rgb: torch.Tensor, depth: torch.Tensor
    ) -> Tuple[torch.Tensor, List[ObjPredictionTh]]:
        with torch.no_grad():
            (
                heatmap_out,
                abs_pose_out,
                latent_emb_out,
                joint_code_out,
            ) = self.lit_rgb_model(rgb, depth)
            heatmap_out = heatmap_out.squeeze(0)
            abs_pose_out = abs_pose_out.squeeze(0)
            latent_emb_out = latent_emb_out.squeeze(0)
            joint_code_out = joint_code_out.squeeze(0)
            obj_predictions = extract_obj_predictions(
                heatmap_out, abs_pose_out, latent_emb_out, joint_code_out
            )
        return heatmap_out, obj_predictions

    def get_full_predictions(
        self, rgb_uint8_np: np.ndarray, depth_np: np.ndarray
    ) -> Tuple[np.ndarray, List[FullObjPred]]:
        rgb_th, depth_th = self.np_to_torch(rgb_uint8_np, depth_np)
        heatmap_out, obj_predictions = self.predict(rgb_th, depth_th)
        sgdf_preds = [
            self.sgdf_inference_net.predict_reconstruction(
                pred.embedding, pred.joint_code
            )
            for pred in obj_predictions
        ]
        full_preds = [
            FullObjPred.from_net_predictions(rgb_pred, sgdf_pred)
            for rgb_pred, sgdf_pred in zip(obj_predictions, sgdf_preds)
        ]
        heatmap_np = heatmap_out.detach().cpu().numpy()
        return heatmap_np, full_preds

    def get_raw_shape_predictions(
        self, rgb_uint8_np: np.ndarray, depth_np: np.ndarray
    ) -> Tuple[np.ndarray, List[FullObjPred]]:
        rgb_th, depth_th = self.np_to_torch(rgb_uint8_np, depth_np)
        heatmap_out, obj_predictions = self.predict(rgb_th, depth_th)
        sgdf_preds = [
            self.sgdf_inference_net.predict_reconstruction(
                pred.embedding, pred.joint_code
            )
            for pred in obj_predictions
        ]
        return sgdf_preds


class UMPNetInference:
    def __init__(self):
        self.device = torch.device(f"cuda:0")
        self.ump_model = UMPModel(num_directions=64, model_type="sgn_mag")
        self.ump_model = self.ump_model.to(self.device, self.device)
        checkpoint = torch.load("ckpt_ump/latest.pth", map_location=self.device)
        self.ump_model.pos_model.load_state_dict(checkpoint["pos_state_dict"])
        print("==> pos model loaded")
        self.ump_model.dir_model.load_state_dict(checkpoint["dir_state_dict"])
        print("==> dir model loaded")

    def postprocess_affordance_map(self, affordance_map, boundary_size=20):
        """
        Removes boundary pixels from the affordance map by setting their values to zero.

        Parameters:
            affordance_map (numpy.ndarray): The affordance map with shape (480, 640).
            boundary_size (int): The number of pixels to remove from each boundary. Default is 10.

        Returns:
            numpy.ndarray: The affordance map with boundary pixels set to zero.
        """
        affordance_map[:boundary_size, :] = 0  # Top boundary
        affordance_map[-boundary_size:, :] = 0  # Bottom boundary
        affordance_map[:, :boundary_size] = 0  # Left boundary
        affordance_map[:, -boundary_size:] = 0  # Right boundary

        return affordance_map

    def predict(self, observation):
        self.ump_model.eval()
        torch.set_grad_enabled(False)
        position_affordance = self.ump_model.get_position_affordance([observation])[0]
        position_affordance = self.postprocess_affordance_map(position_affordance)
        action, score = get_position_action(
            position_affordance, epsilon=0, image=observation, prev_actions=list()
        )
        return action, score
