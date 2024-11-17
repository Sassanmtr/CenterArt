import torch
import numpy as np
import open3d as o3d
from typing import Tuple
from dataclasses import dataclass
import centerart_model.utils.data_utils as data_utils
from centerart_model.utils.configs import DEVICE
from centerart_model.sdf.grid import Grid3D
from centerart_model.sgdf.training_deep_sgdf import load_sgdf_model


@dataclass
class SGDFPrediction:
    pc_o3d: o3d.geometry.PointCloud
    grasp_poses: np.ndarray
    grasp_confidences: np.ndarray

    @classmethod
    def from_torch(
        cls, pc_th: torch.Tensor, grasps_th: torch.Tensor, confs_th: torch.Tensor
    ):
        pc, grasps, confs = data_utils.th_to_np(pc_th, grasps_th, confs_th)
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        # Transform from palm to hand frame
        grasps[..., :3, 3] -= 0.0634 * grasps[..., :3, 2]
        return cls(pc_o3d, grasps, confs)


class SGDFInference:
    def __init__(self, sgdf_model: str, data_dict: dict):
        self.lit_model, sgdf_specs = load_sgdf_model(sgdf_model)
        self.lit_model.eval()
        self.sgdf_specs = sgdf_specs
        self.data_dict = data_dict
        return

    def get_embedding_from_name(self, obj_name):
        idx_th = torch.tensor(self.data_dict[obj_name]["object_index"]).to(DEVICE)
        print("embedding index: ", idx_th)
        shape_embedding = self.lit_model.embeddings(idx_th)
        shape_embedding = torch.cat([shape_embedding])
        return shape_embedding.unsqueeze(0)

    def get_joint_code_from_name(self, obj_name):
        joint_state = float(self.data_dict[obj_name]["joint_state"])
        joint_lower_limit = self.data_dict[obj_name]["joint_lower_limit"]
        joint_upper_limit = self.data_dict[obj_name]["joint_upper_limit"]
        normalized_joint_state = (joint_state - joint_lower_limit) / (
            joint_upper_limit - joint_lower_limit
        )
        joint_code = torch.tensor([normalized_joint_state]).to(DEVICE)
        return joint_code.unsqueeze(0)

    def predict_reconstruction(
        self,
        embeddings: torch.Tensor,
        joint_code: torch.Tensor,
        grid_density: int = 128,  # TODO: Change to 64
        grid_half_dim: float = 1.5,
    ) -> SGDFPrediction:
        grid_3d = Grid3D(
            density=grid_density, grid_dim=grid_half_dim, device=str(DEVICE)
        )
        sdf_th, grasp_poses_th, confidence_th = self.predict(
            grid_3d.points, embeddings, joint_code
        )
        pointcloud_th = grid_3d.get_masked_surface_iso(sdf_th, threshold=0.001)
        sgdf_prediction = SGDFPrediction.from_torch(
            pointcloud_th, grasp_poses_th, confidence_th
        )
        return sgdf_prediction

    def predict(
        self,
        points: torch.Tensor,
        shape_embedding: torch.Tensor,
        joint_code: torch.Tensor,
        distance_threshold: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Note: Using no grad here saves memory but disallows us to use the iso-surface projection
        with torch.no_grad():
            if len(points.shape) == 2:
                points = points.unsqueeze(0)
            if len(shape_embedding.shape) == 1:
                shape_embedding = shape_embedding.unsqueeze(0)
            embeddings_expanded = shape_embedding.unsqueeze(1).expand(
                -1, points.shape[1], -1
            )
            joint_code_expanded = (
                joint_code.view(1, -1).unsqueeze(1).expand(-1, points.shape[1], -1)
            )
            sdf, grasp_poses = self.lit_model.predict(
                points, embeddings_expanded, joint_code_expanded
            )
            distance = torch.linalg.vector_norm(
                grasp_poses[:, :3, 3] - points.squeeze(0), dim=-1
            )

            # Filter out grasps that are too far away
            indeces = torch.nonzero(distance < distance_threshold).squeeze(-1)
            distance = distance[indeces]
            grasp_poses = grasp_poses[indeces]

            # The confidence is the negative distance
            confidence = -distance
        return sdf, grasp_poses, confidence

    def predict_th2np(
        self,
        points: torch.Tensor,
        shape_embedding: torch.Tensor,
        distance_threshold: float = 0.02,
    ) -> Tuple[np.ndarray]:
        sdf, grasp_poses, confidence = self.predict(
            points, shape_embedding, distance_threshold
        )
        return data_utils.th_to_np(sdf, grasp_poses, confidence)

    def predict_np2th(
        self, points: np.ndarray, shape_embedding_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.from_numpy(points, dtype=torch.float32)
        points = points.to(DEVICE).unsqueeze(0)
        shape_embedding = torch.from_numpy(shape_embedding_np).to(DEVICE).unsqueeze(0)
        sdf, grasp_poses, confidence = self.predict(points, shape_embedding)
        return sdf, grasp_poses, confidence

    def predict_np2np(
        self, points: np.ndarray, shape_embedding_np: np.ndarray
    ) -> Tuple[np.ndarray]:
        sdf, grasp_poses, confidence = self.predict_np2th(points, shape_embedding_np)
        return data_utils.th_to_np(sdf, grasp_poses, confidence)

    def get_embeddings_np(self) -> np.ndarray:
        embeddings_np = (
            self.lit_model.embeddings.weight.detach().cpu().numpy()
        )  # Number of Objects x Code Dimensions
        return embeddings_np

    def extract_reconstruction(
        self, predictions, joint_code, grid_density=64, grid_dim=0.25
    ):
        """
        Args:
            predictions: List of ObjectPredictionTh
        """
        for pred in predictions:
            grid_3d = Grid3D(
                density=grid_density, grid_dim=grid_dim, device=str(DEVICE)
            )
            sdf_th, grasp_poses_th, confidence_th = self.predict(
                grid_3d.points, pred.shape, joint_code
            )
            pointcloud_th = grid_3d.get_masked_surface_iso(sdf_th, threshold=0.001)
            pointcloud_th = pointcloud_th @ pred.pose[:3, :3].T + pred.pose[:3, 3]
            grasp_poses_th = pred.pose @ grasp_poses_th
            pred.pointcloud = pointcloud_th
            pred.grasp_poses = grasp_poses_th
            pred.grasp_confidences = confidence_th
        return predictions
