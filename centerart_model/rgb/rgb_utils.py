from typing import Optional
import torch
import numpy as np


class ObjectPredictionTh:
    def __init__(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        binary_mask: Optional[np.ndarray] = None,
        invariance: Optional[int] = None,
        pointcloud: Optional[torch.Tensor] = None,
        grasp_poses: Optional[torch.Tensor] = None,
        grasp_confidences: Optional[torch.Tensor] = None,
    ):
        """
        pose: 4x4 matrix in camera frame (torch.tensor)
        shape_code: 32 (torch.tensor)
        binary_mask: (np.array)
        """
        self.pose = pose
        self.shape = shape
        self.binary_mask = binary_mask
        self.invariance = invariance
        self.pointcloud = pointcloud
        self.grasp_poses = grasp_poses
        self.grasp_confidences = grasp_confidences
        return
