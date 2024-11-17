import torch
import pathlib
import numpy as np
import spatialmath as sm
import open3d as o3d
from dataclasses import dataclass
from centerart_model.utils.camera import CameraParamsIdeal

ZED2_PARAMS = CameraParamsIdeal(width=1920, height=1080, f_xy=1060)
ZED2HALF_PARAMS = CameraParamsIdeal(width=960, height=512, f_xy=530)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Directories:
    DATA = pathlib.Path.home() / "CenterArt/datasets"
    TEXTURES = DATA / "textures"
    FRANKA = DATA / "franka"
    GRASPS = DATA / "centerart_model" / "grasps"
    SGDF = DATA / "centerart_model" / "sgdf"
    RGBD = DATA / "centerart_model" / "rgbd"
    ROOT = pathlib.Path(__file__).parent.parent
    CONFIGS = ROOT / "configs"


@dataclass
class GripperKinTree:
    hand = np.eye(4)
    leftfinger = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.04],
            [0.0, 0.0, 1.0, 0.0584],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rightfinger = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.0584],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@dataclass
class WSConfigs:
    table_half_size = np.array([0.5, 0.5, 0.001])
    table_position = np.array([0.0, 0.0, 0.2 - table_half_size[2]])

    @property
    def ws_aabb(self) -> o3d.geometry.AxisAlignedBoundingBox:
        table_min_bound = self.table_position - self.table_half_size
        table_max_bound = self.table_position + self.table_half_size
        ws_min_bound = np.array([table_min_bound[0], table_min_bound[1], 0.0])
        ws_max_bound = np.array([table_max_bound[0], table_max_bound[1], 0.5])
        ws_aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=ws_min_bound, max_bound=ws_max_bound
        )
        return ws_aabb

    @property
    def ws_obb(self) -> o3d.geometry.OrientedBoundingBox:
        obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
            self.ws_aabb
        )
        return obb


COLOR_PALETTE = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
]

CAM_POSE_REAL = sm.SE3(
    np.array(
        [
            [-0.00288795, -0.84104389, 0.54095919, 0.46337467],
            [-0.99974617, -0.00965905, -0.02035441, -0.07190939],
            [0.0223441, -0.54088066, -0.84080251, 1.06529043],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    check=False,
)
