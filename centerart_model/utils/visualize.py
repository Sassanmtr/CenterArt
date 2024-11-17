from typing import List, Optional, Tuple
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
import copy

if not os.uname()[1].startswith("rlgpu"):
    import rerun as rr

from centerart_model.utils.camera import ZED2_RESOLUTION_HALF
from centerart_model.gdf import gdf_utils
import centerart_model.rgb.heatmaps as heatmaps
import centerart_model.rgb.pose_utils as pose_utils
import centerart_model.rgb.rgb_data as rgb_data
from centerart_model.sgdf.sgdf_inference import SGDFPrediction
from centerart_model.rgb.rgb_inference import FullObjPred, PostprObjPred
from simnet.lib.transform import Pose


def color_pcd(pcd_np: np.ndarray, color: Optional[Tuple] = None):
    """
    Args:
        pcd: numpy array of shape (N, 3)
    """
    if not color:
        min_z = pcd_np[:, 2].min()
        max_z = pcd_np[:, 2].max()
        cmap_norm = mpl.colors.Normalize(vmin=min_z, vmax=max_z)
        #'hsv' is changeable to any name as stated here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        point_colors = plt.get_cmap("jet")(cmap_norm(pcd_np[:, 2]))[
            :, :3
        ]  # We don't need alpha
    else:
        assert len(color) == 3
        N, _ = pcd_np.shape
        point_colors = np.tile(color, (N, 1))
    return point_colors


class Open3DOfflineRenderer:
    def __init__(
        self,
        width: int = ZED2_RESOLUTION_HALF[0],
        height: int = ZED2_RESOLUTION_HALF[1],
    ) -> None:
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.reset()

    def add_grasps(
        self, grasps: List[trimesh.Trimesh], base_name="grasp", color=[0.0, 1.0, 0.0]
    ):
        grasp_mesh: trimesh.Trimesh
        for idx, grasp_trimesh in enumerate(grasps):
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            name = f"{base_name}_{idx}"
            self.renderer.scene.remove_geometry(name)
            grasp_o3d = copy.deepcopy(grasp_trimesh.as_open3d)
            grasp_o3d.paint_uniform_color(color)
            self.renderer.scene.add_geometry(name, grasp_o3d, mat)

    def add_pointcloud(
        self, pcd_np, name: str = "object_pc", color: Optional[Tuple] = None
    ):
        """
        Places a point cloud in the scene, if there is already one with the same name, replaces it
        """
        if len(pcd_np) == 0:  # Don't add empty pcs
            return
        self.renderer.scene.remove_geometry(name)
        point_colors = color_pcd(pcd_np, color=color)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np))
        # Set colors
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        # Default material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Add to scene
        self.renderer.scene.add_geometry(name, pcd, mat)

    def reset(self):
        self.renderer.scene.camera.look_at(
            [0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.0, 0.0, 1.0]
        )
        # self.renderer.scene.set_background()

    def render(self):
        """
        Renders the scene to a numpy array
        """
        img_o3d = self.renderer.render_to_image()
        return np.asarray(img_o3d)


class RerunViewer:
    def __init__(self, addr=None):
        if addr is None:
            rr.init("centerart")
            rr.spawn()
        else:
            rr.connect(addr)
        return

    @staticmethod
    def visualize_prediction(preds_raw: FullObjPred, preds_pp: PostprObjPred, idx: int):
        RerunViewer.add_o3d_pointcloud(
            f"vis/shapes_raw/{idx}", preds_raw.pc_o3d, radii=0.003
        )
        RerunViewer.add_o3d_pointcloud(
            f"vis/shapes_icp/{idx}", preds_pp.pc_o3d, radii=0.003
        )
        RerunViewer.add_grasps(f"vis/grasps/{idx}", preds_pp.grasp_poses)
        return

    @staticmethod
    def vis_sgdf_prediction(name: str, data: SGDFPrediction, num_grasps: int = 10):
        grasps_idx = np.random.choice(
            data.grasp_poses.shape[0], size=num_grasps, replace=False
        )
        RerunViewer.add_o3d_pointcloud(f"vis/{name}/pc", data.pc_o3d, radii=0.002)
        RerunViewer.add_grasps(f"vis/{name}/grasps", data.grasp_poses[grasps_idx])

    @staticmethod
    def vis_rgbd_data(data: rgb_data.RgbdDataNp):
        RerunViewer.add_rgb("rgb", data.rgb)
        RerunViewer.add_depth("depth", data.depth)
        RerunViewer.add_heatmap("heatmap", data.rgb, data.heatmap)
        if data.poses is not None:
            RerunViewer.add_poses("poses", data.rgb, data.poses)
        return

    @staticmethod
    def add_heatmap(name: str, rgb: np.ndarray, heatmap: np.ndarray):
        net_heatmap_vis = heatmaps.visualize_heatmap(rgb, heatmap, with_peaks=True)
        rr.log_image(name, net_heatmap_vis)

    @staticmethod
    def add_poses(name: str, rgb: np.ndarray, poses: np.ndarray):
        _poses = [Pose(camera_T_object=pose) for pose in poses]
        poses_vis = pose_utils.visualize_poses(rgb, _poses)
        rr.log_image(name, poses_vis)

    @staticmethod
    def add_o3d_pointcloud(name, pointcloud, radii=None):
        points = np.asanyarray(pointcloud.points)
        colors = np.asanyarray(pointcloud.colors) if pointcloud.has_colors() else None
        colors_uint8 = (
            (colors * 255).astype(np.uint8) if pointcloud.has_colors() else None
        )
        rr.log_points(name, positions=points, colors=colors_uint8, radii=radii)
        return

    @staticmethod
    def add_pointcloud(name, points, colors=None, radii=None):
        rr.log_points(name, positions=points, colors=colors, radii=radii)
        return

    @staticmethod
    def add_trimeshes(name, trimeshes):
        for i, mesh in enumerate(trimeshes):
            rr.log_mesh(
                name + f"_{i}",
                positions=mesh.vertices,
                indices=mesh.faces,
                normals=mesh.vertex_normals,
                vertex_colors=mesh.visual.vertex_colors,
            )
        return

    @staticmethod
    def add_grasps(name, grasp_poses, color=[0.0, 1.0, 0.0]):
        grasps_trimesh = gdf_utils.create_markers_multiple(
            grasp_poses, color, axis_frame=True, highlight_first=True
        )
        RerunViewer.add_trimeshes(name, grasps_trimesh)
        return

    @staticmethod
    def add_axis(name, pose, size=0.04):
        mesh = trimesh.creation.axis(origin_size=size, transform=pose)
        RerunViewer.add_trimeshes(name, [mesh])
        return

    @staticmethod
    def add_grid_bounding_box(name, grid_dim):
        half_size = [
            grid_dim,
            grid_dim,
            grid_dim,
        ]  # This version of rerun has a bug with half_size
        rr.log_obb(
            name, half_size=half_size, position=[0, 0, 0], rotation_q=[0, 0, 0, 1]
        )
        return

    @staticmethod
    def add_sphere():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100.0)
        sphere.paint_uniform_color([0.8, 0.8, 0.8])
        rr.log_mesh(
            "vis/background",
            positions=sphere.vertices,
            indices=sphere.triangles,
            vertex_colors=sphere.vertex_colors,
        )

    @staticmethod
    def add_rgb(name, rgb):
        rr.log_image(name, rgb)

    @staticmethod
    def add_depth(name, detph):
        rr.log_depth_image(name, detph)

    @staticmethod
    def clear():
        rr.log_cleared("pointclouds", recursive=True)
        rr.log_cleared("pointclouds_giga", recursive=True)
        rr.log_cleared("vis", recursive=True)
        return
