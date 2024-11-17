from pathlib import Path
import time
import numpy as np
import open3d as o3d
import spatialmath as sm
from scipy.spatial.transform import Rotation
from typing import List
from centerart_model.sapien.sapien_utils import Obs
import centerart_model.utils.data_utils as data_utils
from centerart_model.rgb.rgb_data import RgbdDataNp
from centerart_model.utils.visualize import RerunViewer
from centerart_model.rgb.rgb_inference import (
    RGBInference,
    UMPNetInference,
)
from centerart_model.umpnet import ump_utils
from centerart_model.rgb.pred_postprocessing import postprocess_predictions
from centerart_model.utils.camera import CameraConventions
import matplotlib.pyplot as plt


class CenterArtPipeline:
    def __init__(
        self,
        rgb_model: str = "n90qyp90",
        data_dict: dict = None,
        seed: int = 123,
        use_icp: bool = False,
        visualize: bool = True,
    ):
        self.use_icp = use_icp
        self.data_dict = data_dict
        self.rgb_net = RGBInference(rgb_model, data_dict)
        self.rng = np.random.default_rng(seed)
        self.visualize = visualize
        return

    def _centerart_raw_shape_predictions(self, obs: Obs):
        # Predict
        start_time = time.time()
        sgdf_preds = self.rgb_net.get_raw_shape_predictions(
            obs.camera.rgb, obs.camera.depth
        )
        inference_time = time.time() - start_time
        return sgdf_preds, inference_time

    def _centerart_predictions(self, obs: Obs, confidence_map=None, num_grasps=10):
        # Predict
        start_time = time.time()
        heatmap_out, full_preds = self.rgb_net.get_full_predictions(
            obs.camera.rgb, obs.camera.depth
        )

        # Postprocessing
        poses = [pred.pose for pred in full_preds]
        rgbd_data = RgbdDataNp(
            obs.camera.rgb, obs.camera.depth, heatmap_out, None, poses, None
        )
        postpr_preds, full_pcd = postprocess_predictions(
            rgbd_data,
            full_preds,
            num_grasps,
            self.use_icp,
            confidence_map,
        )
        inference_time = time.time() - start_time
        # random.shuffle(postpr_preds)

        # Visualize
        if self.visualize:
            RerunViewer()
            # qualysis_addr = "127.0.0.1:9876"
            RerunViewer.clear()
            # create large white sphere to serve as background
            RerunViewer.add_sphere()
            RerunViewer.vis_rgbd_data(rgbd_data)
            RerunViewer.add_o3d_pointcloud("vis/full_pcd", full_pcd, radii=0.0015)
            for idx in range(len(postpr_preds)):
                RerunViewer.visualize_prediction(
                    full_preds[idx], postpr_preds[idx], idx
                )
            # input("Press enter to continue...")

        return postpr_preds, full_pcd, inference_time

    def _extract_grasp(self, chosen_grasp, wTcam) -> sm.SE3:
        camTgrasp_1 = sm.SE3(chosen_grasp, check=False)
        camTgrasp_2 = sm.SE3(chosen_grasp, check=False) * sm.SE3.Rz(-np.pi)
        handTcam = sm.SE3.Rz(np.pi / 2)  # Assumes gripper camera
        handTgrasp_1 = handTcam * camTgrasp_1
        handTgrasp_2 = handTcam * camTgrasp_2
        error_rpy_1 = sm.smb.tr2rpy(
            handTgrasp_1.R, unit="rad", order="zyx", check=False
        )
        error_rpy_2 = sm.smb.tr2rpy(
            handTgrasp_2.R, unit="rad", order="zyx", check=False
        )
        camTgrasp = (
            camTgrasp_1
            if np.linalg.norm(error_rpy_1) <= np.linalg.norm(error_rpy_2)
            else camTgrasp_2
        )
        from centerart_model.utils.camera import CameraConventions

        transform = sm.SE3(CameraConventions.robotics_T_opencv, check=False)
        # wTeegoal = wTcam * transform * camTgrasp
        wTeegoal = wTcam * camTgrasp

        def get_forward_vector(grasp_pose, distance):
            """
            Calculate the forward vector based on a position, orientation, and distance.
            """
            rotation_matrix = grasp_pose.R
            # Define the forward vector (assuming Z-axis is the forward direction)
            forward_vector = np.array([0, 0, 1])
            rotated_forward_vector = rotation_matrix @ forward_vector
            forward_vector_scaled = (
                distance
                * rotated_forward_vector
                / np.linalg.norm(rotated_forward_vector)
            )
            res = np.eye(4)
            res[:3, :3] = rotation_matrix
            res[:3, 3] = grasp_pose.t + forward_vector_scaled
            res = sm.SE3(res, check=False)
            return res

        wTeegoal = get_forward_vector(wTeegoal, 0.05)

        return wTeegoal

    def predict_shape(self, obs: Obs) -> np.ndarray:
        predictions, _, _ = self._centerart_predictions(obs)
        pcs_list = []
        combined_pcs = o3d.geometry.PointCloud()
        for i, pred in enumerate(predictions):
            pred.pc_o3d.transform(obs.camera_pose[i].reshape(4, 4))
            pcs_list.append(np.asarray(pred.pc_o3d.points))
            combined_pcs += pred.pc_o3d
        return np.asarray(combined_pcs.points), pcs_list

    def predict_grasp(self, obs: Obs, confidence_map=None) -> List[sm.SE3]:
        predictions, _, _ = self._centerart_predictions(obs, confidence_map)
        # Get the best grasp for each object
        best_grasps = [
            pred.grasp_poses[0] for pred in predictions if len(pred.grasp_poses) > 0
        ]
        wTeegoal_list = [
            self._extract_grasp(grasp, obs.camera_pose) for grasp in best_grasps
        ]
        return wTeegoal_list

    def predict_shape_and_grasps(self, obs: Obs):
        predictions, _, _ = self._centerart_predictions(obs, num_grasps=10)
        # Get shape
        pcs_list = []
        combined_pcs = o3d.geometry.PointCloud()
        for pred in predictions:
            pred.pc_o3d.transform(obs.camera_pose.A)
            pcs_list.append(pred.pc_o3d)
            combined_pcs += pred.pc_o3d
        # Get the best grasp for each object
        all_grasps = [
            grasp
            for pred in predictions
            for grasp in pred.grasp_poses
            if len(pred.grasp_poses) > 0
        ]
        wTeegoal_list_all = [
            self._extract_grasp(grasp, obs.camera_pose) for grasp in all_grasps
        ]

        best_grasps = [
            pred.grasp_poses[0] for pred in predictions if len(pred.grasp_poses) > 0
        ]
        wTeegoal_list_best = [
            self._extract_grasp(grasp, obs.camera_pose) for grasp in best_grasps
        ]
        return combined_pcs, pcs_list, wTeegoal_list_all, wTeegoal_list_best



class GTGraspPipeline:
    # Pipeline with ground-truth grasps (for debugging purposes)
    def __init__(
        self,
        scene_obj_info: dict = None,
        grasp_path: Path = None,
        seed: int = 123,
    ):
        self.scene_obj_info = scene_obj_info
        self.grasp_path = grasp_path
        self.rng = np.random.default_rng(seed)
        return

    def udpate_scene_obj_info(self, scene_obj_info: dict):
        self.scene_obj_info = scene_obj_info
        return

    def predict_grasp(self, obs: Obs) -> List[sm.SE3]:
        wTeegoal_list = []
        for _, obj_info in self.scene_obj_info.items():
            # Load the corresponding grasp labels
            print("obj_info: ", obj_info)
            self.grasp_path = (
                Path.cwd() / "datasets" / "decoder" / obj_info["name"] / "grasps.npy"
            )
            predictions = np.load(self.grasp_path)
            # Get the best grasp for each object
            raw_grasp = predictions[0]
            best_grasp = self._extract_grasp(raw_grasp, obj_info["pose"])
            # TODO: Remove the offset
            best_grasp.t -= np.array([0.7, 0.4, 0])
            wTeegoal_list.append(best_grasp)

        return wTeegoal_list

    def _extract_grasp(self, chosen_grasp, wTobj) -> sm.SE3:
        wTeegoal = wTobj @ chosen_grasp
        wTeegoal = sm.SE3(wTeegoal, check=False)
        return wTeegoal

class UMPNetPipeline:
    def __init__(
        self,
        data_dir: Path = None,
        seed: int = 123,
        visualize: bool = True,
    ):
        self.ump_net = UMPNetInference()
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed)
        self.visualize = visualize
        return

    def get_observations(self, idx):
        target_size = (640, 480)
        rgb_path = self.data_dir / "rgb" / f"{idx}.png"
        depth_path = self.data_dir / "depth" / f"{idx}.png"
        segmentation_path = self.data_dir / "sem_seg" / f"{idx}.png"
        camera_pose_path = self.data_dir / "cam_pose" / f"{idx}.npy"
        self.camera_pose = np.load(camera_pose_path)
        # remove the last row
        self.camera_pose = self.camera_pose[:3, :]
        self.rgb = data_utils.load_rgb_from_file(rgb_path)
        self.depth = data_utils.load_depth_from_file(depth_path)
        self.segmentation = data_utils.load_semantic_from_file(segmentation_path)
        self.rgb = self.rgb[:, :, :3].astype(np.float32) / 255.0
        test_intrinsics = np.array(
            [[530.0, 0.0, 480.0], [0.0, 530.0, 256.0], [0.0, 0.0, 1.0]]
        )
        self.cam_pose4x4 = np.eye(4)
        self.cam_pose4x4[:3, :] = self.camera_pose
        self.cam_pose4x4 = self.cam_pose4x4 @ CameraConventions.robotics_T_opencv
        (
            self.xyz_pts,
            self.color_pts,
            self.segmentation_pts,
        ) = ump_utils.get_ump_pointcloud(
            self.depth,
            self.rgb,
            self.segmentation,
            test_intrinsics,
            self.cam_pose4x4[:3, :],
        )

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.xyz_pts)
        self.pcd.colors = o3d.utility.Vector3dVector(self.color_pts)
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        self.pcd.orient_normals_towards_camera_location(
            camera_location=self.cam_pose4x4[:3, 3]
        )
        self.normals = np.array(self.pcd.normals)

        # crop (remove plane)
        self.cropped_pcd = self.pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([-np.inf, -np.inf, 0.1]),
                max_bound=np.array([np.inf, np.inf, np.inf]),
            )
        )
        num_points = np.asarray(self.cropped_pcd.points).shape[0]
        selected_idx = np.random.choice(
            num_points, min(num_points, 10000), replace=False
        )
        self.xyz_pts_sample = np.asarray(self.cropped_pcd.points)[selected_idx, :]
        self.color_pts_sample = np.asarray(self.cropped_pcd.colors)[selected_idx, :]
        self.normal_pts_sample = np.asarray(self.cropped_pcd.normals)[selected_idx, :]
        self.pcd = np.concatenate(
            [self.xyz_pts_sample, self.color_pts_sample, self.normal_pts_sample], axis=1
        ).astype(np.float32)
        # # Visualization for Debugging
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        image = np.concatenate(
            [
                self.xyz_pts.reshape(self.rgb.shape),
                self.rgb,
                self.normals.reshape(self.rgb.shape),
                self.depth[:, :, np.newaxis],
            ],
            axis=2,
        ).astype(np.float32)
        return image

    def predict_grasp(self, idx):
        image = self.get_observations(idx)
        action, score = self.ump_net.predict(image)
        print("action: ", action)
        pixel_index = np.ravel_multi_index(action, self.depth.shape)
        position_start = self.xyz_pts[pixel_index]
        surface_normal = self.normals[pixel_index]
        rotation_start = Rotation.from_euler("xyz", surface_normal)
        rotation_matrix_start = rotation_start.as_matrix()
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = rotation_matrix_start
        grasp_pose[:3, 3] = position_start
        grasp_pose = sm.SE3(grasp_pose, check=False)
        print("Grasp position: ", grasp_pose.t)
        return [grasp_pose]



def visualize_point_cloud(point_cloud):
    """
    Visualize a point cloud.

    Parameters:
        point_cloud (numpy.ndarray): The point cloud with shape (N, 3).

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Plot the point cloud
    ax.scatter(x, y, z, c=z, cmap="viridis", marker=".")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

