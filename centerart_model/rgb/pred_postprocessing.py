import copy
import numpy as np
import open3d as o3d
import spatialmath as sm
import mplib.pymp.fcl as fcl
from typing import List, Tuple
from centerart_model.utils.configs import ZED2HALF_PARAMS
from centerart_model.utils.camera import CameraParamsIdeal
from centerart_model.rgb.rgb_data import RgbdDataNp
from centerart_model.rgb.rgb_inference import FullObjPred, PostprObjPred

def get_full_pcd(
    rgb_data: RgbdDataNp,
    confidence_map: np.ndarray = None,
    project_valid_depth_only=False,
    cam_params: CameraParamsIdeal = ZED2HALF_PARAMS,
) -> o3d.geometry.PointCloud:
    o3d_camera_intrinsic = cam_params.to_open3d()
    if confidence_map is not None:
        depth = np.where(confidence_map < 20, rgb_data.depth, 0.0)
    else:
        depth = rgb_data.depth
    rgb_o3d = o3d.geometry.Image(rgb_data.rgb)
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
    )
    full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d_camera_intrinsic,
        project_valid_depth_only=project_valid_depth_only,
    )
    return full_pcd


def postprocess_predictions(
    rgb_data: RgbdDataNp,
    obj_preds: List[FullObjPred],
    num_grasps: int = 10,
    use_icp: bool = True,
    confidence_map: np.ndarray = None,
) -> Tuple[List[PostprObjPred], o3d.geometry.PointCloud]:
    full_pcd = get_full_pcd(rgb_data, confidence_map)
    postpr_predictions = [
        postprocess_prediction(
            obj_pred,
            full_pcd,
            num_grasps,
            use_icp,
        )
        for obj_pred in obj_preds
    ]
    return postpr_predictions, full_pcd


def postprocess_prediction(
    obj_pred: FullObjPred,
    full_pcd: o3d.geometry.PointCloud,
    num_grasps: int,
    use_icp: bool,
) -> PostprObjPred:
    # Shape postprocessing
    if use_icp:
        postpr_obj_pred = postprocess_shape(obj_pred, full_pcd)
    else:
        postpr_obj_pred = PostprObjPred(obj_pred.pc_o3d, obj_pred.grasp_poses)
    # Grasp postprocessing
    postpr_obj_pred = postprocess_grasps(
        postpr_obj_pred, num_grasps)
    return postpr_obj_pred


def postprocess_shape(
    pred: FullObjPred, full_pcd: o3d.geometry.PointCloud
) -> PostprObjPred:
    masked_pcd = full_pcd.select_by_index(np.flatnonzero(pred.bmask))
    masked_pcd.remove_non_finite_points()

    # Hidden point removal
    _, pt_map = pred.pc_o3d.hidden_point_removal([0.0, 0.0, 0.0], radius=100)
    pred_view_pcd = pred.pc_o3d.select_by_index(pt_map)

    # ICP
    masked_pcd.estimate_normals()
    masked_pcd.orient_normals_towards_camera_location()
    initial_shift = np.eye(4)
    initial_shift[:3, 3] = masked_pcd.get_center() - pred_view_pcd.get_center()
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=pred_view_pcd,
        target=masked_pcd,
        max_correspondence_distance=0.05,
        init=initial_shift,
    )
    corrected_shape = copy.deepcopy(pred.pc_o3d).transform(reg_p2l.transformation)
    if np.linalg.norm(corrected_shape.get_center() - masked_pcd.get_center()) > 0.3:
        print("WARNING: icp might not have converged, skipping")
        return PostprObjPred(pred.pc_o3d, pred.grasp_poses)
    corrected_grasps = reg_p2l.transformation @ pred.grasp_poses
    return PostprObjPred(corrected_shape, corrected_grasps)


def postprocess_grasps(
    pred: PostprObjPred,
    num_grasps: int,
) -> PostprObjPred:
    best_grasps = []
    for grasp_pose in pred.grasp_poses:
        # If we have enough grasps, stop
        if len(best_grasps) >= num_grasps:
            break
        best_grasps.append(grasp_pose)
    pred.grasp_poses = np.array(best_grasps)
    return pred