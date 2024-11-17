from pathlib import Path
import json
import numpy as np
from centerart_model.pipelines.model_pipeline import CenterArtPipeline
from centerart_model.rgb.rgb_data import RGBDReader, RGBDataset, RgbdDataNp
from centerart_model.rgb.rgb_inference import extract_obj_predictions, FullObjPred
from centerart_model.rgb.pred_postprocessing import postprocess_predictions
from centerart_model.sapien.sapien_utils import CameraObs, Obs
from centerart_model.sgdf.sgdf_inference import SGDFInference
from centerart_model.utils.visualize import RerunViewer

def get_data_dict(data_dict_path):
    with open(data_dict_path, "r") as json_file:
        data = json.load(json_file)
    return data

def vis_sgdf_net_scene(sgdf_model: str, data_dict: dict, start_idx: int, mode: str):
    rgbd_dataset = RGBDataset(sgdf_model, mode)
    sgdf_inference = SGDFInference(sgdf_model, data_dict)
    for i in range(start_idx, start_idx + 1000):
        (
            rgb,
            depth,
            heatmap_target,
            pose_target,
            shape_target,
            joint_target,
            _,
        ) = rgbd_dataset[i]
        rgbd_data = RgbdDataNp.from_torch(rgb, depth, heatmap_target)
        rgb_gt_preds = extract_obj_predictions(
            heatmap_target, pose_target, shape_target, joint_target
        )
        sgdf_preds = [
            sgdf_inference.predict_reconstruction(pred.embedding, pred.joint_code)
            for pred in rgb_gt_preds
        ]
        full_preds = [
            FullObjPred.from_net_predictions(rgb_pred, sgdf_pred)
            for rgb_pred, sgdf_pred in zip(rgb_gt_preds, sgdf_preds)
        ]
        postpr_preds, full_pcd = postprocess_predictions(
            rgbd_data, full_preds, use_icp=True
        )
        print("Joint Codes: ", [pred.joint_code for pred in rgb_gt_preds])
        print()
        RerunViewer()
        RerunViewer.clear()
        RerunViewer.vis_rgbd_data(rgbd_data)
        RerunViewer.add_o3d_pointcloud("vis/full_pcd", full_pcd, radii=0.0015)
        for idx in range(len(postpr_preds)):
            RerunViewer.visualize_prediction(full_preds[idx], postpr_preds[idx], idx)
        input("Press enter to continue...")
    return


def vis_full_prediction_sim(rgb_model: str, data_dict: dict, start_idx: int, mode: str):
    rgbd_reader = RGBDReader(mode)
    pipeline = CenterArtPipeline(rgb_model, data_dict, use_icp=True)
    for i in range(start_idx, start_idx + 1000):
        data = rgbd_reader.get_data_np(i)
        cam_obs = CameraObs(data.rgb, depth_real=data.depth[..., np.newaxis])
        obs = Obs(camera=cam_obs, camera_pose=data.poses)
        _, _, _ = pipeline._centerart_predictions(obs, num_grasps=1)
        input("Press enter to continue...")
    return

if __name__ == "__main__":
    ## Validation of Entire Pipeline
    data_path = Path.cwd() / "configs" / "object_configurations.json"
    data_dict = get_data_dict(data_path)
    vis_full_prediction_sim("x9f0te7z", data_dict, 0, "valid")

