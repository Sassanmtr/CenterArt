import os
from pathlib import Path
import numpy as np
import json
import wandb
import tqdm
from centerart_model import set_seeds
from scripts.make_rgb_single_dataset import get_normalized_joint_state
from centerart_model.sapien.behaviors import GraspBehavior
from centerart_model.sapien.sapien_envs import GraspEnv
from centerart_model.pipelines.model_pipeline import (
    CenterArtPipeline,
    GTGraspPipeline,
    UMPNetPipeline,
)
from centerart_model.sapien.sapien_utils import CameraObsConfig
from centerart_model.sapien.robots import ROBOTS_DICT
from centerart_model.rgb.rgb_data import RGBDReader
from centerart_model.sapien.sapien_utils import CameraObs, Obs


def get_obs_test(json_file: json, data_dir: Path, urdf_dir: Path, idx: str):
    camTobjs_dir = data_dir / "pose"
    camTobjs_path = str(camTobjs_dir / f"{idx}.npy")
    wTcam_dir = data_dir / "cam_pose"
    wTcam_path = str(wTcam_dir / f"{idx}.npy")
    camTobjs = np.load(camTobjs_path)
    wTcam = np.load(wTcam_path)
    wTobjs = wTcam @ camTobjs
    objs_info_dir = data_dir / "segm"
    first_idx = idx.split("_")[0]
    objs_info_path = str(objs_info_dir / f"{first_idx}_info.json")
    with open(objs_info_path, "r") as f:
        objs_info = json.load(f)
    #
    scene_objs_dict = {}
    uuids_info = get_uuids_info(objs_info, json_file, urdf_dir, wTobjs)
    scene_objs_dict["wTcam"] = wTcam
    scene_objs_dict["objs_info"] = uuids_info
    counter_dir = data_dir / "counter_range"
    counter_path = str(counter_dir / f"{idx}.npy")
    if os.path.exists(counter_path):
        counter_range = np.load(counter_path)
        return scene_objs_dict, counter_range
    return scene_objs_dict, False


def get_uuids_info(obj_infos, json_file, urdf_dir, wTobjs):
    uuids_info = {}
    for i, obj_info in enumerate(obj_infos):
        emb_index = obj_info["emb_index"]
        normilized_joint_state = obj_info["joint_state"]
        for obj in json_file:
            norm_joint_state = get_normalized_joint_state(json_file[obj])
            if (
                json_file[obj]["object_index"] == emb_index
                and norm_joint_state == normilized_joint_state
            ):
                dict_info = make_dict_info(
                    json_file[obj], obj, urdf_dir, wTobjs[i], emb_index
                )
                uuids_info[i] = dict_info
                break
    return uuids_info


def make_dict_info(object_uuid_data, uuid, urdf_dir, wTobjs, emb_index):
    dict_info = {}
    dict_info["uuid"] = uuid
    dict_info["urdf_fpath"] = str(
        urdf_dir / object_uuid_data["object_name"] / "mobility.urdf"
    )
    dict_info["obj_id"] = object_uuid_data["object_name"]
    dict_info["joint_state"] = object_uuid_data["joint_state"]
    dict_info["normalized_joint_state"] = get_normalized_joint_state(object_uuid_data)
    dict_info["max_joint_state"] = object_uuid_data["joint_upper_gt"]
    dict_info["pose4x4"] = wTobjs
    dict_info["scale"] = object_uuid_data["scale"]
    dict_info["volume"] = object_uuid_data["volume"]
    dict_info["link_direction"] = object_uuid_data["link_direction"]
    dict_info["object_index"] = emb_index
    return dict_info


def main(
    rgb_model: str,
    robot_type: str,
    method: str,
    mode: str,
    data_dir: Path,
    urdf_dir: Path,
    obj_data: dict,
    seed: int,
    headless: bool,
    log_wandb: bool,
):
    set_seeds(seed)

    camera_config = CameraObsConfig(rgb=True, depth_gt=True)
    robot = ROBOTS_DICT[robot_type]()
    environment = GraspEnv(
        camera_obs_config=camera_config,
        seed=seed,
        obj_data=obj_data,
        data_dir=str(Path.cwd()),
        headless=headless,
        sapien_robot=robot,
        scene_obj_info={},
    )

    if method == "centerart":
        pipeline = CenterArtPipeline(rgb_model, seed, visualize=not headless)
    elif method == "centerart_noicp":
        pipeline = CenterArtPipeline(
            rgb_model, seed, visualize=not headless, use_icp=False
        )
    elif method == "umpnet":
        pipeline = UMPNetPipeline(data_dir, seed, visualize=not headless)
    elif method == "gt":
        scene_obj_info = environment._get_scene_obj_info()
        pipeline = GTGraspPipeline(scene_obj_info, seed)
    else:
        raise ValueError(f"Invalid method: {method}")
    wandb.init(
        project="[CenterArt] SimEval",
        config={
            "robot_type": robot_type,
            "method": method,
            "depth": "gt",
            "seed": seed,
            "mode": mode,
            "rgb_model": rgb_model,
        },
        mode="online" if log_wandb else "disabled",
    )

    behavior = GraspBehavior(environment, pipeline)

    grasp_attempts = 0
    completed_runs = 0
    total_objects = 0
    max_aborted_runs = 1
    scene_success = 0
    obj_success = 0
    rgbd_reader = RGBDReader(mode=mode)
    for episode in tqdm.tqdm(range(len(rgbd_reader.rgb_paths))):
        # Get the index
        path, data = rgbd_reader.get_data_np_test(episode)
        idx = str(path).split("/")[-1][:-4]
        cam_obs = CameraObs(data.rgb, depth_real=data.depth[..., np.newaxis])
        scene_objs_dict, counter_range = get_obs_test(obj_data, data_dir, urdf_dir, idx)
        if counter_range is not False:
            environment.load_counter(counter_range)
        camera_pose = scene_objs_dict["wTcam"]
        obs = Obs(camera=cam_obs, camera_pose=camera_pose)
        environment.reset_test(scene_objs_dict)
        environment._get_scene_obj_info()
        aborted_runs = 0
        total_objects += environment.num_objs
        while not (
            environment.episode_is_complete() or aborted_runs >= max_aborted_runs
        ):
            info = behavior.run(obs, idx, method)
            print("Run info: ", info)
            grasp_attempts += info["attempts"]
            scene_success_ratio = (
                info["success_grasps"] / info["completed_runs"]
                if info["completed_runs"] > 0
                else 0
            )
            if scene_success_ratio >= 0.5:
                scene_success += 1
            completed_runs += info["completed_runs"]
            obj_success += info["success_grasps"]
            aborted_runs = info["aborted_runs"]
            environment.remove_obj()
        obj_success_rate = obj_success / completed_runs if completed_runs > 0 else 0
        scene_success_rate = scene_success / episode if episode > 0 else 0
        log_data = {
            "grasp_attempts": grasp_attempts,
            "completed_runs": completed_runs,
            "obj_success_rate": obj_success_rate,
            "scene_success_rate": scene_success_rate,
        }
        wandb.log(log_data)
    wandb.finish()
    return


if __name__ == "__main__":
    mode = "valid"
    json_path = Path.cwd() / "configs" / "object_configurations.json"
    data_dir = Path.cwd() / "datasets" / "centerart_model" / "rgbd" / mode
    urdf_dir = Path.cwd() / "datasets" / "urdfs"
    # load json
    with open(json_path, "r") as f:
        json_data = json.load(f)
    main(
        # rgb_model="n90qyp90",  #single obj scenes
        rgb_model="x9f0te7z",  # multiple objs scenes
        robot_type="gripper",
        method="centerart_noicp",
        mode=mode,
        data_dir=data_dir,
        urdf_dir=urdf_dir,
        obj_data=json_data,
        seed=123,
        headless=False,
        log_wandb=False,
    )
