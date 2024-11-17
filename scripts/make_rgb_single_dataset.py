import sapien.core as sapien
import numpy as np
import json
import os
import math
from pathlib import Path
from tqdm import tqdm
from centerart_model.sapien.scenes import SceneImgRenderer
from centerart_model.utils.mesh_utils import SceneObject
from centerart_model.utils.configs import Directories
import centerart_model.sapien.sapien_utils as sapien_utils
from centerart_model.rgb.rgb_data import write_rgbd_data, get_rgbd_paths, check_exists
import random

np.random.seed(0)
random.seed(0)


def get_normalized_joint_state(json_data):
    joint_state = float(json_data["joint_state"])
    joint_upper_limit = json_data["joint_upper_limit"]
    joint_lower_limit = json_data["joint_lower_limit"]
    normailized_joint_state = (joint_state - joint_lower_limit) / (
        joint_upper_limit - joint_lower_limit
    )
    return normailized_joint_state


def load_counter(scene_renderer, vol, mode):
    if scene_renderer.counter is not None:
        scene_renderer.scene.remove_actor(scene_renderer.counter)
    material = sapien_utils.render_material_from_ambient_cg_texture(
        scene_renderer.renderer, np.random.choice(scene_renderer.textures)
    )
    counter_half_size = [
        np.random.uniform(0.4, 0.7),
        np.random.uniform(0.4, 0.7),
        np.random.uniform(0.01, 0.01),
    ]
    counter_position = [0.0, 0.0, 0.2 - counter_half_size[2]]
    if mode == "test":
        scene_renderer.counter = sapien_utils.add_counter(
            scene_renderer.scene,
            half_size=counter_half_size,
            position=counter_position,
        )
    else:
        scene_renderer.counter = sapien_utils.add_counter(
            scene_renderer.scene,
            half_size=counter_half_size,
            position=counter_position,
            material=material,
        )
    return


def make_multiple_scene_objs(data_dir, json_data, poses_raw):
    scene_objs = []
    for obj_id, pose_raw in poses_raw.items():
        object_name = json_data[obj_id]["object_name"]
        pose4x4 = sapien.Pose(pose_raw, [0, 0, 0, 1]).to_transformation_matrix()
        obj = SceneObject(
            urdf_fpath=os.path.join(data_dir, object_name, "mobility.urdf"),
            id=obj_id,
            joint_state=json_data[obj_id]["joint_state"],
            normalized_joint_state=get_normalized_joint_state(json_data[obj_id]),
            max_joint_state=json_data[obj_id]["joint_upper_limit"],
            scale=json_data[obj_id]["scale"],
            volume=[
                json_data[obj_id]["volume"][2],
                json_data[obj_id]["volume"][0],
                json_data[obj_id]["volume"][1],
            ],
            emb_index=json_data[obj_id]["object_index"],
            pose4x4=pose4x4,
        )
        scene_objs.append(obj)
    return scene_objs


def load_objs_to_scene(scene_renderer, scene_objs):
    robots = []
    for obj in scene_objs:
        urdf_path = obj.urdf_fpath
        loader: sapien.URDFLoader = scene_renderer.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = obj.scale
        loader.load_multiple_collisions_from_file = True
        robot: sapien.Articulation = loader.load(urdf_path)
        robot.set_qpos(np.array([obj.joint_state]))
        robot.set_root_pose(sapien.Pose(obj.pose4x4[:3, 3], [0, 0, 0, 1]))
        obj.articulation = robot
        robots.append(robot)
    return robots


def make_data_dirs(mode="train"):
    (Directories.RGBD / mode / "rgb").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "depth_gt").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "depth_noisy").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "pose").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "cam_pose").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "segm").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "sem_seg").mkdir(parents=True, exist_ok=True)
    (Directories.RGBD / mode / "counter_range").mkdir(parents=True, exist_ok=True)


def add_object_texture(robot, scene_renderer):
    obj_texture = sapien_utils.render_material_from_ambient_cg_texture(
        scene_renderer.renderer,
        np.random.choice(scene_renderer.textures),
    )
    for link in robot.get_links():
        for body in link.get_visual_bodies():
            for shape in body.get_render_shapes():
                shape.set_material(obj_texture)


def object_collector(collected_objs, json_data, number_of_objs=4):
    mic_flag = False
    high_flag = False
    target_objs = []
    # Divide the objects by category
    available_objs = []
    for obj_id, obj_data in json_data.items():
        if obj_data["object_name"] + "_" + str(obj_data["scale"]) in collected_objs:
            available_objs.append(obj_id)

    while len(target_objs) < number_of_objs:
        selected_obj = np.random.choice(available_objs)
        if (
            (json_data[selected_obj]["category"] == "Microwave" and not mic_flag)
            or (
                json_data[selected_obj]["category"] != "Microwave"
                and json_data[selected_obj]["volume"][1] > 0.8
                and not high_flag
            )
            or (
                json_data[selected_obj]["category"] != "Microwave"
                and json_data[selected_obj]["volume"][1] < 0.8
            )
        ):
            target_objs.append(selected_obj)
            available_objs.remove(selected_obj)
            if json_data[selected_obj]["category"] == "Microwave":
                mic_flag = True
            if json_data[selected_obj]["volume"][1] > 0.8:
                high_flag = True

    random.shuffle(target_objs)

    return target_objs


def compute_offset(y_vol, joint_state):
    # Check if joint state is greater than 1.57 (90 degrees)
    joint_state = float(joint_state)
    if joint_state > 1.57:
        # Compute the offset based on the volume of the object
        y_offset = y_vol * math.sin(joint_state - 1.57)
        return y_offset
    else:
        # If joint state is not greater than 1.57, no offset is needed
        return 0.0


def get_order_and_counter_clearance(scene_objs, json_data):
    # Collet the volumes of the objects first
    x_max = 0
    z_max = 0
    y_start_offset = 0
    y_end_offset = 0

    volumes = {}
    for obj_id in scene_objs:
        volume = json_data[obj_id]["volume"]
        sapien_volume = [
            volume[2],
            volume[0],
            volume[1],
        ]
        volumes[obj_id] = sapien_volume
    # Sort the volumes based on the z axis. With 50% chance increasing or decreasing
    decreasing = np.random.rand() > 0.5
    sorted_volumes = sorted(volumes.items(), key=lambda x: x[1][2], reverse=decreasing)
    if decreasing:
        for obj_id, volume in sorted_volumes:
            if volume[0] > x_max:
                x_max = volume[0]
            if volume[2] > z_max and volume[-1] <= 0.8:
                z_max = volume[2]
            if volume[-1] > 0.8:
                y_start_offset += volume[1]
    else:
        for obj_id, volume in sorted_volumes:
            if volume[0] > x_max:
                x_max = volume[0]
            if volume[2] > z_max and volume[-1] <= 0.8:
                z_max = volume[2]
            if volume[-1] > 0.8:
                y_end_offset -= volume[1]

    counter_y_offset = [y_start_offset, y_end_offset]

    return sorted_volumes, x_max, z_max, counter_y_offset


def get_poses_raw(json_data, scene_objs, threshold=0.15):
    current_position = [-2.0, -3, 0]
    next_position = [-2.0, -3, 0]
    poses_raw = {}
    y_positions = []
    y_volumes = []
    sorted_vol, x_max, z_max, counter_y_offset = get_order_and_counter_clearance(
        scene_objs, json_data
    )
    target_x = current_position[0] + x_max + 0.02
    for obj_id, volume in sorted_vol:
        obj_data = json_data[obj_id]
        link_direction = obj_data["link_direction"]
        joint_state = obj_data["joint_state"]
        if obj_data["category"] != "Microwave":
            x_offset = volume[0] / 2
            y_offset = volume[1] / 2
            z_offset = volume[2] / 2
            current_position = [
                target_x - x_offset,
                current_position[1] + y_offset + threshold,
                current_position[2] + z_offset + 0.02,
            ]
            next_position = [
                target_x - x_offset,
                current_position[1] + y_offset,
                current_position[2] + z_offset + 0.02,
            ]
            if link_direction == "left" or link_direction == "right":
                additional_y_offset = compute_offset(volume[1], joint_state)
                next_position[1] = next_position[1] + additional_y_offset
                if link_direction == "right":
                    current_position[1] = current_position[1] + additional_y_offset

            poses_raw[obj_id] = current_position
            y_positions.append(current_position[1])
            y_volumes.append(volume[1])
            next_position[0] = -1.0
            next_position[2] = 0
            current_position = next_position
    # Postprocessing to make the objects centers near the 0 in y axis
    y_offset = np.mean(np.array(y_positions))
    y_initial = y_positions[0] - y_offset - (y_volumes[0] / 2)
    y_final = y_positions[-1] - y_offset + (y_volumes[-1] / 2)
    center_y = (y_initial + y_final) / 2
    for obj_id in scene_objs:
        # if the object is not a microwave, then adjust the y position
        if json_data[obj_id]["category"] != "Microwave":
            poses_raw[obj_id][1] = poses_raw[obj_id][1] - y_offset
    counter_x_range = [-2.0, -2.0 + x_max]
    counter_y_range = [
        y_initial + counter_y_offset[0],
        y_final + counter_y_offset[1],
    ]
    counter_z_range = [z_max + 0.04, z_max + 0.04]
    counter = [counter_x_range, counter_y_range, counter_z_range]
    for obj_id in sorted_vol:
        if json_data[obj_id[0]]["category"] == "Microwave":
            poses_raw[obj_id[0]] = [
                np.mean(np.array(counter_x_range)),
                np.mean(np.array(counter_y_range)),
                np.mean(np.array(counter_z_range)) + (obj_id[1][2] / 2) + 0.03,
            ]
    return poses_raw, center_y, counter


def get_poses_raw_single(json_data, scene_objs, threshold=0.15):
    current_position = [-2.0, -3, 0]
    next_position = [-2.0, -3, 0]
    poses_raw = {}
    y_positions = []
    y_volumes = []
    sorted_vol, x_max, z_max, counter_y_offset = get_order_and_counter_clearance(
        scene_objs, json_data
    )
    target_x = current_position[0] + x_max + 0.02
    for obj_id, volume in sorted_vol:
        obj_data = json_data[obj_id]
        link_direction = obj_data["link_direction"]
        joint_state = obj_data["joint_state"]

        x_offset = volume[0] / 2
        y_offset = volume[1] / 2
        z_offset = volume[2] / 2
        current_position = [
            target_x - x_offset,
            current_position[1] + y_offset + threshold,
            current_position[2] + z_offset + 0.02,
        ]
        next_position = [
            target_x - x_offset,
            current_position[1] + y_offset,
            current_position[2] + z_offset + 0.02,
        ]

        poses_raw[obj_id] = current_position
        poses_raw[obj_id][1] = 0.0
        y_positions.append(current_position[1])
        y_volumes.append(volume[1])
        next_position[0] = -1.0
        next_position[2] = 0
        current_position = next_position
    # Postprocessing to make the objects centers near the 0 in y axis
    y_offset = np.mean(np.array(y_positions))
    y_initial = y_positions[0] - y_offset - (y_volumes[0] / 2)
    y_final = y_positions[-1] - y_offset + (y_volumes[-1] / 2)
    center_y = (y_initial + y_final) / 2
    counter_x_range = [-2.0, -2.0 + x_max]
    counter_y_range = [
        y_initial + counter_y_offset[0],
        y_final + counter_y_offset[1],
    ]
    counter_z_range = [z_max + 0.04, z_max + 0.04]
    counter = [counter_x_range, counter_y_range, counter_z_range]
    return poses_raw, center_y, counter


def split_object_collector(available_objs, urdf_dir, mode):
    microwave_objs = []
    oven_objs = []
    refrigerator_objs = []
    dishwasher_objs = []
    furniture_objs = []
    collected_objs = []
    for obj in available_objs:
        object_name = obj.split("_")[0]
        object_urdf_path = os.path.join(urdf_dir, object_name, "meta.json")
        with open(object_urdf_path, "r") as file:
            json_data = json.load(file)
        if json_data["model_cat"] == "Microwave":
            microwave_objs.append(obj)
        elif json_data["model_cat"] == "Oven":
            oven_objs.append(obj)
        elif json_data["model_cat"] == "Refrigerator":
            refrigerator_objs.append(obj)
        elif json_data["model_cat"] == "Dishwasher":
            dishwasher_objs.append(obj)
        else:
            furniture_objs.append(obj)

    if mode == "train":
        train_objs = []
        test_mic_objs = []
        test_oven_objs = []
        test_ref_objs = []
        test_dish_objs = []
        test_fur_objs = []
        current_obj = []
        # # Randomly select 2 object from Microwave
        while True:
            chosen_obj = np.random.choice(microwave_objs)
            if chosen_obj.split("_")[0] not in current_obj:
                current_obj.append(chosen_obj.split("_")[0])
                test_mic_objs.append(chosen_obj)
            if len(test_mic_objs) == 2:
                current_obj = []
                break
        # Randomly select 3 object from Oven
        while True:
            chosen_obj = np.random.choice(oven_objs)
            if chosen_obj.split("_")[0] not in current_obj:
                current_obj.append(chosen_obj.split("_")[0])
                test_oven_objs.append(chosen_obj)
            if len(test_oven_objs) == 3:
                current_obj = []
                break
        # Randomly select 1 object from Refrigerator
        test_ref_objs.append(np.random.choice(refrigerator_objs))
        # Randomly select 3 object from Dishwasher
        while True:
            chosen_obj = np.random.choice(dishwasher_objs)
            if chosen_obj.split("_")[0] not in current_obj:
                current_obj.append(chosen_obj.split("_")[0])
                test_dish_objs.append(chosen_obj)
            if len(test_dish_objs) == 3:
                current_obj = []
                break
        # Randomly select 3 object from Furniture
        while True:
            chosen_obj = np.random.choice(furniture_objs)
            if chosen_obj.split("_")[0] not in current_obj:
                current_obj.append(chosen_obj.split("_")[0])
                test_fur_objs.append(chosen_obj)
            if len(test_fur_objs) == 3:
                current_obj = []
                break
        test_objs = (
            test_mic_objs
            + test_oven_objs
            + test_ref_objs
            + test_dish_objs
            + test_fur_objs
        )
        # train objects are the ones that are not in the test objects
        for obj in available_objs:
            if obj not in test_objs:
                train_objs.append(obj)
        split_objs = {"train": train_objs, "test": test_objs}
        # Save the split objects to a json file if not exists
        split_file_path = str(Path.cwd() / "configs" / "encoder_split_objects.json")
        if not os.path.exists(split_file_path):
            with open(split_file_path, "w") as file:
                json.dump(split_objs, file)
        else:
            # load the split objects from the json file
            with open(split_file_path, "r") as file:
                split_objs = json.load(file)
        collected_objs = split_objs["train"]
    elif mode == "valid":
        # load the split objects from the json file
        split_file_path = str(Path.cwd() / "configs" / "encoder_split_objects.json")
        with open(split_file_path, "r") as file:
            split_data = json.load(file)
        collected_objs = split_data["train"]
    elif mode == "test":
        # load the split objects from the json file
        split_file_path = str(Path.cwd() / "configs" / "encoder_split_objects.json")
        with open(split_file_path, "r") as file:
            split_data = json.load(file)
        collected_objs = split_data["test"]
    else:
        raise ValueError("Invalid mode")
    return collected_objs


def make_single_dataset(
    collected_objs,
    data_dir,
    json_data,
    ray_tracing,
    object_texture,
    mode,
    imgs_per_scene,
    objs_per_scene,
    nr_scenes,
    lock=None,
):
    robots = None
    scene_renderer = SceneImgRenderer(
        headless=True, raytracing=ray_tracing, imgs_per_scene=imgs_per_scene, lock=lock
    )
    # if mode is valid change the seed to have different scenes from the training
    if mode == "valid":
        np.random.seed(1)
        random.seed(1)
    # Make scene objects
    make_data_dirs(mode)
    for i in tqdm(range(nr_scenes)):
        print("i: ", i)
        if robots is not None:
            for robot in robots:
                scene_renderer.scene.remove_articulation(robot)
        # Collect the objects to be imported in the scene
        scene_objs = object_collector(collected_objs, json_data, objs_per_scene)
        poses_raw, y_center, counter_range = get_poses_raw_single(json_data, scene_objs)
        scene_objs = make_multiple_scene_objs(data_dir, json_data, poses_raw)
        # Load objects to the scene
        robots = load_objs_to_scene(scene_renderer, scene_objs)

        if object_texture and mode != "test":
            if np.random.rand() > 0.2:  # 80% chance to change the table texture
                for robot in robots:
                    add_object_texture(robot, scene_renderer)
        rgbd_paths = get_rgbd_paths(mode, i, imgs_per_scene)
        if check_exists(rgbd_paths):
            continue
        gt_data_list, objs_info = scene_renderer.make_data(
            scene_objs, counter_range, mode, y_center
        )
        write_rgbd_data(rgbd_paths, gt_data_list, objs_info)


if __name__ == "__main__":
    mode = "test"
    headless = True
    raytracing = True
    object_texture = True
    imgs_per_scene = 1
    objs_per_scene = 1
    nr_scenes = 200
    data_dir = str(Path.cwd() / "datasets" / "urdfs")
    json_file_path = str(Path.cwd() / "configs" / "object_configurations.json")
    with open(json_file_path, "r") as file:
        json_file = json.load(file)
    # Create a list of all object name with scale in the json file
    available_objs = []
    for obj_id, obj_data in json_file.items():
        new_obj = obj_data["object_name"] + "_" + str(obj_data["scale"])
        if new_obj not in available_objs:
            available_objs.append(new_obj)
    # split the objects into train and test
    collected_objs = split_object_collector(available_objs, data_dir, mode)

    make_single_dataset(
        collected_objs,
        data_dir,
        json_file,
        raytracing,
        object_texture,
        mode,
        imgs_per_scene,
        objs_per_scene,
        nr_scenes,
    )
