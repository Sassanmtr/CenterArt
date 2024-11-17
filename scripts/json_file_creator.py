import os
from pathlib import Path
import json
import uuid
import urdfpy
import random
import numpy as np

# Seeds
seed = 12345
random.seed(seed)
np.random.seed(seed)


def object_collectors(urdf_dir, categories):
    target_objects = {}
    urdf_files = os.listdir(urdf_dir)
    urdf_files.sort()
    for obj in urdf_files:
        subdir_path = os.path.join(urdf_dir, obj)
        meta_file_path = os.path.join(subdir_path, "meta.json")
        with open(meta_file_path, "r") as meta_file:
            meta_data = json.load(meta_file)
            object_category = meta_data.get("model_cat")
            if object_category in categories:
                target_objects[obj] = [object_category]
    return target_objects


def main(dataset_dir, categories):
    grasp_dir = dataset_dir / "grasps"
    urdf_dir = dataset_dir / "urdfs"
    link_path = str(dataset_dir.parent / "configs" / "link_handle_info.json")
    # load link_handle_info.json
    with open(link_path, "r") as file:
        link_handle_info = json.load(file)
    target_objects = object_collectors(urdf_dir, categories)
    object_configurations = {}
    embedded_files = []
    embedded_index = -1
    grasp_files = os.listdir(grasp_dir)
    grasp_files.sort()
    for grasp_file in grasp_files:
        object_name = grasp_file.split("_")[0]
        scale = float(grasp_file.split("_")[1])
        emb_file = str(object_name) + "_" + str(scale)
        joint_state = grasp_file.split("_")[2][:-4]
        if object_name in target_objects.keys():
            # Get Category and Index
            object_category = target_objects[object_name][0]
            if emb_file not in embedded_files:
                embedded_index += 1
            embedded_files.append(emb_file)
            # Get Scaled Volume
            subdir_path = os.path.join(urdf_dir, object_name)
            bounding_box_file_path = os.path.join(subdir_path, "bounding_box.json")
            with open(bounding_box_file_path, "r") as bounding_box_file:
                bb_data = json.load(bounding_box_file)
                volume = [
                    bb_data["max"][0] - bb_data["min"][0],
                    bb_data["max"][1] - bb_data["min"][1],
                    bb_data["max"][2] - bb_data["min"][2],
                ]
            scaled_volume = [
                volume[0] * scale,
                volume[1] * scale,
                volume[2] * scale,
            ]
            # Get URDF File
            urdf_file_path = os.path.join(subdir_path, "mobility.urdf")
            urdf_object = urdfpy.URDF.load(urdf_file_path)
            # Get Joint Info
            actuated_joint = urdf_object.actuated_joints
            if len(actuated_joint) > 1:
                raise Exception(
                    print(f"More than one actuated joint for {object_name}")
                )
            else:
                actuated_joint = actuated_joint[0]
            joint_type = actuated_joint.joint_type
            joint_upper_limit = actuated_joint.limit.upper
            joint_lower_limit = actuated_joint.limit.lower
            # Generate UUID
            config_uuid = str(uuid.uuid4())
            configuration = {
                "object_name": object_name,
                "object_index": embedded_index,
                "scale": scale,
                "joint_state": joint_state,
                "joint_type": joint_type,
                "joint_lower_limit": joint_lower_limit,
                "joint_upper_limit": 3.14,
                "joint_upper_gt" : joint_upper_limit,
                "volume": scaled_volume,
                "link_direction": link_handle_info[object_name]["link"],
                "category": object_category,
            }

            object_configurations[config_uuid] = configuration

    output_file_path = "object_configurations.json"
    with open(output_file_path, "w") as output_file:
        json.dump(object_configurations, output_file, indent=4)

    print(f"Saved object configurations to {output_file_path}")
    return


if __name__ == "__main__":
    dataset_dir = Path.cwd() / "datasets"

    categories = ["Microwave", "Oven", "Dishwasher", "Refrigerator", "StorageFurniture"]
    main(dataset_dir, categories)
