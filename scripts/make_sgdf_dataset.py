from pathlib import Path
import os
import json
import numpy as np

urdf_dir = Path.cwd() / "datasets" / "urdfs"
grasp_dir = Path.cwd() / "datasets" / "grasps"
sdfs_value_dir = Path.cwd() / "datasets" / "sdfs_value"
sdfs_point_dir = Path.cwd() / "datasets" / "sdfs_point"
saved_dir = Path.cwd() / "datasets" / "centerart_model" / "sgdf"
json_file_path = Path.cwd() / "configs" / "object_configurations.json"
data_split = Path.cwd() / "configs" / "decoder_data_split.json"
with open(json_file_path, "r") as file:
    data = json.load(file)
with open(data_split, "r") as file:
    data_split = json.load(file)
    train_objects = data_split["train"]
    valid_objects = data_split["valid"]

for object_id, object_data in data.items():
    scale = object_data["scale"]
    joint_state = object_data["joint_state"]
    object_name = object_data["object_name"]
    urdf_path = os.path.join(urdf_dir, object_name, "mobility.urdf")
    grasp_name = str(object_name) + "_" + str(scale) + "_" + str(joint_state) + ".npy"
    grasp_path = os.path.join(str(grasp_dir), grasp_name)
    grasps = np.load(grasp_path)
    sdf_value_path = os.path.join(str(sdfs_value_dir), grasp_name)
    sdf_values = np.load(sdf_value_path)
    sdf_point_path = os.path.join(str(sdfs_point_dir), grasp_name)
    sdf_points = np.load(sdf_point_path)
    if object_id in train_objects:
        mode = "train"
    elif object_id in valid_objects:
        mode = "valid"
    else:
        raise ValueError("Object id not in train or valid")

    object_saved_dir = os.path.join(saved_dir, mode, object_id)
    os.makedirs(object_saved_dir, exist_ok=True)

    # Save points, sdf, and grasps as .npy files
    np.save(os.path.join(mode, object_saved_dir, "sdf_points.npy"), sdf_points)
    np.save(os.path.join(mode, object_saved_dir, "sdf_values.npy"), sdf_values)
    np.save(os.path.join(mode, object_saved_dir, "grasps.npy"), grasps)

    print("Saved data for:", object_id)
