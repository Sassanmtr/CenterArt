import os
from pathlib import Path
import numpy as np
from scripts.make_sgdf_dataset import object_to_trimesh, object_to_point_cloud
from centerart_model.utils.visualize import RerunViewer

# Load grasp files and visualize them 
current_directory = Path.cwd()
grasp_dir = str(current_directory / "datasets" / "grasps")
files = os.listdir(grasp_dir)
files.sort()
for file in files:
    # if ends with .npy
    if file.endswith(".npy"):
        # get the object name
        object_name = file.split("_")[0]
        scale = float(file.split("_")[1])
        joint_state = file.split("_")[2][:-4]
        print(
            "Object name: ", object_name, "Scale: ", scale, "Jointstate: ", joint_state
        )
        urdf_path = str(current_directory / "datasets" / "urdfs" / object_name / "mobility.urdf")
        grasp_path = os.path.join(grasp_dir, file)
        # load grasps and generate pointcloud
        grasps = np.load(grasp_path)
        object_trimesh = object_to_trimesh(urdf_path, scale, joint_state)
        pc_points, _ = object_to_point_cloud(object_trimesh)
        vis = RerunViewer()
        vis.add_pointcloud("pcd", pc_points, radii=0.002)
        vis.add_grasps(f"grasps/_", grasps)
        input("Press enter to continue")
