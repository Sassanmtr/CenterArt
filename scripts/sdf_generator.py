import mesh_to_sdf
from pathlib import Path
import os
from typing import Tuple
import trimesh
import urdfpy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_urdf_meshes(urdf_file_path, scale, joint_state):
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    # Open the articulated joints to the desired joint state
    joint_name = arti_obj.actuated_joints[0].name
    transformed_meshes = []
    for mesh, pose in arti_obj.visual_trimesh_fk({joint_name: joint_state}).items():
        scale_mat = np.eye(4) * scale
        scale_mat[3, 3] = 1.0
        mesh.apply_transform(scale_mat @ pose["pose"])
        transformed_meshes.append(mesh)

    return transformed_meshes


def object_to_trimesh(urdf_file_path, scale, joint_state):
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    joint_name = arti_obj.actuated_joints[0].name
    fk = arti_obj.visual_trimesh_fk({joint_name: joint_state})
    trimesh_scene = trimesh.Scene()
    for mesh, pose in fk.items():
        scale_mat = np.eye(4) * scale
        scale_mat[3, 3] = 1.0
        try:
            trimesh_scene.add_geometry(mesh, transform=scale_mat @ pose)
        except:
            trimesh_scene.add_geometry(mesh, transform=scale_mat @ pose["pose"])
    return trimesh_scene


def object_to_sdf(
    object_trimesh, number_samples: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a urdf_object and a (partial) joint configuration
    Returns:
      - points
      - sdf
    """
    points, sdf = mesh_to_sdf.sample_sdf_near_surface(
        object_trimesh,
        surface_point_method="scan",
        transform_back=True,
        sign_method="normal",  # The normals are mis-aligned for some PartNetMobility objects
        number_of_points=number_samples,  # default 500000
    )
    return points, sdf


def sdf_visualizer(points, sdf):
    # You can adjust the following parameters to change the appearance of the plot.
    marker_size = 1
    cmap = "coolwarm"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Scatter plot the points with SDF values as colors
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=sdf, s=marker_size, cmap=cmap
    )
    # Set labels for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def object_to_point_cloud(
    object_trimesh, number_samples: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    mesh, transform = mesh_to_sdf.utils.scale_to_unit_sphere(
        object_trimesh, get_transform=True
    )
    surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(
        mesh,
        sample_point_count=number_samples,
        surface_point_method="sample",  # [scan, sample] To allow inside of the mesh?
        calculate_normals=True,
    )
    points = surface_point_cloud.points * transform["scale"] + transform["translation"]
    normals = surface_point_cloud.normals
    return points, normals


def point_cloud_visualizer(pc_points):
    pc_points = pc_points  # Replace this with your point cloud data
    # Create a 3D scatter plot for the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Set the point size for visualization
    point_size = 0.01  # Adjust this to your preference
    ax.scatter(pc_points[:, 0], pc_points[:, 1], pc_points[:, 2], s=point_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def main_data_save(urdf_dir, grasp_dir, save_sdfvalue_dir, save_sdfpoint_dir):
    grasp_files = os.listdir(grasp_dir)
    grasp_files.sort()
    sdf_files = os.listdir(save_sdfvalue_dir)
    # for grasp_file in grasp_files:
    for grasp_file in tqdm(grasp_files):
        if grasp_file not in sdf_files:
            object_name = grasp_file.split("_")[0]
            scale = float(grasp_file.split("_")[1])
            joint_state = float(grasp_file.split("_")[2][:-4])
            urdf_path = os.path.join(urdf_dir, object_name, "mobility.urdf")

            object_trimesh = object_to_trimesh(urdf_path, scale, joint_state)
            points, sdf = object_to_sdf(object_trimesh)
            np.save(os.path.join(save_sdfvalue_dir, grasp_file), sdf)
            np.save(os.path.join(save_sdfpoint_dir, grasp_file), points)
            print("Saved data for:", grasp_file)


if __name__ == "__main__":
    urdf_dir = Path.cwd() / "datasets" / "urdfs"
    grasp_dir = Path.cwd() / "datasets" / "grasps"
    save_sdfvalue_dir = Path.cwd() / "datasets" / "sdfs_value"
    save_sdfpoint_dir = Path.cwd() / "datasets" / "sdfs_point"
    main_data_save(urdf_dir, grasp_dir, save_sdfvalue_dir, save_sdfpoint_dir)
