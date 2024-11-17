import os
import random
import numpy as np
import spatialmath as sm
import open3d as o3d
import sapien.core as sapien
from typing import List
import trimesh
from centerart_model.sapien.robots import SapienRobot
from centerart_model.utils.camera import CameraConventions
from centerart_model.utils.configs import ZED2HALF_PARAMS, WSConfigs
import centerart_model.sapien.sapien_utils as sapien_utils
from scripts.make_rgb_single_dataset import get_normalized_joint_state
from centerart_model.utils.utils import mesh_utils
from centerart_model.utils.camera import sample_cam_poses_shell
from centerart_model.sapien.sapien_utils import CameraObsConfig, Obs, Trajectory
from centerart_model.rgb.rgb_data import RgbdDataNp
from centerart_model.rgb.pred_postprocessing import get_full_pcd


def pc_from_box(center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    box = o3d.geometry.TriangleMesh.create_box(
        width=half_size[0] * 2, height=half_size[1] * 2, depth=half_size[2] * 2
    )
    left_bottom_corner = center - half_size
    box.translate(left_bottom_corner)
    pc = np.asarray(box.sample_points_uniformly(number_of_points=2000).points)
    return pc


class BaseEnv:
    def __init__(
        self,
        camera_obs_config: CameraObsConfig = CameraObsConfig(),
        physics_fps: int = 240,
        render_fps: int = 30,
        raytracing: bool = False,
        headless: bool = False,
        sapien_robot: SapienRobot = None,
    ):
        if physics_fps % render_fps != 0:
            raise ValueError(f"{physics_fps=} must be a multiple of {render_fps=}")
        self.physics_dt = 1 / physics_fps
        self.render_each = physics_fps // render_fps
        self.camera_obs_config = camera_obs_config
        if raytracing:
            sapien_utils.enable_raytracing()
        self.engine, self.renderer, self.scene = sapien_utils.init_sapien(
            headless, self.physics_dt
        )
        if not headless:
            self.viewer = sapien_utils.init_viewer(
                self.scene, self.renderer, show_axes=True
            )
        sapien_utils.init_default_material(self.scene)
        sapien_utils.init_lights(self.scene)
        self.scene.add_ground(0.0)
        self.load_walls()
        self.counter = None
        self.table_pc = pc_from_box(WSConfigs.table_position, WSConfigs.table_half_size)
        self.sapien_objs: List[sapien.Articulation] = []
        side_camera = sapien_utils.add_camera(
            self.scene, ZED2HALF_PARAMS, name="side_camera"
        )
        side_cam_pose = sample_cam_poses_shell(
            center=[0, 0, 2], coi_half_size=0.05, num_poses=1
        )
        side_camera.set_pose(sapien.Pose.from_transformation_matrix(side_cam_pose[0]))
        self.camera = side_camera
        self.headless = headless
        self.physics_fps = physics_fps
        self.sapien_robot = sapien_robot
        if sapien_robot is not None:
            sapien_robot.initialize(self.scene)
        return

    @property
    def static_scene_pc(self) -> np.ndarray:
        return self.table_pc

    def _render_step(self, idx: int):
        if idx % self.render_each == 0:
            self.scene.update_render()
            if not self.headless:
                self.viewer.render()
        return



    def load_walls(self) -> None:

        main_wall_half_size = [0.1, 3, 3]
        main_wall_position = [-2.1, 0, 0]

        left_wall_half_size = [3, 0.1, 3]
        left_wall_position = [0, -3, 0]

        right_wall_half_size = [3, 0.1, 3]
        right_wall_position = [0, 3, 0]

        self.main_wall = sapien_utils.add_wall(
            self.scene, half_size=main_wall_half_size, position=main_wall_position
        )
        self.left_wall = sapien_utils.add_wall(
            self.scene, half_size=left_wall_half_size, position=left_wall_position
        )
        self.right_wall = sapien_utils.add_wall(
            self.scene, half_size=right_wall_half_size, position=right_wall_position
        )


    def add_vis_marker(self, pose: sm.SE3, name: str = "marker"):
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.01] * 3, color=[0, 1, 0])
        marker = builder.build_kinematic(name=name)
        marker.set_pose(sapien.Pose.from_transformation_matrix(pose.A))
        marker.hide_visual()
        return marker

    def remove_articulation(self, actor: sapien.Articulation):
        self.scene.remove_articulation(actor)
        return

    def get_obs(self) -> Obs:
        transform = sm.SE3(CameraConventions.robotics_T_opencv, check=False)
        obs = Obs()
        obs.camera = sapien_utils.get_camera_obs(self.camera, self.camera_obs_config)
        obs.joint_state = (
            self.sapien_robot.get_joint_state() if self.sapien_robot else None
        )
        obs.camera_pose = self.get_camera_pose() * transform
        return obs

    def get_camera_pose(self) -> sm.SE3:
        cam_pose_np = self.camera.get_pose().to_transformation_matrix()
        cam_pose = sm.SE3(cam_pose_np, check=False)
        return cam_pose

    def get_scene_pc(self, obs: Obs) -> np.ndarray:
        rgb_data = RgbdDataNp(obs.camera.rgb, obs.camera.depth, None, None, None, None)
        full_pcd = get_full_pcd(rgb_data, project_valid_depth_only=True)
        full_pcd.transform(obs.camera_pose)
        ws_pcd = full_pcd.crop(WSConfigs().ws_aabb)
        return np.asarray(ws_pcd.points)

    def step_once(self, idx: int = 0):
        if self.sapien_robot is not None:
            self.sapien_robot.update_qf()
        self.scene.step()
        self._render_step(idx)
        return

    def step_physics(self, seconds: float = 1.0):
        n_steps = int(seconds * self.physics_fps)
        for i in range(n_steps):
            self.step_once(i)
        return

    def open_gripper(self):
        self.sapien_robot.open_gripper()
        self.step_physics(1)
        return

    def close_gripper(self):
        self.sapien_robot.close_gripper()
        self.step_physics(1)
        return

    def execute_traj(self, trajectory: Trajectory):
        for i in range(len(trajectory.position)):
            self.sapien_robot.set_arm_targets(
                trajectory.position[i], trajectory.velocity[i]
            )
            self.step_once(i)
        return

    def move_to_qpos(self, qpos: np.ndarray):
        self.sapien_robot.set_qpos_target(qpos)
        while np.mean(np.abs(self.sapien_robot.get_joint_state() - qpos)) > 1e-3:
            self.step_physics(0.1)
        return

    def reset_robot(self):
        self.sapien_robot.reset()
        self.step_once()
        return


class GraspEnv(BaseEnv):
    def __init__(
        self,
        seed: int,
        scene_obj_info: dict,
        obj_data,
        data_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.episode_idx = -1
        self.scene_obj_info = scene_obj_info
        self.obj_data = obj_data
        self.data_dir = data_dir
        self.scene_objs: List[mesh_utils.SceneObject] = []
        self.rng = np.random.default_rng(seed)
        self.reset()
        return
    
    def _get_scene_obj_info(self):
        return self.scene_obj_info
    
    def reset(self):
        for obj in self.scene_objs:
            self.scene.remove_articulation(obj.articulation)
        self.scene_objs: List[mesh_utils.SceneObject] = []
        self.scene_obj_info = {}
        self.episode_idx += 1
        self._load_objs()
        self._load_objs_to_scene()
        self._get_scene_obj_info()
        self.num_objs = len(self.scene_objs)
        if self.sapien_robot is not None:
            self.reset_robot()
        self.step_physics(0.5)
        return

    def reset_test(self, scene_objs_dict):
        for obj in self.scene_objs:
            self.scene.remove_articulation(obj.articulation)
        self.scene_objs: List[mesh_utils.SceneObject] = []
        self.scene_obj_info = {}
        self.episode_idx += 1
        self._load_objs_test(scene_objs_dict)
        self._load_objs_to_scene()
        self._get_scene_obj_info()
        self.num_objs = len(self.scene_objs)
        if self.sapien_robot is not None:
            self.reset_robot()
        self.step_physics(0.5)
        return

    def obj_sampler(self):
        keys_list = list(self.obj_data.keys())
        obj_id = random.choice(keys_list)
        return obj_id

    def _scene_obj_creator(self):
        obj_id = self.obj_sampler()
        obj = mesh_utils.SceneObject(
            urdf_fpath=os.path.join(
                "/path/to/urdfs",
                self.obj_data[obj_id]["object_name"],
                "mobility.urdf",
            ),
            id=self.obj_data[obj_id]["object_name"],
            joint_state=self.obj_data[obj_id]["joint_state"],
            normalized_joint_state=get_normalized_joint_state(self.obj_data[obj_id]),
            # max_joint_state=self.obj_data[obj_id]["joint_upper_limit"],
            max_joint_state=self.obj_data[obj_id]["joint_upper_gt"],
            scale=self.obj_data[obj_id]["scale"],
            volume=self.obj_data[obj_id]["volume"],
            emb_index=self.obj_data[obj_id]["object_index"],
        )
        return obj_id, obj

    def _load_objs(self):
        obj_id, obj = self._scene_obj_creator()
        new_obj_position = [
            0.0,
            0.0,
            (obj.volume[2] / 2) + WSConfigs.table_position[-1] - 0.06,
        ]
        obj_pose = sapien.Pose(
            new_obj_position, [0, 0, 0, 1]
        ).to_transformation_matrix()
        obj.pose4x4 = obj_pose
        self.scene_objs.append(obj)
        self.scene_obj_info[obj_id] = {
            "name": obj_id,
            "pose": obj.pose4x4,
        }
        return

    def _load_objs_to_scene(self):
        for obj in self.scene_objs:
            urdf_path = obj.urdf_fpath
            loader: sapien.URDFLoader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.scale = obj.scale
            loader.load_multiple_collisions_from_file = True
            robot: sapien.Articulation = loader.load(urdf_path)
            robot.set_qpos(np.array([obj.joint_state]))
            robot.set_root_pose(sapien.Pose(obj.pose4x4[:3, 3], [0, 0, 0, 1]))
            for arti_joint in robot.get_active_joints():
                arti_joint.set_drive_property(stiffness=0, damping=50)
                arti_joint.set_friction(0.5)
            obj.articulation = robot
        return robot

    def remove_obj(self):
        for obj in self.scene_objs:
            self.scene.remove_articulation(obj.articulation)
            # self.scene_obj_info.pop(obj.id)
        self.scene_objs: List[mesh_utils.SceneObject] = []
        self.scene_obj_info = {}
        return

    def evaluate_success(self, obj_idx, mode, current_state) -> bool:
        obj = self.scene_objs[obj_idx]
        init_state = float(obj.joint_state)
        if mode == "open":
            if np.mean(current_state - init_state) > 0.17:
                result = True
            else:
                result = False
        elif mode == "close":
            if np.mean(current_state - init_state) < -0.17:
                result = True
            else:
                result = False
        else:
            raise ValueError("Invalid mode")
        return result


    def episode_is_complete(self) -> bool:
        done = len(self.scene_objs) == 0
        return done

    def get_gt_pc(self) -> np.ndarray:
        pc_list = []
        for scene_obj in zip(self.scene_objs):
            pose = scene_obj.pose4x4
            mesh_fpath = scene_obj.urdf_fpath
            mesh_trimesh = trimesh.load(mesh_fpath)
            mesh_trimesh.apply_transform(pose)
            pc = np.asarray(mesh_trimesh.sample(1000))
            pc_list.append(pc)
        combined_pc = np.concatenate(pc_list, axis=0)
        return combined_pc

    def load_counter(self, counter_range):
        if self.counter is not None:
            self.scene.remove_actor(self.counter)
        pos = [
            np.mean(np.array(counter_range[0])),
            np.mean(np.array(counter_range[1])),
            counter_range[2][0],
        ]
        half_size = [
            (counter_range[0][1] - counter_range[0][0]) / 2,
            (counter_range[1][1] - counter_range[1][0]) / 2,
            0.02,
        ]
        self.counter = sapien_utils.add_counter(
            self.scene, half_size=half_size, position=pos
        )

    def _load_objs_test(self, scene_objs_dict):
        objs_info = scene_objs_dict["objs_info"]
        for i, info in objs_info.items():
            obj = mesh_utils.SceneObject(
                urdf_fpath=info["urdf_fpath"],
                id=info["obj_id"],
                joint_state=info["joint_state"],
                normalized_joint_state=info["normalized_joint_state"],
                max_joint_state=info["max_joint_state"],
                scale=info["scale"],
                volume=info["volume"],
                link_direction=info["link_direction"],
                emb_index=info["object_index"],
                pose4x4=info["pose4x4"]
            ) 
            self.scene_objs.append(obj)
            self.scene_obj_info[i] = {
                "name": obj.id,
                "pose": obj.pose4x4,
            }
        return


