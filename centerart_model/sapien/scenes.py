import os
import numpy as np
from typing import Optional
import sapien.core as sapien
from dataclasses import dataclass
from multiprocessing.synchronize import Lock as LockBase
from centerart_model.utils.configs import Directories, ZED2HALF_PARAMS
from centerart_model.utils.camera import sample_cam_poses_shell
from centerart_model.rgb.heatmaps import heatmap_from_segmentation
from centerart_model.utils.mesh_utils import SceneObject, AmbientCGTexture
import centerart_model.sapien.sapien_utils as sapien_utils

# For debugging
import numpy as np
import matplotlib.pyplot as plt


# end of debugging func
@dataclass
class GTData:
    rgb: np.ndarray
    depth_gt: np.ndarray
    depth_noisy: np.ndarray
    camTposes: np.ndarray
    binary_masks: np.ndarray
    heatmap: np.ndarray
    segmentation: np.ndarray
    camera_pose: np.ndarray = None
    counter_range: np.ndarray = None


class SceneImgRenderer:
    def __init__(
        self,
        headless: bool,
        raytracing: bool,
        imgs_per_scene: int,
        lock: Optional[LockBase] = None,
    ) -> None:
        # Initialize SAPIEN
        if raytracing:
            sapien_utils.enable_raytracing()
        self.engine, self.renderer, self.scene = sapien_utils.init_sapien(
            headless=headless, physics_dt=1 / 240
        )
        if not headless:
            self.viewer = sapien_utils.init_viewer(
                self.scene, self.renderer, show_axes=False
            )
        sapien_utils.init_default_material(self.scene)
        self.sensor = sapien_utils.add_sensor(
            self.scene, ZED2HALF_PARAMS, name="sensor"
        )
        self.camera_obs_config = sapien_utils.CameraObsConfig(
            rgb=True, depth_real=True, depth_gt=True, segmentation=True, normal=False
        )
        self.textures = [
            AmbientCGTexture(path) for path in Directories.TEXTURES.iterdir()
        ]
        self.lights = sapien_utils.init_lights(self.scene)
        self.sapien_objs: list[sapien.Actor] = []
        self.table: sapien.Actor = None
        self.counter: sapien.Actor = None
        self.ground: sapien.ActorStatic = None
        self.main_wall: sapien.ActorStatic = None
        self.left_wall: sapien.ActorStatic = None
        self.right_wall: sapien.ActorStatic = None
        self.imgs_per_scene = imgs_per_scene
        self.lock = lock
        return

    def _randomize_light(self) -> None:
        for light in self.lights:
            light_position = np.random.uniform(-2, 2, size=3)
            light_position[2] += 3
            light.set_position(light_position)
        return

    def _load_random_floor(self, mode) -> None:
        if self.ground is not None:
            self.scene.remove_actor(self.ground)
        material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        if mode == "test":
            self.ground = self.scene.add_ground(altitude=0)
        else:
            if np.random.rand() > 0.1:
                self.ground = self.scene.add_ground(
                    altitude=0, render_material=material
                )
            else:
                self.ground = self.scene.add_ground(altitude=0)
        return

    def _load_random_walls(self, mode) -> None:
        if self.main_wall is not None:
            self.scene.remove_actor(self.main_wall)
        if self.left_wall is not None:
            self.scene.remove_actor(self.left_wall)
        if self.right_wall is not None:
            self.scene.remove_actor(self.right_wall)

        main_wall_material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        left_wall_material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        right_wall_material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )

        main_wall_half_size = [0.1, 3, 3]
        main_wall_position = [-2.1, 0, 0]

        left_wall_half_size = [3, 0.1, 3]
        left_wall_position = [0, -3, 0]

        right_wall_half_size = [3, 0.1, 3]
        right_wall_position = [0, 3, 0]

        if mode == "test":
            self.main_wall = sapien_utils.add_wall(
                self.scene, half_size=main_wall_half_size, position=main_wall_position
            )
            self.left_wall = sapien_utils.add_wall(
                self.scene, half_size=left_wall_half_size, position=left_wall_position
            )
            self.right_wall = sapien_utils.add_wall(
                self.scene, half_size=right_wall_half_size, position=right_wall_position
            )
        else:
            if np.random.rand() > 0.1:
                self.main_wall = sapien_utils.add_wall(
                    self.scene,
                    half_size=main_wall_half_size,
                    position=main_wall_position,
                    material=main_wall_material,
                )
                self.left_wall = sapien_utils.add_wall(
                    self.scene,
                    half_size=left_wall_half_size,
                    position=left_wall_position,
                    material=left_wall_material,
                )
                self.right_wall = sapien_utils.add_wall(
                    self.scene,
                    half_size=right_wall_half_size,
                    position=right_wall_position,
                    material=right_wall_material,
                )
            else:
                self.main_wall = sapien_utils.add_wall(
                    self.scene,
                    half_size=main_wall_half_size,
                    position=main_wall_position,
                )
                self.left_wall = sapien_utils.add_wall(
                    self.scene,
                    half_size=left_wall_half_size,
                    position=left_wall_position,
                )
                self.right_wall = sapien_utils.add_wall(
                    self.scene,
                    half_size=right_wall_half_size,
                    position=right_wall_position,
                )

    def _load_specified_counter(self, counter_range, mode) -> None:
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
        material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        if mode == "test":
            self.counter = sapien_utils.add_counter(
                self.scene, half_size=half_size, position=pos
            )
        else:
            if np.random.rand() > 0.1:
                self.counter = sapien_utils.add_counter(
                    self.scene,
                    half_size=half_size,
                    position=pos,
                    material=material,
                )
            else:
                self.counter = sapien_utils.add_counter(
                    self.scene, half_size=half_size, position=pos
                )

    def _load_random_table(self) -> None:
        if self.table is not None:
            self.scene.remove_actor(self.table)
        material = sapien_utils.render_material_from_ambient_cg_texture(
            self.renderer, np.random.choice(self.textures)
        )
        table_half_size = [
            np.random.uniform(0.4, 0.7),
            np.random.uniform(0.4, 0.7),
            np.random.uniform(0.01, 0.01),
        ]
        table_position = [0.0, 0.0, 0.2 - table_half_size[2]]
        self.table = sapien_utils.add_table(
            self.scene,
            half_size=table_half_size,
            position=table_position,
            material=material,
        )
        return

    def _sample_random_camera_poses(self, objs_center: np.ndarray) -> np.ndarray:
        camera_poses = sample_cam_poses_shell(
            center=objs_center, coi_half_size=0.05, num_poses=self.imgs_per_scene
        )
        return camera_poses

    def _render_obs(self, cam_pose: np.ndarray) -> sapien_utils.CameraObs:
        self.sensor.set_pose(sapien.Pose.from_transformation_matrix(cam_pose))
        self.scene.update_render()
        if self.lock is not None:
            self.lock.acquire()
        camera_obs = sapien_utils.get_sensor_obs(self.sensor, self.camera_obs_config)
        if self.lock is not None:
            self.lock.release()
        return camera_obs

    def _objs_pose_in_cam_frame(
        self, objs: list[SceneObject], cam_pose: np.ndarray
    ) -> np.ndarray:
        camTobjs = np.array([np.linalg.inv(cam_pose) @ obj.pose4x4 for obj in objs])
        return camTobjs

    def _load_obj(self, data_dir, json_data, randomize_obj):
        # Load robot
        urdf_path = os.path.join(data_dir, json_data["object_name"], "mobility.urdf")
        volume = json_data["volume"]
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = json_data["scale"]
        loader.load_multiple_collisions_from_file = True
        robot: sapien.Articulation = loader.load(urdf_path)
        robot.set_qpos(np.array([json_data["joint_state"]]))
        new_robot_position = [
            self.table.pose.p[0],
            self.table.pose.p[1],
            (volume[2] / 2) + self.table.pose.p[2] - 0.06,
        ]
        robot.set_root_pose(sapien.Pose(new_robot_position, [0, 0, 0, 1]))
        if randomize_obj:
            render_material = sapien_utils.render_material_from_ambient_cg_texture(
                self.renderer, np.random.choice(self.textures)
            )
            for link in robot.get_links():
                for body in link.get_collision_visual_bodies():
                    for shape in body.get_render_shapes():
                        shape.set_material(render_material)
        return

    def _setup_scene(self, counter_range, mode) -> None:
        self.renderer.clear_cached_resources()
        self._load_random_floor(mode)
        self._load_random_walls(mode)
        self._load_specified_counter(counter_range, mode)
        self._randomize_light()
        self.scene.step()
        return

    def _make_gt(
        self, objs: list[SceneObject], camera_pose: np.ndarray, counter_range: list
    ) -> GTData:
        camera_obs = self._render_obs(camera_pose)
        indices = []
        for obj in objs:
            link_list = obj.articulation.get_links()
            indices.append([link.id for link in link_list])

        segmentation = camera_obs.segmentation
        heatmap, simple_bmasks = heatmap_from_segmentation(
            segmentation=camera_obs.segmentation, indices=indices
        )
        
        camTposes = self._objs_pose_in_cam_frame(objs, camera_pose)

        return GTData(
            camera_obs.rgb,
            camera_obs.depth_gt,
            camera_obs.depth_real,
            camTposes,
            simple_bmasks,
            heatmap,
            segmentation,
            camera_pose,
            np.array(counter_range),
        )

    def make_data(
        self, objs: list[SceneObject], counter_range, mode, y_center: float = 0.0
    ) -> tuple[list[GTData], list[dict]]:
        self._setup_scene(counter_range, mode)
        objs_center = np.mean([obj.pose4x4[:3, 3] for obj in objs], axis=0)
        objs_center[1] = y_center
        objs_info = [
            {
                "object_id": obj.id,
                "emb_index": obj.emb_index,
                "joint_state": obj.normalized_joint_state,
            }
            for obj in objs
        ]
        camera_poses = self._sample_random_camera_poses(objs_center)
        gt_data_list = [
            self._make_gt(objs, camera_pose, counter_range)
            for camera_pose in camera_poses
        ]
        return gt_data_list, objs_info
