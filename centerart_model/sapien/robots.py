import numpy as np
import sapien.core as sapien
from abc import ABC, abstractmethod
from centerart_model.utils.configs import Directories
import centerart_model.sapien.sapien_utils as sapien_utils

class SapienRobot(ABC):
    @property
    @abstractmethod
    def q_home(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def robot(self) -> sapien.Articulation:
        pass

    @abstractmethod
    def initialize(self, scene: sapien.Scene):
        pass

    @abstractmethod
    def reset(self, qpos: np.ndarray):
        pass

    @abstractmethod
    def open_gripper(self):
        pass

    @abstractmethod
    def close_gripper(self):
        pass

    @abstractmethod
    def update_qf(self):
        pass

    @abstractmethod
    def set_arm_targets(self, position: np.ndarray, velocity: np.ndarray):
        pass

    def get_root_pose(self) -> np.ndarray:
        return self.robot.get_root_pose().to_transformation_matrix()

    def set_root_pose(self, pose: np.ndarray):
        self.robot.set_root_pose(sapien.Pose.from_transformation_matrix(pose))
        return

    def get_joint_state(self) -> np.ndarray:
        return self.robot.get_qpos()

    @abstractmethod
    def set_root_velocity(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        pass

    def set_qpos_target(self, qpos_target: np.ndarray):
        self.robot.set_drive_target(qpos_target)
        return


class GripperSapien(SapienRobot):
    START_ROOT_POSE = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 1.5],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, q_home: np.ndarray = np.array([0.04, 0.04])):
        self._q_home = q_home
        return

    @property
    def q_home(self):
        return self._q_home

    @property
    def robot(self) -> sapien.Articulation:
        return self._robot

    def add_franka_gripper(
        self, scene: sapien.Scene, pose: np.ndarray = np.eye(4)
    ) -> sapien.Articulation:
        gripper_urdf_path = Directories.FRANKA / "hand.urdf"
        robot = sapien_utils.add_robot(
            scene, gripper_urdf_path, pose, fix_root_link=False
        )
        assert robot.dof == 2  # 2 fingers
        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=10000, damping=100)
        # Increase friction of fingers
        physical_material = scene.create_physical_material(
            static_friction=1e5, dynamic_friction=1e5, restitution=0.0
        )
        for link in robot.get_links():
            for shape in link.get_collision_shapes():
                shape.set_physical_material(physical_material)
        return robot

    def initialize(self, scene: sapien.Scene) -> sapien.Articulation:
        robot = self.add_franka_gripper(scene)
        self.gripper_joints = robot.get_active_joints()

        # Add drive to root, so that it floats
        drive = scene.create_drive(
            None,
            sapien.Pose(),
            robot.get_links()[0],
            sapien.Pose(),
        )
        drive.set_x_properties(stiffness=40, damping=1e6)
        drive.set_y_properties(stiffness=40, damping=1e6)
        drive.set_z_properties(stiffness=40, damping=1e6)
        drive.set_x_twist_properties(stiffness=40, damping=1e6)
        drive.set_yz_swing_properties(stiffness=40, damping=1e6)
        self.drive = drive
        self._robot = robot
        self.reset()
        return

    def set_root_velocity(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        self.drive.set_target_velocity(linear_vel, angular_vel)
        return

    def reset(self):
        self.robot.set_root_pose(
            sapien.Pose.from_transformation_matrix(self.START_ROOT_POSE)
        )
        self.robot.set_qpos(self.q_home)
        self.robot.set_drive_target(self.q_home)
        self.fq_target = np.zeros(2)
        return

    def open_gripper(self):
        self.fq_target = np.zeros(2)
        self.gripper_joints[0].set_drive_target(0.04)
        self.gripper_joints[1].set_drive_target(0.04)
        return

    def close_gripper(self):
        self.fq_target = np.zeros(2)
        self.fq_target[7:] = -5
        self.gripper_joints[0].set_drive_target(0.0)
        self.gripper_joints[1].set_drive_target(0.0)
        return

    def update_qf(self):
        qf = self.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True, external=False
        )
        self.robot.set_qf(qf + self.fq_target)
        return

    def set_arm_targets(self, position: np.ndarray, velocity: np.ndarray):
        raise NotImplementedError("Gripper does not have arm joints")


ROBOTS_DICT = {
    "gripper": GripperSapien,
}
