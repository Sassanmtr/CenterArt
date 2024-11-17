import os
import numpy as np
import time
import spatialmath as sm
from centerart_model.sapien.sapien_envs import GraspEnv


class GraspBehavior:
    def __init__(
            self,
            environment: GraspEnv,
            pipeline: None,
    ):
        self.environment = environment
        self.pipeline = pipeline

    def set_pose(self, bTtarget: sm.SE3):
        self.environment.sapien_robot.set_root_pose(bTtarget.A)
        return

    def move_to_pose(self, bTtarget: sm.SE3):
        speed = 0.01
        delta_space = bTtarget.t - self.environment.sapien_robot.get_root_pose()[:3, 3]
        distance = np.linalg.norm(delta_space)
        direction = delta_space / distance
        velocity = direction * speed
        time = distance / speed
        self.environment.sapien_robot.set_root_velocity(velocity, np.zeros(3))
        self.environment.step_physics(time)
        self.environment.sapien_robot.set_root_velocity(np.zeros(3), np.zeros(3))
        return

    def move_to_pose_ang(
        self,
        obj_idx: int,
        current_ee_pose: sm.SE3,
        next_ee_pose: sm.SE3,
        mode: str,
    ):
        speed = 0.1
        relative_transform = current_ee_pose.inv() * next_ee_pose
        new_dist = next_ee_pose.t - current_ee_pose.t
        error_t = new_dist
        error_rpy = sm.smb.tr2rpy(
            relative_transform.R, unit="rad", order="zyx", check=False
        )
        linear_velocity = error_t * speed
        angular_velocity = error_rpy * speed
        if mode == "open":
            angular_velocity = -angular_velocity
        elif mode == "close":
            linear_velocity = -linear_velocity

        self.environment.sapien_robot.set_root_velocity(
            linear_velocity,
            angular_velocity,
        )
        distance = np.linalg.norm(error_t)
        time = distance / speed
        self.environment.step_physics(time)
        current_pose = self.environment.scene_objs[obj_idx].articulation.get_qpos()
        return current_pose

    def get_forward_vector(self, grasp_pose, distance):
        """
        Calculate the forward vector based on a position, orientation, and distance.
        """
        rotation_matrix = grasp_pose.R
        # Define the forward vector (assuming Z-axis is the forward direction)
        forward_vector = np.array([0, 0, 1])
        rotated_forward_vector = rotation_matrix @ forward_vector
        forward_vector_scaled = (
            distance * rotated_forward_vector / np.linalg.norm(rotated_forward_vector)
        )
        res = np.eye(4)
        res[:3, :3] = rotation_matrix
        res[:3, 3] = grasp_pose.t + forward_vector_scaled
        res = sm.SE3(res, check=False)
        return res

    def run(self, obs, data_idx, method) -> dict:
        success_grasps = 0
        attempts = 0
        completed_runs = 0
        abborted = 0
        obs.camera_pose = sm.SE3(obs.camera_pose, check=False)
        if method == "umpnet":
            wTee_list = self.pipeline.predict_grasp(data_idx)
        else:
            wTee_list = self.pipeline.predict_grasp(obs)
        if len(wTee_list) < len(self.environment.scene_objs):
            info = {
                "attempts": 0,
                "aborted_runs": 0,
                "completed_runs": 0,
                "success_grasps": 0,
            }
            return info
        # sort the grasps based on the y axis
        wTee_list = sorted(wTee_list, key=lambda x: x.t[1])
        # sort scene objects based on the y axis
        self.environment.scene_objs = sorted(
            self.environment.scene_objs, key=lambda x: x.pose4x4[1, 3]
        )
        gt_grasp_poses = self.get_gt_grasps(self.environment.scene_objs)

        for obj_idx in range(len(self.environment.scene_objs)):
            init_state = float(self.environment.scene_objs[obj_idx].joint_state)
            max_joint_state = self.environment.scene_objs[obj_idx].max_joint_state
            if max_joint_state - init_state < 0.8:
                mode = "close"
            else:
                mode = "open"
            # TODO: Remove try except
            try:
                link_direction = self.environment.scene_objs[obj_idx].link_direction
            except AttributeError:
                link_direction = "top"

            gt_grasp = gt_grasp_poses[obj_idx][10]
            gt_grasp = self.get_forward_vector(gt_grasp, 0.01)

            run_complete, current_state = self.run_behavior(
                obj_idx, gt_grasp, init_state, mode, link_direction
            )
            grasp_success = self.environment.evaluate_success(
                obj_idx, mode, current_state
            )
            self.environment.reset_robot()
            if run_complete:
                completed_runs += 1
                if grasp_success:
                    success_grasps += 1
            else:
                abborted += 1
            attempts += 1

        info = {
            "attempts": attempts,
            "aborted_runs": abborted,
            "completed_runs": completed_runs,
            "success_grasps": success_grasps,
        }
        return info

    def run_relaxed(self, obs, data_idx, method) -> dict:
        attempts = 0
        distances = []
        obs.camera_pose = sm.SE3(obs.camera_pose, check=False)
        if method == "umpnet":
            wTee_list = self.pipeline.predict_grasp(data_idx)
        else:
            wTee_list = self.pipeline.predict_grasp(obs)
        if len(wTee_list) < len(self.environment.scene_objs):
            info = {"attempts": 0, "distances": distances}
            return info
        # sort the grasps based on the y axis
        wTee_list = sorted(wTee_list, key=lambda x: x.t[1])
        # sort scene objects based on the y axis
        self.environment.scene_objs = sorted(
            self.environment.scene_objs, key=lambda x: x.pose4x4[1, 3]
        )
        gt_grasp_poses = self.get_gt_grasps(self.environment.scene_objs)
        for obj_idx, _ in enumerate(self.environment.scene_objs):
            # Take the minimum distance between the predicted grasp wTee_list[obj_idx] and the ground truth grasps gt_grasp_poses[obj]
            min_distance = 1000
            for gt_grasp in gt_grasp_poses[obj_idx]:
                wTee = self.get_forward_vector(wTee_list[obj_idx], 0.22)
                distance = np.linalg.norm(wTee.t - gt_grasp.t)
                if distance < min_distance:
                    min_distance = distance
            distances.append(min_distance)
        info = {"attempts": attempts, "distances": distances}
        return info

    def get_gt_grasps(self, scene_objs):
        grasp_dir = os.path.join(self.environment.data_dir, "datasets", "grasps")
        gt_grasp_poses = {}
        for obj_idx, obj in enumerate(scene_objs):
            obj_name = obj.id
            obj_scale = str(obj.scale)
            obj_joint_state = obj.joint_state
            file_name = f"{obj_name}_{obj_scale}_{obj_joint_state}.npy"
            grasp_file = os.path.join(grasp_dir, file_name)
            transform = sm.SE3(obj.pose4x4, check=False)
            raw_grasp_poses = np.load(grasp_file, allow_pickle=True)
            grasp_poses = [
                transform * sm.SE3(pose, check=False) for pose in raw_grasp_poses
            ]
            gt_grasp_poses[obj_idx] = grasp_poses
        return gt_grasp_poses

    def run_behavior(self, j, wTee, init_state, mode, link_direction):
        # Run pipeline
        print(
            "init dist: ",
            np.abs(self.environment.scene_objs[j].articulation.get_qpos() - init_state),
        )

        if (
            np.abs(self.environment.scene_objs[j].articulation.get_qpos() - init_state)
            > 0.01
        ):
            return False, self.environment.scene_objs[j].articulation.get_qpos()[0]
        # wTee = self.get_forward_vector(wTee, 0.22)
        # wTee = self.get_forward_vector(wTee, 0.09)
        # wTee = self.get_forward_vector(wTee, 0.0266)
        wTee = wTee.A.reshape(4, 4)
        wTee = sm.SE3(wTee, check=False)
        wTbase = self.get_forward_vector(wTee, -0.1034)
        wTbase = sm.SE3(wTbase, check=False)
        wTbase_pregrasp = self.get_forward_vector(wTbase, -0.1)
        self.set_pose(wTbase_pregrasp)

        ## Move to grasp
        self.move_to_pose(wTbase)

        # if robot did not move the run is aborted
        cur_pose = self.environment.sapien_robot.get_root_pose()
        if np.linalg.norm(cur_pose - wTbase_pregrasp.A) < 0.02:
            current_pose = self.environment.scene_objs[j].articulation.get_qpos()

            print(
                "ABORTED with initail distance: ",
                np.linalg.norm(cur_pose - wTbase_pregrasp.A),
            )
            return False, current_pose

        # Close gripper
        self.environment.close_gripper()
        # If the time is more than 10 seconds terminate the
        start_time = time.time()
        for i in range(6):
            current_wTbase = (
                self.environment.sapien_robot.robot.pose.to_transformation_matrix()
            )
            current_wTbase = sm.SE3(current_wTbase, check=False)
            current_wTee = self.get_forward_vector(current_wTbase, 0.1034)
            wThinge = (
                self.environment.scene_objs[0]
                .articulation.get_active_joints()[0]
                .get_global_pose()
                .to_transformation_matrix()
            )
            if link_direction == "top":
                wThinge[:3, :3] = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
            else:
                wThinge[:3, :3] = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
            # if link_direction is equal to "left", switch the mode
            new_mode = mode
            if link_direction == "left":
                new_mode = "close" if mode == "open" else "open"
            wThinge_SE = sm.SE3(wThinge, check=False)
            hingeTee = wThinge_SE.inv() * current_wTee
            theta = np.pi / 6
            rotation_matrix = sm.SE3.AngleAxis(theta, [1.0, 0.0, 0.0])
            next_pose = wThinge_SE * rotation_matrix * hingeTee

            current_pose = self.move_to_pose_ang(j, current_wTee, next_pose, new_mode)
            if time.time() - start_time > 10:
                return True, current_pose
            
            if mode == "open":
                if current_pose[0] - init_state >= 0.27:
                    break
            elif mode == "close":
                if current_pose[0] - init_state <= 0.27:
                    break
        return True, current_pose
