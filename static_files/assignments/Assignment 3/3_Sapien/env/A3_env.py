# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from .stacking_env import StackingEnv
import sapien.core as sapien
import numpy as np
from typing import List, Tuple, Sequence

class A3Env(StackingEnv):
    def __init__(self, timestep: float):
        """Class for A3

        Args:
            timestep: timestep to balance the precision and the speed of physical simulation
        """
        StackingEnv.__init__(self, timestep)

        self.near, self.far = 0.1, 100
        self.camera_link = self.scene.create_actor_builder().build(is_kinematic=True)
        self.gl_pose = sapien.Pose([0, 0, 0], [0.5, -0.5, 0.5, -0.5])
        self.camera_link.set_pose(sapien.Pose([1.2, 0.0, 0.8], [0, -0.258819, 0, 0.9659258]))
        self.camera = self.scene.add_mounted_camera('fixed_camera', self.camera_link,
                                                    sapien.Pose(), 1920, 1080,
                                                    np.deg2rad(50), np.deg2rad(50), self.near, self.far)

        self.arm_joints = [joint for joint in self.robot.get_joints() if
                           joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]
        self.set_joint_group_property(self.arm_joints, 1000, 400)
        assert len(self.arm_joints) == self.robot.dof - 2
        self.set_joint_group_property(self.gripper_joints, 200, 60)

        self.step()
        self.robot.set_drive_target(self.robot.get_qpos())

    def cam2base_gt(self) -> np.ndarray:
        """Get ground truth transformation of camera to base transformation

        Returns:
            Ground truth transformation from camera to robot base

        """
        camera_pose = self.camera_link.get_pose() * self.gl_pose
        base_pose = self.robot.get_root_pose()
        return self.pose2mat(base_pose.inv() * camera_pose)

    def get_current_ee_pose(self) -> np.ndarray:
        """Get current end effector pose for calibration calculation

        Returns:
            Transformation from end effector to robot base

        """
        return self.pose2mat(self.end_effector.get_pose())

    def get_object_point_cloud(self, seg_id: int) -> np.ndarray:
        """Fetch the object point cloud given segmentation id

        For example, you can use this function to directly get the point cloud of a colored box and use it for further
        calculation.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()

        Returns:
            (3, n) dimension point cloud in the camera frame with(x, y, z) order

        """
        self.scene.update_render()
        self.camera.take_picture()
        camera_matrix = self.camera.get_camera_matrix()[:3, :3]
        gl_depth = self.camera.get_depth()
        y, x = np.where(gl_depth < 1)
        z = self.near * self.far / (self.far + gl_depth * (self.near - self.far))

        point_cloud = (np.dot(np.linalg.inv(camera_matrix),
                              np.stack([x, y, np.ones_like(x)] * z[y, x], 0)))

        seg_mask = self.camera.get_segmentation()[y, x]
        selected_index = np.nonzero(seg_mask == seg_id)[0]
        return point_cloud[:, selected_index]

    @staticmethod
    def compute_ik(ee2base: np.ndarray) -> List[List[float]]:
        """Compute the inverse kinematics of franka panda robot.

        This function is provided to help do the inverse kinematics calculation.
        The output of this function is deterministic.
        It will return a list of solutions for the given cartesian pose.
        In practice, some solutions are not physically-plausible due to self collision.
        So in this homework, you may need to choose the free_joint_value and which solution to use by yourself.

        References:
            ikfast_pybind:
            ikfast_pybind is a python binding generation library for the analytic kinematics engine IKfast.
            Please visit here for more information: https://pypi.org/project/ikfast-pybind/

            ikfast:
            ikfast is a powerful inverse kinematics solver provided within
            Rosen Diankov’s OpenRAVE motion planning software.
            Diankov, R. (2010). Automated construction of robotic manipulation programs.

        Args:
            ee2base: transformation from end effector to base

        Returns:
            A list of possible IK solutions when the last joint value is set as free_joint_value

        """
        try:
            import ikfast_franka_panda as panda
        except ImportError:
            print("Please install ikfast_pybind before using this function")
            print("Install: pip3 install ikfast-pybind")
            raise ImportError

        link72ee = np.array([[0.7071068, -0.7071068, 0, 0], [0.7071068, 0.7071068, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        link7_pose = ee2base @ link72ee
        pos = link7_pose[:3, 3]
        rot = link7_pose[:3, :3]
        return panda.get_ik(pos, rot, [0.785])

    def close_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.001)

    def open_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.04)

    def clear_velocity_command(self):
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(0)

    def wait_n_steps(self, n: int):
        self.clear_velocity_command()
        for i in range(n):
            passive_force = self.robot.compute_passive_force()
            self.robot.set_qf(passive_force)
            self.step()
            self.render()
        self.robot.set_qf([0] * self.robot.dof)

    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.

        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == len(self.arm_joints)
        target_qpos = qvel * self.scene.get_timestep() + self.robot.get_drive_target()[:-2]
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def evaluate_first_two_box(self) -> bool:
        """Evaluate whether you stack the first two boxes successfully"""
        position, size = self.target
        rbox, gbox, _ = self.boxes
        contacts = self.scene.get_contacts()

        red_target_position = np.array([position[0], position[1], size])
        green_target_position = np.array([position[0], position[1], 3 * size])
        red_in_place = np.linalg.norm(rbox.get_pose().p - red_target_position) < 0.01
        green_in_place = np.linalg.norm(gbox.get_pose().p - green_target_position) < 0.01
        return green_in_place and red_in_place

    def evaluate_final_result(self) -> bool:
        """Evaluate whether you stack the all three boxes successfully"""
        position, size = self.target
        rbox, gbox, bbox = self.boxes
        static = (rbox.velocity @ rbox.velocity) < 1e-5 and \
                 (gbox.velocity @ gbox.velocity) < 1e-5 and \
                 (bbox.velocity @ bbox.velocity) < 1e-5
        if not static:
            return False

        first_two_box_success = self.evaluate_first_two_box()
        blue_target_position = np.array([position[0], position[1], 5 * size])
        blue_in_place = np.linalg.norm(bbox.get_pose().p - blue_target_position) < 0.01
        return blue_in_place and first_two_box_success

    ####################################################################################################################
    # ============================== You will need to implement all the functions below ================================
    ####################################################################################################################
    @staticmethod
    def pose2mat(pose: sapien.Pose) -> np.ndarray:
        """You need to implement this function

        You will need to implement this function first before any other functions.
        In this function, you need to convert a (position: pose.p, quaternion: pose.q) into a SE(3) matrix

        You can not directly use external library to transform quaternion into rotation matrix.
        Only numpy can be used here.
        Args:
            pose: sapien Pose object, where Pose.p and Pose.q are position and quaternion respectively

        Hint: the convention of quaternion

        Returns:
            (4, 4) transformation matrix represent the same pose

        """

        raise NotImplementedError

    def pose2exp_coordinate(self, pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """You may need to implement this function

        Compute the exponential coordinate corresponding to the given SE(3) matrix
        Note: unit twist is not a unit vector

        Args:
            pose: (4, 4) transformation matrix

        Returns:
            Unit twist: (6, ) vector represent the unit twist
            Theta: scalar represent the quantity of exponential coordinate
        """

        raise NotImplementedError


    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """You need to implement this function

        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        ee_jacobian = np.zeros([6, self.robot.dof - 2])  # (6, 7)
        ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3:self.end_effector_index * 6, :7]
        ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6:self.end_effector_index * 6 - 3, :7]

        raise NotImplementedError

    def move_to_target_pose_with_internal_controller(self, target_ee_pose: np.ndarray, num_steps: int) -> None:
        """You need to implement this function

        Move the robot hand dynamically to a given target pose
        You may need to call self.internal_controller and your self.compute_joint_velocity_from_twist in this function

        To make command (e.g. internal controller) take effect and simulate all the physical effects, you need to step
        the simulation world for one step and render the new scene for visualization by something like:
            for i in range(num_step):
                # Do something
                self.internal_controller(target_joint_velocity)
                self.step()
                self.render()

        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps: how much steps to reach to target pose, each step correspond to self.scene.get_timestep() seconds
                in physical simulation

        """
        executed_time = num_steps * self.scene.get_timestep()

        raise NotImplementedError

    def pick_up_object_with_internal_controller(self, seg_id: int, height: float) -> None:
        """You need to implement this function

        Pick up a specific box to a target height using the given internal controller


        You can use the following function to get the segmented point cloud:
            point_cloud = self.get_object_point_cloud(seg_id)

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            height: target height of the box

        """

        raise NotImplementedError

    def place_object_with_internal_controller(self, seg_id: int, target_object_position: np.ndarray) -> None:
        """You need to implement this function

        Place a specific box to a target position
        This function do not consider rotation, so you can just assume a fixed rotation or any orientation you want

        After place the box, you also need to move the gripper to some safe place. Thus when you pick up next box, it
        will not jeopardise your previous result.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            target_object_position: target position of the box

        """

        raise NotImplementedError
