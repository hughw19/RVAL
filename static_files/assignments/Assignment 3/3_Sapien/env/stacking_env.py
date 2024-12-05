# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

import sapien.core as sapien
from sapien.core import Pose
import numpy as np
import os

sim = sapien.Engine()
renderer = sapien.OptifuserRenderer()
renderer.enable_global_axes(False)
sim.set_renderer(renderer)
renderer_controller = sapien.OptifuserController(renderer)
renderer_controller.set_camera_position(-0.7, 1.32, 1.27)
renderer_controller.set_camera_rotation(-1, -0.6)


class StackingEnv:
    def __init__(self, timestep=1 / 500):
        self.scene = sim.create_scene(gravity=[0, 0, -9.81])
        self.super_friction_material = sim.create_physical_material(4, 4, 0.01)
        renderer_controller.set_current_scene(self.scene)
        self.scene.add_ground(altitude=0)
        self.scene.set_timestep(timestep)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([-1, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_directional_light([-1, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([2, 0, 2], [1, 1, 1])
        self.scene.add_point_light([0.5, 1.5, 2], [1, 1, 1])
        self.scene.add_point_light([0.5, -1.5, 2], [1, 1, 1])

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 "../assets/robot/panda_checkerboard.urdf")
        self.robot = loader.load(file_path, self.super_friction_material)
        for joint in self.robot.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)
        self.robot.set_qpos(np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, 0.4, 0.4]))

        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'panda_hand'][0]
        self.gripper_joints = [joint for joint in self.robot.get_joints() if
                               joint.get_name().startswith("panda_finger_joint")]
        self.target = None
        self.setup_target([0.5, 0])
        self.setup_boxes()
        self.step_ = 0
        self._windows = False

        self.render_rate = 1
        self.render_step_ = 0

        self.step()

    def render(self):
        self.render_step_ += 1
        if not self._windows:
            self._windows = True
            renderer_controller.show_window()
        self.scene.update_render()
        if self.render_step_ % self.render_rate == 0:
            renderer_controller.render()

    def step(self):
        self.step_ += 1
        self.scene.step()

    def close_render(self):
        self._windows = False
        renderer_controller.hide_window()

    def setup_boxes(self):
        object_material = sim.create_physical_material(1, 1, 0.1)
        size = self.target[1]
        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[size] * 3, material=object_material)
        b.add_box_visual(size=[size] * 3, color=[1, 0, 0])
        red_box = b.build()
        red_box.set_pose(Pose([0.35, 0, size]))
        red_box.set_name('red')

        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[size] * 3, material=object_material)
        b.add_box_visual(size=[size] * 3, color=[0, 1, 0])
        green_box = b.build()
        green_box.set_pose(Pose([0.35, 0.2, size]))
        green_box.set_name('green')

        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[size] * 3, material=object_material)
        b.add_box_visual(size=[size] * 3, color=[0, 0, 1])
        blue_box = b.build()
        blue_box.set_pose(Pose([0.35, -0.2, size]))
        blue_box.set_name('blue')

        self.boxes = red_box, green_box, blue_box

    def setup_target(self, position, size=0.02):
        self.target = (position, size)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(size=[size] * 3, color=[0, 1, 1])
        indicator = builder.build_static('goal')
        indicator.set_pose(Pose([position[0], position[1], -size + 0.001]))

    def success(self):
        position, size = self.target
        rbox, gbox, bbox = self.boxes
        static = (rbox.velocity @ rbox.velocity) < 1e-5 and \
                 (gbox.velocity @ gbox.velocity) < 1e-5 and \
                 (bbox.velocity @ bbox.velocity) < 1e-5
        if not static:
            return False

        if not all(np.abs(position - rbox.pose.p[:2]) < size):
            return False

        contacts = self.scene.get_contacts()
        r_touch_g = any(
            [{c.actor1, c.actor2} == {rbox, gbox} for c in contacts if c.separation < 1e-2])
        g_touch_b = any(
            [{c.actor1, c.actor2} == {gbox, bbox} for c in contacts if c.separation < 1e-2])
        return r_touch_g and g_touch_b and bbox.pose.p[2] > gbox.pose.p[2] > rbox.pose.p[2]

    def success_test(self):
        self.scene.remove_articulation(self.robot)
        self.boxes[0].set_pose(Pose([0.6, 0, 0.03]))
        self.boxes[1].set_pose(Pose([0.61, 0.01, 0.15]))
        self.boxes[2].set_pose(Pose([0.59, -0.01, 0.3]))

    def should_quit(self):
        return renderer_controller.should_quit

    def close(self):
        renderer_controller.set_current_scene(None)
        self.scene = None

    @staticmethod
    def set_joint_group_property(joints, stiffness, damping):
        for joint in joints:
            joint.set_drive_property(stiffness, damping)
