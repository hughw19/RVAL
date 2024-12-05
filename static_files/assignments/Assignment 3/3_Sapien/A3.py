# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from env.A3_env import A3Env
import numpy as np

def main():
    np.set_printoptions(precision=4)
    env = A3Env(timestep=1 / 500)

    # Increase this rate to a higher value if you find your visualization is too slow
    # E.g. if you are using a old mac book pro, you can do: env.render_rate = 5
    env.render_rate = 1
    env.wait_n_steps(100)

    print("Press q to continue")
    while not env.should_quit():
        env.render()

    # ===============================Test Basic Function==============================
    test_pose = np.array(
        [[0.57797184, 0.42202815, 0.698456, 0.99482830], [0.42202815, 0.57797184, -0.69845599, 0.00517169],
         [-0.698456, 0.698456, 0.15594369, 1.81894014], [0, 0, 0, 1]])
    test_theta = 1.4142135
    test_twist = np.array([1, 1, 0, 0, 1, 2]) / test_theta
    unit_twist, theta = env.pose2exp_coordinate(test_pose)
    assert np.allclose(unit_twist, test_twist) and np.allclose(test_theta, theta)

    target_pose = env.get_current_ee_pose()
    target_pose[2, 3] -= 0.1
    env.move_to_target_pose_with_internal_controller(target_pose, 100)
    env.wait_n_steps(250)
    pose_after_move = env.get_current_ee_pose()
    translational_error = target_pose[:3, 3] - pose_after_move[:3, 3]
    print("Translational error for each direction is: {}".format(translational_error))
    assert np.allclose(target_pose[:3, 3], pose_after_move[:3, 3], atol=3e-3), "transitional error is too larger"

    # ===================================Preparation==================================
    red_box, green_box, blue_box = env.boxes
    position, size = env.target
    pick_up_height = 0.2
    red_target_position = np.array([position[0], position[1], size])
    green_target_position = np.array([position[0], position[1], size * 3])
    blue_target_position = np.array([position[0], position[1], size * 5])

    # =============================Place Red and Green Box============================
    red_height_before_pick = red_box.get_pose().p[2]
    env.pick_up_object_with_internal_controller(red_box.get_id(), pick_up_height)
    red_height_after_pick = red_box.get_pose().p[2]
    print("Target height of red box: {}, achieved height of red box: {}".format(pick_up_height + red_height_before_pick,
                                                                            red_height_after_pick))
    assert np.abs(pick_up_height + red_height_before_pick - red_height_after_pick) < 0.02, "Lift red box task fail."

    env.place_object_with_internal_controller(red_box.get_id(), red_target_position)
    env.pick_up_object_with_internal_controller(green_box.get_id(), pick_up_height)
    env.place_object_with_internal_controller(green_box.get_id(), green_target_position)

    env.wait_n_steps(50)
    evaluate_first_two_box(env)


def evaluate_first_two_box(env: A3Env):
    if env.evaluate_first_two_box():
        print("Successfully stack the first two boxes to target place")
        print("Press q to continue")
        while not env.should_quit():
            env.render()
    else:
        print("Fail to stack the first two boxes to target place")
        print("Observe what happen and press q to quit")
        while not env.should_quit():
            env.render()
        env.scene = None
        exit(0)
    return


if __name__ == '__main__':
    main()
