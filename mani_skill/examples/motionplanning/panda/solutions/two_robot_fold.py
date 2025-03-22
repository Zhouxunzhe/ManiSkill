import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotFoldEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: TwoRobotFoldEnv, seed=None, debug=False, vis=False):
    options = {
        "reconfigure": True
    }
    env.reset(options=options)
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0
    )
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # get information about articulated suitcase

    # # retrieves the object oriented bounding box (trimesh box object)
    # obb = get_articulate_obb(env.lid_links_meshes)
    #
    # approaching = np.array([0, 0, -1])
    # # get transformation matrix of the tcp pose, is default batched and on torch
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.suitcase.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    right_planner.move_to_pose_with_screw(env.right_waypoint1, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(env.right_waypoint2, other_gripper_state=left_planner.gripper_state)
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(env.right_waypoint3, other_gripper_state=left_planner.gripper_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    right_res = right_planner.move_to_pose_with_screw(env.right_waypoint4, other_gripper_state=left_planner.gripper_state)

    # left_planner.move_to_pose_with_screw(env.left_waypoint1, other_gripper_state=right_planner.gripper_state)
    # left_planner.move_to_pose_with_screw(env.left_waypoint2, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.left_waypoint3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.left_waypoint4, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.left_waypoint5, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.left_waypoint6, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.left_waypoint7, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.left_waypoint8, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(env.left_waypoint9, other_gripper_state=right_planner.gripper_state)

    left_planner.close()
    right_planner.close()

    return left_res, right_res
