import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPickCubeYCBEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotPickCubeYCBEnv, seed=None, debug=False, vis=False):
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

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    mid_pose1 = sapien.Pose(p=[0.1, 0, 0.4], q=env.left_agent.tcp.pose.q[0])
    mid_pose2 = sapien.Pose(p=[0.1, 0, 0.4], q=env.left_agent.tcp.pose.q[0]*euler2quat(np.pi, -np.pi, np.pi))
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    goal_pose = sapien.Pose(env.obj.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])
    reach_pose2 = goal_pose * sapien.Pose([0, 0, -0.5])
    reach_pose3 = goal_pose * sapien.Pose([0, 0, -0.1])

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.05])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    left_planner.move_to_pose_with_screw(mid_pose1, other_gripper_state=right_planner.gripper_state)
    right_planner.move_to_pose_with_screw(mid_pose2, other_gripper_state=left_planner.gripper_state)
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)
    # right_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=left_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
