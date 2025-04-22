import numpy as np
import sapien
import random
from transforms3d.euler import euler2quat
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from mani_skill.envs.tasks import PickCubeYCBEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PickCubeYCBEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    RAND_WEIGHT = 0.04
    # RAND_WEIGHT = 0.0
    env = env.unwrapped

    obb = get_actor_obb(env.source_obj)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.source_obj.pose.sp.p)
    reach_pose1 = grasp_pose * sapien.Pose([
        0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
        0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
        -0.3 + np.random.uniform(-1, 1) * RAND_WEIGHT
    ])
    # reach_pose2 = grasp_pose * sapien.Pose([
    #     0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
    #     0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
    #     -0.2 + np.random.uniform(-1, 1) * RAND_WEIGHT
    # ])
    reach_pose3 = grasp_pose * sapien.Pose([
        0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
        0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
        -0.1 + np.random.uniform(-1, 1) * RAND_WEIGHT
    ])

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(reach_pose1)
    # planner.move_to_pose_with_screw(reach_pose2)
    planner.move_to_pose_with_screw(reach_pose3)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    goal_pose = sapien.Pose(env.target_obj.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])
    reach_pose4 = goal_pose * sapien.Pose([
        0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
        0 + np.random.uniform(-1, 1) * RAND_WEIGHT,
        -0.2 + np.random.uniform(-1, 1) * RAND_WEIGHT
    ])
    if env.is_pour:
        reach_pose4.q = euler2quat(np.pi / 2, np.pi / 2, np.pi)
    planner.move_to_pose_with_screw(reach_pose4)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.open_gripper()

    planner.close()
    return res
