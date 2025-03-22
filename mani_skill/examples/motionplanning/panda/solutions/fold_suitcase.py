import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import FoldSuitcaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: FoldSuitcaseEnv, seed=None, debug=False, vis=False):
    options = {
        "reconfigure": True
    }
    env.reset(options=options)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
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
    planner.move_to_pose_with_screw(env.waypoint_pos1)
    planner.move_to_pose_with_screw(env.waypoint_pos2)
    planner.close_gripper()

    planner.move_to_pose_with_screw(env.waypoint_pos3)
    planner.move_to_pose_with_screw(env.waypoint_pos4)
    planner.move_to_pose_with_screw(env.waypoint_pos5)
    planner.open_gripper()

    res = planner.move_to_pose_with_screw(env.waypoint_pos6)

    planner.close()
    return res
