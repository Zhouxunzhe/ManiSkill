from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
from transforms3d.euler import euler2quat
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

REALMAN_RIGHT_COLLISION_BIT = 28
REALMAN_LEFT_COLLISION_BIT = 29
REALMAN_WHEELS_COLLISION_BIT = 30
"""Collision bit of the mobile_realman robot wheel links"""
REALMAN_BASE_COLLISION_BIT = 31
"""Collision bit of the mobile_realman base"""


@register_agent()
class Realman(BaseAgent):
    disable_self_collisions = True
    uid = "mobile_realman"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/mobile_realman/dual_65B_arm_robot/urdf/dual_65B_arm_robot.urdf"
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            **{
                f"right_{k}_1": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["thumb", "index", "middle", "ring", "little"]
            },
            **{
                f"left_{k}_1": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["thumb", "index", "middle", "ring", "little"]
            },
            "right_thumb_2": dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            "left_thumb_2": dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array(
                ([0.0] * (41)) * 1
            ),
        )
    )

    head_joints = [
        "head_joint1",
        "head_joint2",
        # "camera_joint",
        # "r_base_joint1",
        # "l_base_joint1",
        # "l_body_arm_joint",
        # "r_body_arm_joint",
        # "dipan_zhuti_joint",
        # "r_wheel_joint_1",
        # "l_wheel_joint2",
        # "min_wheel_joint",
        # "medium_wheel_joint2",
        # "swivel_wheel_joint1_1",
        # "swivel_wheel_joint_1_2",
        # "swivel_wheel_joint2_1",
        # "swivel_wheel_joint2_2",
        # "swivel_wheel_joint3_1",
        # "swivel_wheel_joint3_2",
        # "swivel_wheel_joint4_1",
        # "swivel_wheel_joint4_2",
        # "l_arm_hand_joint",
        # "r_arm_hand_joint",
    ]
    right_arm_joints = [
        "r_joint1",
        "r_joint2",
        "r_joint3",
        "r_joint4",
        "r_joint5",
        "r_joint6",
    ]
    left_arm_joints = [
        "l_joint1",
        "l_joint2",
        "l_joint3",
        "l_joint4",
        "l_joint5",
        "l_joint6",
    ]
    right_finger_joints = [
        "right_thumb_1_joint",
        "right_thumb_2_joint",
        # "right_thumb_3_joint",
        # "right_thumb_4_joint",
        "right_index_1_joint",
        # "right_index_2_joint",
        "right_middle_1_joint",
        # "right_middle_2_joint",
        "right_ring_1_joint",
        # "right_ring_2_joint",
        "right_little_1_joint",
        # "right_little_2_joint",
    ]
    left_finger_joints = [
        "left_thumb_1_joint",
        "left_thumb_2_joint",
        # "left_thumb_3_joint",
        # "left_thumb_4_joint",
        "left_index_1_joint",
        # "left_index_2_joint",
        "left_middle_1_joint",
        # "left_middle_2_joint",
        "left_ring_1_joint",
        # "left_ring_2_joint",
        "left_little_1_joint",
        # "left_little_2_joint"
    ]
    right_ee_link_name = "r_link6"
    left_ee_link_name = "l_link6"
    base_joints = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
    ]

    head_stiffness = 1e3
    head_damping = 1e2
    head_force_limit = 100
    left_arm_stiffness = 1e3
    left_arm_damping = 1e2
    left_arm_force_limit = 100
    right_arm_stiffness = 1e3
    right_arm_damping = 1e2
    right_arm_force_limit = 100
    left_finger_stiffness = 1e3
    left_finger_damping = 1e2
    left_finger_force_limit = 100
    right_finger_stiffness = 1e3
    right_finger_damping = 1e2
    right_finger_force_limit = 100

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="realman_head",
                pose=sapien.Pose(p=[0.05, 0, 0.46], q=euler2quat(0, np.pi / 6, 0)),
                width=128,
                height=128,
                fov=np.pi,
                near=0.01,
                far=100,
                entity_uid="head_link2",
            ),
            CameraConfig(
                uid="realman_right_hand",
                pose=Pose.create_from_pq([-0.1, 0, 0.1], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                entity_uid="r_hand_base_link",
            ),
            CameraConfig(
                uid="realman_left_hand",
                pose=Pose.create_from_pq([-0.1, 0, 0.1], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                entity_uid="l_hand_base_link",
            ),
        ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joints,
            None,
            None,
            self.right_arm_stiffness,
            self.right_arm_damping,
            self.right_arm_force_limit,
            normalize_action=False,
        )
        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joints,
            -0.1,
            0.1,
            self.right_arm_stiffness,
            self.right_arm_damping,
            self.right_arm_force_limit,
            use_delta=True,
        )
        right_arm_pd_joint_target_delta_pos = deepcopy(right_arm_pd_joint_delta_pos)
        right_arm_pd_joint_target_delta_pos.use_target = True
        # PD joint velocity
        right_arm_pd_joint_vel = PDJointVelControllerConfig(
            self.right_arm_joints,
            -1.0,
            1.0,
            self.right_arm_damping,  # this might need to be tuned separately
            self.right_arm_force_limit,
        )
        # PD joint position and velocity
        right_arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.right_arm_joints,
            None,
            None,
            self.right_arm_stiffness,
            self.right_arm_damping,
            self.right_arm_force_limit,
            normalize_action=True,
        )
        right_arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.right_arm_joints,
            -0.1,
            0.1,
            self.right_arm_stiffness,
            self.right_arm_damping,
            self.right_arm_force_limit,
            use_delta=True,
        )

        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joints,
            None,
            None,
            self.left_arm_stiffness,
            self.left_arm_damping,
            self.left_arm_force_limit,
            normalize_action=False,
        )
        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joints,
            -0.1,
            0.1,
            self.left_arm_stiffness,
            self.left_arm_damping,
            self.left_arm_force_limit,
            use_delta=True,
        )
        left_arm_pd_joint_target_delta_pos = deepcopy(left_arm_pd_joint_delta_pos)
        left_arm_pd_joint_target_delta_pos.use_target = True
        # PD joint velocity
        left_arm_pd_joint_vel = PDJointVelControllerConfig(
            self.left_arm_joints,
            -1.0,
            1.0,
            self.left_arm_damping,  # this might need to be tuned separately
            self.left_arm_force_limit,
        )
        # PD joint position and velocity
        left_arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.left_arm_joints,
            None,
            None,
            self.left_arm_stiffness,
            self.left_arm_damping,
            self.left_arm_force_limit,
            normalize_action=True,
        )
        left_arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.left_arm_joints,
            -0.1,
            0.1,
            self.left_arm_stiffness,
            self.left_arm_damping,
            self.left_arm_force_limit,
            use_delta=True,
        )

        # PD ee position
        right_arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.right_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.right_arm_stiffness,
            damping=self.right_arm_damping,
            force_limit=self.right_arm_force_limit,
            ee_link=self.right_ee_link_name,
            urdf_path=self.urdf_path,
        )
        right_arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.right_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.right_arm_stiffness,
            damping=self.right_arm_damping,
            force_limit=self.right_arm_force_limit,
            ee_link=self.right_ee_link_name,
            urdf_path=self.urdf_path,
        )

        right_arm_pd_ee_target_delta_pos = deepcopy(right_arm_pd_ee_delta_pos)
        right_arm_pd_ee_target_delta_pos.use_target = True
        right_arm_pd_ee_target_delta_pose = deepcopy(right_arm_pd_ee_delta_pose)
        right_arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        right_arm_pd_ee_delta_pose_align = deepcopy(right_arm_pd_ee_delta_pose)
        right_arm_pd_ee_delta_pose_align.frame = "ee_align"

        left_arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.left_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.left_arm_stiffness,
            damping=self.left_arm_damping,
            force_limit=self.left_arm_force_limit,
            ee_link=self.left_ee_link_name,
            urdf_path=self.urdf_path,
        )
        left_arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.left_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.left_arm_stiffness,
            damping=self.left_arm_damping,
            force_limit=self.left_arm_force_limit,
            ee_link=self.left_ee_link_name,
            urdf_path=self.urdf_path,
        )

        left_arm_pd_ee_target_delta_pos = deepcopy(left_arm_pd_ee_delta_pos)
        left_arm_pd_ee_target_delta_pos.use_target = True
        left_arm_pd_ee_target_delta_pose = deepcopy(left_arm_pd_ee_delta_pose)
        left_arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        left_arm_pd_ee_delta_pose_align = deepcopy(left_arm_pd_ee_delta_pose)
        left_arm_pd_ee_delta_pose_align.frame = "ee_align"

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        right_finger_pd_joint_pos = PDJointPosControllerConfig(
            self.right_finger_joints,
            -10.0,  # a trick to have force when the object is thin
            10.0,
            self.right_finger_stiffness,
            self.right_finger_damping,
            self.right_finger_force_limit,
        )
        left_finger_pd_joint_pos = PDJointPosControllerConfig(
            self.left_finger_joints,
            -10.0,  # a trick to have force when the object is thin
            10.0,
            self.left_finger_stiffness,
            self.left_finger_damping,
            self.left_finger_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.head_joints,
            None,
            None,
            self.head_stiffness,
            self.head_damping,
            self.head_force_limit,
            use_delta=True,
        )

        # useful to keep body unmoving from passed position
        stiff_body_pd_joint_pos = PDJointPosControllerConfig(
            self.head_joints,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joints,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                right_arm=right_arm_pd_joint_delta_pos,
                right_gripper=right_finger_pd_joint_pos,
                left_arm=left_arm_pd_joint_delta_pos,
                left_gripper=left_finger_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos=dict(
                right_arm=right_arm_pd_joint_pos,
                right_gripper=right_finger_pd_joint_pos,
                left_arm=left_arm_pd_joint_pos,
                left_gripper=left_finger_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pos=dict(
                right_arm=right_arm_pd_ee_delta_pos,
                right_gripper=right_finger_pd_joint_pos,
                left_arm=left_arm_pd_ee_delta_pos,
                left_gripper=left_finger_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # pd_ee_delta_pose=dict(
            #     right_arm=right_arm_pd_ee_delta_pose,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_ee_delta_pose,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_ee_delta_pose_align=dict(
            #     right_arm=right_arm_pd_ee_delta_pose_align,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_ee_delta_pose_align,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_joint_target_delta_pos=dict(
            #     right_arm=right_arm_pd_joint_target_delta_pos,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_joint_target_delta_pos,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_ee_target_delta_pos=dict(
            #     right_arm=right_arm_pd_ee_target_delta_pos,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_ee_target_delta_pos,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_ee_target_delta_pose=dict(
            #     right_arm=right_arm_pd_ee_target_delta_pose,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_ee_target_delta_pose,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # # Caution to use the following controllers
            # pd_joint_vel=dict(
            #     right_arm=right_arm_pd_joint_vel,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_joint_vel,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_joint_pos_vel=dict(
            #     right_arm=right_arm_pd_joint_pos_vel,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_joint_pos_vel,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_joint_delta_pos_vel=dict(
            #     right_arm=right_arm_pd_joint_delta_pos_vel,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_joint_delta_pos_vel,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            #     base=base_pd_joint_vel,
            # ),
            # pd_joint_delta_pos_stiff_body=dict(
            #     right_arm=right_arm_pd_joint_delta_pos,
            #     right_gripper=right_finger_pd_joint_pos,
            #     left_arm=left_arm_pd_joint_delta_pos,
            #     left_gripper=left_finger_pd_joint_pos,
            #     body=stiff_body_pd_joint_pos,
            #     base=base_pd_joint_vel,
            # ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.right_thumb_contact_link = self.robot.links_map["right_thumb_4"]
        self.right_index_contact_link = self.robot.links_map["right_index_2"]
        self.right_middle_contact_link = self.robot.links_map["right_middle_2"]
        self.right_ring_contact_link = self.robot.links_map["right_ring_2"]
        self.right_little_contact_link = self.robot.links_map["right_little_2"]
        self.right_tcp = self.robot.links_map["r_link6"]
        self.right_finger_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.right_finger_joints
        ]

        self.left_thumb_contact_link = self.robot.links_map["left_thumb_4"]
        self.left_index_contact_link = self.robot.links_map["left_index_2"]
        self.left_middle_contact_link = self.robot.links_map["left_middle_2"]
        self.left_ring_contact_link = self.robot.links_map["left_ring_2"]
        self.left_little_contact_link = self.robot.links_map["left_little_2"]
        self.left_tcp = self.robot.links_map["l_link6"]
        self.left_finger_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.left_finger_joints
        ]

        # disable collisions between fingers. Done in python here instead of the srdf as we can use less collision bits this way and do it more smartly
        # note that the two link of the fingers can collide with other finger links and the palm link so its not included
        link_names = ["thumb", "index", "middle", "ring", "little"]
        for ln in link_names:
            self.robot.links_map[f"right_{ln}_1"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
            self.robot.links_map[f"right_{ln}_2"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
            self.robot.links_map[f"left_{ln}_1"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
            self.robot.links_map[f"left_{ln}_2"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
        self.robot.links_map["right_thumb_3"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
        self.robot.links_map["right_thumb_4"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
        self.robot.links_map["left_thumb_3"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
        self.robot.links_map["left_thumb_4"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)

        for i in range(1, 7):
            self.robot.links_map[f"r_link{i}"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
            self.robot.links_map[f"l_link{i}"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)

        self.robot.links_map["r_hand_base_link"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
        self.robot.links_map["l_hand_base_link"].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)

        body_links = [
            "body_base_link",
            "base_link",
            "l_base_link",
            "r_base_link",
            "r_base_link1",
            "l_base_link1",
            "root",
            "root_arm_1_link_1",
            "root_arm_1_link_2",
        ]
        wheel_links = [
            "head_link1",
            "head_link2",
            "camera_link",
            "r_wheel_link",
            "l_wheel_link",
            "min_wheel_link1",
            "medium_wheel_link2",
            "swivel_wheel_link1_1",
            "swivel_wheel_link1_2",
            "swivel_wheel_link2_1",
            "swivel_wheel_link2_2",
            "swivel_wheel_link3_1",
            "swivel_wheel_link3_2",
            "swivel_wheel_link4_1",
            "swivel_wheel_link4_2",
        ]

        for body_link in body_links:
            # self.robot.links_map[body_link].set_collision_group_bit(2, 1, 1)
            # self.robot.links_map[body_link].set_collision_group_bit(2, 2, 1)
            self.robot.links_map[body_link].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)
        for wheel_link in wheel_links:
            # self.robot.links_map[wheel_link].set_collision_group_bit(2, 1, 1)
            # self.robot.links_map[wheel_link].set_collision_group_bit(2, 2, 1)
            self.robot.links_map[wheel_link].set_collision_group_bit(2, REALMAN_BASE_COLLISION_BIT, 1)

    def right_hand_dist_to_open_grasp(self):
        """compute the distance from the current qpos to a open grasp qpos for the right hand"""
        return torch.mean(
            torch.abs(self.robot.qpos[:, self.right_finger_joint_indexes]), dim=1
        )

    def left_hand_dist_to_open_grasp(self):
        """compute the distance from the current qpos to a open grasp qpos for the left hand"""
        return torch.mean(
            torch.abs(self.robot.qpos[:, self.left_finger_joint_indexes]), dim=1
        )

    def right_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object with just its right hand

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.right_thumb_contact_link, object
        )
        r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
            self.right_index_contact_link, object
        )
        r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
            self.right_middle_contact_link, object
        )
        r_contact_forces_3 = self.scene.get_pairwise_contact_forces(
            self.right_ring_contact_link, object
        )
        r_contact_forces_4 = self.scene.get_pairwise_contact_forces(
            self.right_little_contact_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
        rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)
        rforce_3 = torch.linalg.norm(r_contact_forces_3, axis=1)
        rforce_4 = torch.linalg.norm(r_contact_forces_4, axis=1)

        # direction to open the gripper
        ldirection = self.right_thumb_contact_link.pose.to_transformation_matrix()[
                     ..., :3, 1
                     ]
        rdirection1 = -self.right_index_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]
        rdirection2 = -self.right_middle_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]
        rdirection3 = -self.right_ring_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]
        rdirection4 = -self.right_little_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]

        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle1 = common.compute_angle_between(rdirection1, r_contact_forces_1)
        rangle2 = common.compute_angle_between(rdirection2, r_contact_forces_2)
        rangle3 = common.compute_angle_between(rdirection3, r_contact_forces_3)
        rangle4 = common.compute_angle_between(rdirection4, r_contact_forces_4)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag1 = torch.logical_and(
            rforce_1 >= min_force, torch.rad2deg(rangle1) <= max_angle
        )
        rflag2 = torch.logical_and(
            rforce_2 >= min_force, torch.rad2deg(rangle2) <= max_angle
        )
        rflag3 = torch.logical_and(
            rforce_3 >= min_force, torch.rad2deg(rangle3) <= max_angle
        )
        rflag4 = torch.logical_and(
            rforce_4 >= min_force, torch.rad2deg(rangle4) <= max_angle
        )
        rflag = rflag1 | rflag2 | rflag3 | rflag4
        return torch.logical_and(lflag, rflag)

    def left_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object with just its left hand

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_thumb_contact_link, object
        )
        r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
            self.left_index_contact_link, object
        )
        r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
            self.left_middle_contact_link, object
        )
        r_contact_forces_3 = self.scene.get_pairwise_contact_forces(
            self.left_ring_contact_link, object
        )
        r_contact_forces_4 = self.scene.get_pairwise_contact_forces(
            self.left_little_contact_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
        rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)
        rforce_3 = torch.linalg.norm(r_contact_forces_3, axis=1)
        rforce_4 = torch.linalg.norm(r_contact_forces_4, axis=1)

        # direction to open the gripper
        ldirection = self.left_thumb_contact_link.pose.to_transformation_matrix()[
                     ..., :3, 1
                     ]
        rdirection1 = -self.left_index_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]
        rdirection2 = -self.left_middle_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]
        rdirection3 = -self.left_ring_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]
        rdirection4 = -self.left_little_contact_link.pose.to_transformation_matrix()[
                       ..., :3, 1
                       ]

        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle1 = common.compute_angle_between(rdirection1, r_contact_forces_1)
        rangle2 = common.compute_angle_between(rdirection2, r_contact_forces_2)
        rangle3 = common.compute_angle_between(rdirection3, r_contact_forces_3)
        rangle4 = common.compute_angle_between(rdirection4, r_contact_forces_4)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag1 = torch.logical_and(
            rforce_1 >= min_force, torch.rad2deg(rangle1) <= max_angle
        )
        rflag2 = torch.logical_and(
            rforce_2 >= min_force, torch.rad2deg(rangle2) <= max_angle
        )
        rflag3 = torch.logical_and(
            rforce_3 >= min_force, torch.rad2deg(rangle3) <= max_angle
        )
        rflag4 = torch.logical_and(
            rforce_4 >= min_force, torch.rad2deg(rangle4) <= max_angle
        )
        rflag = rflag1 | rflag2 | rflag3 | rflag4
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        base_qvel = self.robot.get_qvel()[..., :3]
        return torch.all(body_qvel <= threshold, dim=1) & torch.all(
            base_qvel <= base_threshold, dim=1
        )

    # @staticmethod
    # def build_grasp_pose(approaching, closing, center):
    #     """Build a grasp pose (panda_hand_tcp)."""
    #     assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
    #     assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
    #     assert np.abs(approaching @ closing) <= 1e-3
    #     ortho = np.cross(closing, approaching)
    #     T = np.eye(4)
    #     T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    #     T[:3, 3] = center
    #     return sapien.Pose(T)
