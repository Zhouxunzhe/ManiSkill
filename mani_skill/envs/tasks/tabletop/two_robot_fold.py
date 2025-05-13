from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import sapien
import trimesh
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.fold_suitcase import FoldSuitcaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.envs.utils.observations import (
    sensor_data_to_pointcloud,
)
from mani_skill.sensors.camera import (
    Camera,
)
from examples.baselines.diffusion_policy import (
    PlainConv,
    CLIPEncoder,
    DINOv2Encoder,
    ResNetEncoder,
)

SUITCASE_COLLISION_BIT = 29

@register_env("TwoRobotFold-v1", max_episode_steps=50)
class TwoRobotFoldEnv(FoldSuitcaseEnv, BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        model=None,
        **kwargs
    ):
        self.cube_half_size = 0.02
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load suitcase
        self.all_model_ids = np.array(["103755"])
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        # print("model_id:", model_ids)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0.05, 0.4, self.cube_half_size]),
        )
        for i, model_id in enumerate(model_ids):
            if model_id in self.suitcase_list:
                self.origin_base = sapien.Pose(
                    p=[-0.1, -0.1, self.suitcase_half_size],
                    q=euler2quat(np.pi, np.pi / 2, 0)
                )
                self._load_suitcase(self.origin_base, self.lid_types, [model_id], [link_ids[i]])
            elif model_id in self.box_list:
                self.origin_base = sapien.Pose(
                    p=[-0.3,
                       0,
                       self.suitcase_half_size],
                    q=euler2quat(0,
                                 0,
                                 np.pi / 2)
                )
                self._load_box(self.origin_base, self.lid_types, [model_id], [link_ids[i]])
            elif model_id in self.high_box_list:
                self.origin_base = sapien.Pose(
                    p=[-0.3,
                       -0.,
                       self.suitcase_half_size / 2],
                    q=euler2quat(0,
                                 0,
                                 np.pi / 2)
                )
                self._load_high_box(self.origin_base, self.lid_types, [model_id], [link_ids[i]])
            elif model_id in self.laptop_list:
                self.origin_base = sapien.Pose(
                    p=[-0.3,
                       0,
                       0 + self.suitcase_half_size],
                    q=euler2quat(
                        0,
                        0,
                        np.pi / 2)
                )
                self._load_laptop(self.origin_base, self.lid_types, [model_id], [link_ids[i]])

    def _load_suitcase(self, origin_base, joint_types: List[str], model_ids, link_ids):
        self._suitcases = []
        self.lid_links: List[List[Link]] = []
        self.lid_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            suitcase_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode=self.mode
            )
            suitcase_builder.set_scene_idxs(scene_idxs=[i])
            # random new base pose
            # TODO(zxz): set new_base here
            # new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.2 + np.random.uniform(-1, 1) * 0.02, -0.1 + np.random.uniform(-1, 1) * 0.02, self.suitcase_half_size + np.random.uniform(-1, 1) * 0.02],
            #     q=euler2quat(np.pi / 2, np.pi / 2 + np.random.uniform(-1, 1) * 1/16 * np.pi, 0.0 + np.random.uniform(-1, 1) * 1/16 * np.pi)
            # )
            new_base = suitcase_builder.initial_pose = origin_base
            suitcase = suitcase_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(suitcase)
            for link in suitcase.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=SUITCASE_COLLISION_BIT, bit=1
                )
            self._suitcases.append(suitcase)
            self.lid_links.append([])
            self.lid_links_meshes.append([])

            for link, joint in zip(suitcase.links, suitcase.joints):
                if joint.type[0] in joint_types:
                    self.lid_links[-1].append(link)
                    self.lid_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "lid" in render_shape.name,
                            mesh_name="lid",
                        )[0]
                    )

            self.suitcase = Articulation.merge(self._suitcases, name="suitcase")
            self.add_to_state_dict_registry(self.suitcase)

            qpos_max = []
            for i in range(len(self.lid_links[0])):
                target_qlimits = self.lid_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos_max.append(qmax)

            self.lid_link = Link.merge(
                [links[link_ids[i] % len(links)] for i, links in enumerate(self.lid_links)],
                name="lid_link",
            )

            self.lid_link_pos = common.to_tensor(
                np.array(
                    [
                        meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                        if meshes[link_ids[i] % len(meshes)] is not None else np.array([0., 0., 0.])
                        for i, meshes in enumerate(self.lid_links_meshes)
                    ]
                ),
                device=self.device,
            )

            self.right_waypoint1 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.032, 0.402, 0.140],
                    q=[-0.005, 0.71, -0.704, 0.016]
                )
            )
            self.right_waypoint2 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.032, 0.402, -0.0],
                    q=[-0.005, 0.71, -0.704, 0.016]
                )
            )
            # close gripper
            self.right_waypoint3 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.181, 0.042, 0.306],
                    q=[-0.006, 0.711, -0.703, 0.016]
                )
            )
            # open gripper
            self.right_waypoint4 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.279, 0.235, 0.40],
                    q=[-0.006, 0.711, -0.703, 0.016]
                )
            )

            self.left_waypoint1 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.370, -0.217, 0.634],
                    q=[-0.047, 0.858, 0.509, -0.043]
                )
            )
            self.left_waypoint2 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.361, -0.223, 0.231],
                    q=[-0.047, 0.858, 0.509, -0.043]
                )
            )
            self.left_waypoint3 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.4, -0.216, 0.218],
                    q=[-0.231, 0.784, 0.462, -0.344]
                )
            )
            self.left_waypoint4 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.3, -0.218, 0.11],
                    q=[-0.312, 0.714, 0.409, -0.474]
                )
            )
            # close gripper
            self.left_waypoint5 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.34, -0.228, 0.308],
                    q=[-0.331, 0.729, 0.409, -0.444]
                )
            )
            self.left_waypoint6 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[0.191, -0.236, 0.406],
                    q=[-0.226, 0.846, 0.448, -0.180]
                )
            )
            self.left_waypoint7 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.109, -0.222, 0.358],
                    q=[-0.009, 0.840, 0.523, 0.142]
                )
            )
            self.left_waypoint8 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.308, -0.230, 0.266],
                    q=[0.193, 0.684, 0.561, 0.424]
                )
            )
            # open gripper
            self.left_waypoint9 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.409, -0.231, 0.302],
                    q=[0.217, 0.690, 0.561, 0.410]
                )
            )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0, 0.75], [0.0, 0.0, 0.25])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super(FoldSuitcaseEnv, self)._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        close_enough = self.lid_link.joint.qpos <= self.target_qpos
        lid_link_pos = self.lid_link_positions()

        link_is_static = (
                                 torch.linalg.norm(self.lid_link.angular_velocity, axis=1) <= 1
                         ) & (torch.linalg.norm(self.lid_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": close_enough & link_is_static,
            "lid_link_pos": lid_link_pos,
            "close_enough": close_enough,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            # left_tactile=self.left_agent.tactile(self.lid_links[0][0])
        )

        if "state" in self.obs_mode:
            obs.update(
                left_tcp_to_lid_pos=info["lid_link_pos"] - self.left_agent.tcp.pose.p,
                right_tcp_to_lid_pos=info["lid_link_pos"] - self.right_agent.tcp.pose.p,
                # TODO(zxz): modify for PPO
                target_link_qpos=self.lid_link.joint.qpos,
                target_lid_pos=info["lid_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_lid_dist = torch.linalg.norm(
            self.left_agent.tcp.pose.p - info["lid_link_pos"], axis=1
        )
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_lid_dist)
        amount_to_close_left = torch.div(
            self.target_qpos - self.lid_link.joint.qpos, self.target_qpos
        )
        close_reward = 2 * (1 - amount_to_close_left)
        reaching_reward = amount_to_close_left
        # print(close_reward.shape)
        # close_reward[info["close_enough"]] = 3  # give max reward here
        reward = reaching_reward + close_reward
        # reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
