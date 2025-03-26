from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

WARNED_ONCE = False


@register_env("PickCubeYCB-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickCubeYCBEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a random object sampled from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position

    **Randomizations:**
    - the object's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the object's z-axis rotation is randomized
    - the object geometry is randomized by randomly sampling any YCB object. (during reconfiguration)

    **Success Conditions:**
    - the object position is within goal_thresh (default 0.025) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)

    **Goal Specification:**
    - 3D goal position (also visualized in human renders)

    **Additional Notes**
    - On GPU simulation, in order to collect data from every possible object in the YCB database we recommend using at least 128 parallel environments or more, otherwise you will need to reconfigure in order to sample new objects.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickSingleYCB-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, PandaWristCam, Fetch]
    cube_half_size = 0.02
    goal_thresh = 0.05

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                not in [
                    "022_windex_bottle",
                    "028_skillet_lid",
                    "029_plate",
                    "059_chain",
                ]  # NOTE (arth): ignore these non-graspable/hard to grasp ycb objects
            ]
        )
        self.all_model_ids = np.array(["029_plate", "002_master_chef_can", "004_sugar_box", "065-c_cups", "072-a_toy_airplane"])
        # self.all_model_ids = np.array(["005_tomato_soup_can"])
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def generate_spaced_points(self, n, min_dist=0.4, x_range=(-0.8, 0.8), y_range=(-0.8, 0.8)):
        points = []
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        points.append((x, y))
        attempts = 0
        max_attempts = 1000  # 防止无限循环

        while len(points) < n and attempts < max_attempts:
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            new_point = (x, y)
            valid = True
            for existing_point in points:
                if distance(new_point, existing_point) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(new_point)
                attempts = 0  # 重置尝试计数器
            else:
                attempts += 1
        if len(points) < n:
            print(f"Warning：Cannot generate {n} points，only generate {len(points)} points")

        return points

    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        # model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        model_ids = self.all_model_ids
        xy_poses = self.generate_spaced_points(len(model_ids) + 1)

        self._objs: List[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            x, y = xy_poses[i]
            builder.initial_pose = sapien.Pose(p=[x, y, 0])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        cube_x, cube_y = xy_poses[-1]
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 1, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[cube_x, cube_y, self.cube_half_size]),
        )
        # self.obj = Actor.merge(self._objs, name="ycb_object")
        # self.add_to_state_dict_registry(self.obj)

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            for i, obj in enumerate(self._objs):
                xyz = obj.pose.p
                xyz[:, 2] = self.object_zs[i]
                qs = random_quaternions(len(self.all_model_ids) + 1, lock_x=True, lock_y=True)
                obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz = self.cube.pose.p
            qs = random_quaternions(len(self.all_model_ids) + 1, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        obj_to_goal_pos = self.cube.pose.p - self._objs[0].pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self._objs[0].pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self._objs[0].pose.p - self.agent.tcp.pose.p,
                cube_pose=self.cube.pose.raw_pose,
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                cube_to_goal_pos=self._objs[0].pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_cube_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        cube_to_goal_dist = torch.linalg.norm(
            self.cube.pose.p - self._objs[0].pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        reward += place_reward * is_grasped

        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
