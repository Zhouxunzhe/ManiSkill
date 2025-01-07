"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import trimesh
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

SUITCASE_COLLISION_BIT = 29

@register_env("FoldSuitcase-v1", max_episode_steps=500)
class FoldSuitcaseEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to push and move a cube to a goal region in front of it

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

    **Success Conditions:**
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
    """

    # _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    fixed_model_id = "103762"
    # // "103762": {
    # //     "num_target_links": 2,
    # //     "partnet_mobility_id": 103762,
    # //     "scale": 0.3
    # // }
    # ["100767", "101668", "103755", "103761", "103762"]
    # TODO(zxz)
    random_id = True

    # Specify some supported robot types
    agent: Union[Panda, Fetch]
    lid_types = ["revolute_unwrapped", "prismatic"]
    TRAIN_JSON = (
            PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_suitcase.json"
    )

    # set some commonly used values
    max_close_frac = 0.25
    suitcase_half_size = 0.2

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**21, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "hand_camera",   # "hand_camera"
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.2, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load suitcase
        self._load_suitcase(self.lid_types)

    def _transform_point(self, origin_base: sapien.Pose, new_base: sapien.Pose, origin_pos: sapien.Pose):
        from scipy.spatial.transform import Rotation
        # Convert quaternions to rotation objects
        R_O = Rotation.from_quat([origin_base.q[1], origin_base.q[2], origin_base.q[3], origin_base.q[0]])  # scipy uses (x,y,z,w) format
        R_O_new = Rotation.from_quat([new_base.q[1], new_base.q[2], new_base.q[3], new_base.q[0]])

        # Calculate relative position of A with respect to O
        relative_pos = origin_pos.p - origin_base.p

        # Calculate relative rotation of A with respect to O
        R_A = Rotation.from_quat([origin_pos.q[1], origin_pos.q[2], origin_pos.q[3], origin_pos.q[0]])
        relative_rot = R_O.inv() * R_A

        # Calculate new position of A
        relative_pos_rotated = R_O_new.apply(R_O.inv().apply(relative_pos))
        p_A_new = new_base.p + relative_pos_rotated

        # Calculate new rotation of A
        R_A_new = R_O_new * relative_rot
        q_A_new = R_A_new.as_quat()  # Returns (x,y,z,w)

        # Convert back to (w,x,y,z) format
        q_A_new = np.array([q_A_new[3], q_A_new[0], q_A_new[1], q_A_new[2]])

        new_pos = sapien.Pose(
            p=p_A_new,
            q=q_A_new
        )
        return new_pos

    def _load_suitcase(self, joint_types: List[str]):
        # we sample random suitcase model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2 ** 31)

        self._suitcases = []
        self.lid_links: List[List[Link]] = []
        self.lid_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            if not self.random_id:
                model_id = self.fixed_model_id
            self.model_id = model_id
            suitcase_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            suitcase_builder.set_scene_idxs(scene_idxs=[i])
            self.origin_base = sapien.Pose(
                p=[-0.1, 0, self.suitcase_half_size],
                q=euler2quat(np.pi / 2, np.pi / 2, 0)
            )
            # original
            # self.new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.1, 0, self.suitcase_half_size],
            #     q=euler2quat(np.pi / 2, np.pi / 2, 0)
            # )
            # rotate Y
            # self.new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.1, 0, self.suitcase_half_size],
            #     q=euler2quat(np.pi / 2, np.pi * 3/8, 0)
            # )
            # rotate Z
            # self.new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.1, 0, self.suitcase_half_size],
            #     q=euler2quat(np.pi / 2, np.pi / 2, np.pi / 8)
            # )
            # random new base pose
            self.new_base = suitcase_builder.initial_pose = sapien.Pose(
                p=[-0.2 + np.random.uniform(-1, 1) * 0.02, -0.1 + np.random.uniform(-1, 1) * 0.02, self.suitcase_half_size + np.random.uniform(-1, 1) * 0.02],
                q=euler2quat(np.pi / 2, np.pi / 2 + np.random.uniform(-1, 1) * 1/16 * np.pi, 0.0 + np.random.uniform(-1, 1) * 1/16 * np.pi)
            )
            suitcase = suitcase_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(suitcase)
            # this disables self collisions by setting the group 2 bit at SUITCASE_COLLISION_BIT all the same
            # that bit is also used to disable collision with the ground plane
            for link in suitcase.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=SUITCASE_COLLISION_BIT, bit=1
                )
            self._suitcases.append(suitcase)
            self.lid_links.append([])
            self.lid_links_meshes.append([])

            # selecting semantic parts of articulations
            for link, joint in zip(suitcase.links, suitcase.joints):
                if joint.type[0] in joint_types:
                    self.lid_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a lid
                    self.lid_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "lid" in render_shape.name,
                            mesh_name="lid",
                        )[0]
                    )

        # we can merge different articulations/links with different degrees of freedoms into a single view/object
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.suitcase = Articulation.merge(self._suitcases, name="suitcase")
        self.add_to_state_dict_registry(self.suitcase)

        qpos_max = []
        for i in range(len(self.lid_links[0])):
            target_qlimits = self.lid_links[0][i].joint.limits  # [b, 1, 2]
            qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
            qpos_max.append(qmax)

        # self.lid_links[0][0].joint.set_qpos(qpos_max)

        self.lid_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(self.lid_links)],
            name="lid_link",
        )

        if self.model_id == "103762":
            self.lid_link = self.lid_links[0][0]
        # store the position of the lid mesh itself relative to the link it is apart of
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

        self.waypoint_pos1 = self._transform_point(
            self.origin_base,
            self.new_base,
            sapien.Pose(
                p=[-0.2, 0.483, 0.4],
                q=[-0.0, 1.000, 0, 0.0]
            )
        )
        self.waypoint_pos2 = self._transform_point(
            self.origin_base,
            self.new_base,
            sapien.Pose(
                p=[-0.2, 0.32, 0.03],
                q=[0.635, 0.772, 0.016, 0.015]
            )
        )
        if self.model_id == "103762":
            self.waypoint_pos2 = self._transform_point(
                self.origin_base,
                self.new_base,
                sapien.Pose(
                    p=[-0.2, 0.32, 0.12],
                    q=[0.635, 0.772, 0.016, 0.015]
                )
            )
        elif self.model_id == "100767":
            self.waypoint_pos2 = self._transform_point(
                self.origin_base,
                self.new_base,
                sapien.Pose(
                    p=[-0.2, 0.35, 0.12],
                    q=[0.635, 0.772, 0.016, 0.015]
                )
            )
        self.waypoint_pos3 = self._transform_point(
            self.origin_base,
            self.new_base,
            sapien.Pose(
                p=[-0.2, 0.23, 0.4],
                q=[0.3, 0.953, 0.012, 0.013]
            )
        )
        self.waypoint_pos4 = self._transform_point(
            self.origin_base,
            self.new_base,
            sapien.Pose(
                p=[-0.2, 0.019, 0.4],
                q=[-0.086, 0.996, 0.004, 0.026]
            )
        )
        self.waypoint_pos5 = self._transform_point(
            self.origin_base,
            self.new_base,
            sapien.Pose(
                p=[-0.2, -0.16, 0.23],
                q=[-0.503, 0.864, -0.008, 0.018]
            )
        )
        self.waypoint_pos6 = self._transform_point(
            self.origin_base,
            self.new_base,
            sapien.Pose(
                p=[-0.2, -0.331, 0.347],
                q=[-0.503, 0.864, -0.008, 0.018]
            )
        )

    def _after_reconfigure(self, options):
        # To spawn suitcases in the right place, we need to change their z position such that
        # the bottom of the suitcase sits at z=0 (the floor). Luckily the partnet mobility dataset is made such that
        # the negative of the lower z-bound of the collision mesh bounding box is the right value

        # this code is in _after_reconfigure since retrieving collision meshes requires the GPU to be initialized
        # which occurs after the initial reconfigure call (after self._load_scene() is called)
        self.suitcase_zs = []
        for suitcase in self._suitcases:
            collision_mesh = suitcase.get_first_collision_mesh()
            self.suitcase_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.suitcase_zs = common.to_tensor(self.suitcase_zs, device=self.device)

        # get the qmin qmax values of the joint corresponding to the selected links
        target_qlimits = self.lid_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.max_close_frac

    def lid_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.lid_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.lid_link_pos, device=self.device),
            )
        return transform_points(
            self.lid_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.lid_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            qpos_max = []
            for i in range(len(self.lid_links[0])):
                target_qlimits = self.lid_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos_max.append(qmax)

            self.lid_links[0][0].joint.set_qpos(qpos_max)

    def _after_control_step(self):
        # after each control step, we update the goal position of the lid link
        # for GPU sim we need to update the kinematics data to get latest pose information for up to date link poses
        # and fetch it, followed by an apply call to ensure the GPU sim is up to date
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        # even though self.lid_link is a different link across different articulations
        # we can still fetch a joint that represents the parent joint of all those links
        # and easily get the qpos value.
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
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_lid_pos=info["lid_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.lid_link.joint.qpos,
                target_lid_pos=info["lid_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_to_lid_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["lid_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_lid_dist)
        amount_to_close_left = torch.div(
            self.target_qpos - self.lid_link.joint.qpos, self.target_qpos
        )
        close_reward = 2 * (1 - amount_to_close_left)
        reaching_reward[
            amount_to_close_left < 0.999
            ] = 2  # if joint closes even a tiny bit, we don't need reach reward anymore
        # print(close_reward.shape)
        close_reward[info["close_enough"]] = 3  # give max reward here
        reward = reaching_reward + close_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
