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
import time
import numpy as np
import sapien
import trimesh
import torch
import torch.random
from PIL.ImageChops import overlay
from numpy.distutils.misc_util import njoin
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

@register_env("FoldSuitcase-v1", max_episode_steps=500)
class FoldSuitcaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    fixed_model_id = "9748"

    # suitcase
    suitcase_list = ["100767", "101668", "103755", "103761", "103762"]  # Eval: "103762"
    # box
    box_list = ["100189", "102379",]
    high_box_list = ["100141", "47645", "48492", "100174", "100214", "100221", "100243", "100664", "102456"]  # Eval: "100214"
    # laptop
    laptop_list = [
        "9748", "9912", "9960", "9968", "9992", "9996", "10040",
        "10098", "10101", "10125", "10211", "10213", "10238", "10239", "10243",
        "10248", "10269", "10270", "10280", "10289", "10305", "10306", "10383",
        "10626", "10697", "10707", "10885", "10915", "11030", "11075", "11141", "11156",
        "11242", "11248", "11395", "11405", "11406", "11429", "11477", "11581",
        "11586", "11691", "11778", "11854", "11876", "11888", "11945", "12073", "12115"
    ]
    laptop_90 = [
        "9912", "10125", "10213", "10243", "10248", "10270", "10280", "10306", "10885", "11030", "11156",
        "11248", "11395", "11406", "11429", "11586", "11778", "11854", "11876", "11888", "11945"
    ]  # Eval: "10280", "11778",
    laptop_135 = [
        "9960", "10098", "10211", "10239", "10269", "10289", "10383", "10626", "10697", "10915", "11075",
        "11405", "11477", "11538", "11581", "11691", "12073"]  # Eval: "10289", "11581",
    laptop_180 = [
        "9748", "9968", "9992", "9996", "10040", "10101", "10238", "10305", "10356", "10707",
        "11141", "11242"]  # Eval: "10101"
    observations = []

    # Specify some supported robot types
    agent: Union[Panda, Fetch]
    lid_types = ["revolute_unwrapped", "prismatic"]

    # set some commonly used values
    max_close_frac = 0.25
    suitcase_half_size = 0.2
    random_thresh = 0.02

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, mode="train", model=None, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.mode = mode
        if self.mode == "train":
            self.JSON = (
                    PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_fold_train.json"
            )
        elif self.mode == "eval":
            self.JSON = (
                    PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_fold_eval.json"
            )
        elif self.mode == "box":
            self.JSON = (
                    PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_fold_box_train.json"
            )
        elif self.mode == "laptop":
            self.JSON = (
                    PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_fold_laptop_train.json"
            )
        elif self.mode == "suitcase":
            self.JSON = (
                    PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_fold_suitcase_train.json"
            )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.JSON)
        self.model = model
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
                "hand_camera",
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
        # pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.2, 0.35])
        pose = sapien_utils.look_at([0.6, -0.5, 0.6], [0.0, -0.1, 0.35])
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
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        # print("model_id:", model_ids)
        for i, model_id in enumerate(model_ids):
            if model_id in self.suitcase_list:
                self.origin_base = sapien.Pose(
                    p=[-0.1, 0, self.suitcase_half_size],
                    q=euler2quat(np.pi / 2, np.pi / 2, 0)
                )
                self._load_suitcase(self.origin_base, self.lid_types, [model_id], [link_ids[i]])
            elif model_id in self.box_list:
                self.origin_base = sapien.Pose(
                    p=[-0.1,
                       0,
                       self.suitcase_half_size],
                    q=euler2quat(0,
                                 0,
                                 np.pi / 2)
                )
                self._load_box(self.origin_base, self.lid_types, [model_id], [link_ids[i]])
            elif model_id in self.high_box_list:
                self.origin_base = sapien.Pose(
                    p=[-0.1,
                       -0.,
                       self.suitcase_half_size / 2],
                    q=euler2quat(0,
                                 0,
                                 np.pi / 2)
                )
                self._load_high_box(self.origin_base, self.lid_types, [model_id], [link_ids[i]])
            elif model_id in self.laptop_list:
                self.origin_base = sapien.Pose(
                    p=[-0.1,
                       0,
                       0 + self.suitcase_half_size],
                    q=euler2quat(
                        0,
                        0,
                        np.pi / 2)
                )
                self._load_laptop(self.origin_base, self.lid_types, [model_id], [link_ids[i]])

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

    def _load_suitcase(self, origin_base, joint_types: List[str], model_ids, link_ids):
        # we sample random suitcase model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.

        self._suitcases = []
        self.lid_links: List[List[Link]] = []
        self.lid_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            suitcase_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode=self.mode
            )
            suitcase_builder.set_scene_idxs(scene_idxs=[i])
            # original
            # new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.1, 0, self.suitcase_half_size],
            #     q=euler2quat(np.pi / 2, np.pi / 2, 0)
            # )
            # rotate Y
            # new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.1, 0, self.suitcase_half_size],
            #     q=euler2quat(np.pi / 2, np.pi * 3/8, 0)
            # )
            # rotate Z
            # new_base = suitcase_builder.initial_pose = sapien.Pose(
            #     p=[-0.1, 0, self.suitcase_half_size],
            #     q=euler2quat(np.pi / 2, np.pi / 2, np.pi / 8)
            # )
            # random new base pose
            # TODO(zxz): set new_base here
            new_base = suitcase_builder.initial_pose = sapien.Pose(
                p=[-0.2 + np.random.uniform(-1, 1) * self.random_thresh,
                   -0.1 + np.random.uniform(-1, 1) * self.random_thresh,
                   self.suitcase_half_size + np.random.uniform(-1, 1) * self.random_thresh],
                q=euler2quat(np.pi / 2 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             np.pi / 2 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             0.0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi)
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

            if model_id == "103762":
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
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.483, 0.4],
                    q=[-0.0, 1.000, 0, 0.0]
                )
            )
            self.waypoint_pos2 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.32, 0.03],
                    q=[0.635, 0.772, 0.016, 0.015]
                )
            )
            if model_id == "103762":
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.32, 0.12],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            elif model_id == "100767":
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.35, 0.12],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            self.waypoint_pos3 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.23, 0.4],
                    q=[0.3, 0.953, 0.012, 0.013]
                )
            )
            self.waypoint_pos4 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.019, 0.4],
                    q=[-0.086, 0.996, 0.004, 0.026]
                )
            )
            self.waypoint_pos5 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.16, 0.23],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )
            self.waypoint_pos6 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.331, 0.347],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )

    def _load_laptop(self, origin_base, joint_types: List[str], model_ids, link_ids):
        # we sample random suitcase model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.

        self._suitcases = []
        self.lid_links: List[List[Link]] = []
        self.lid_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            suitcase_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode=self.mode
            )
            suitcase_builder.set_scene_idxs(scene_idxs=[i])
            # random new base pose
            new_base = suitcase_builder.initial_pose = sapien.Pose(
                p=[-0.1 + np.random.uniform(-1, 1) * self.random_thresh,
                   -0.0 + np.random.uniform(-1, 1) * self.random_thresh,
                   -0. + self.suitcase_half_size + np.random.uniform(-1, 1) * self.random_thresh],
                q=euler2quat(0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             np.pi / 2 + np.random.uniform(-1, 1) * self.random_thresh * np.pi)
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

            if model_id == "103762":
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

            # waypoint 1
            if model_id in self.laptop_180:
                self.waypoint_pos1 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.483, 0.37],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            if model_id in self.laptop_90:
                self.waypoint_pos1 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.13, 0.5],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            if model_id in ["11691"]:
                self.waypoint_pos1 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.3, 0.5],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            elif model_id in self.laptop_135:
                self.waypoint_pos1 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.45, 0.5],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )

            # waypoint 2
            if model_id in self.laptop_180:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.22, 0.15],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            if model_id in ["10885", "11945"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.15, 0.35],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            elif model_id in ["11030", "11248"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.08, 0.35],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            elif model_id in ["11888"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.04, 0.35],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            elif model_id in self.laptop_90:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.13, 0.35],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            if model_id in ["12073"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.23, 0.36],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )
            elif model_id in ["9960", "10269", "10289", "11477", "11581"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.23, 0.27],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )
            elif model_id in ["10098", "10915", "11242", "11405"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.23, 0.23],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )
            elif model_id in ["10383", "10626", "11075"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.23, 0.2],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )
            elif model_id in ["10697"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.23, 0.17],
                        q=[0.6, 0.753, 0.012, 0.013]
                    )
                )
            elif model_id in self.laptop_135:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.15, 0.3],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )

            # waypoint 3
            self.waypoint_pos3 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.23, 0.3],
                    q=[0.3, 0.953, 0.012, 0.013]
                )
            )
            if model_id in self.laptop_90 or self.laptop_135:
                self.waypoint_pos3 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.1, 0.35],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )

            # waypoint 4
            self.waypoint_pos4 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.019, 0.3],
                    q=[-0.086, 0.996, 0.004, 0.026]
                )
            )

            # waypoint 5
            self.waypoint_pos5 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.16, 0.15],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )

            # # waypoint 6
            self.waypoint_pos6 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.331, 0.247],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )

    def _load_box(self, origin_base, joint_types: List[str], model_ids, link_ids):
        # we sample random suitcase model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.

        self._suitcases = []
        self.lid_links: List[List[Link]] = []
        self.lid_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            suitcase_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode=self.mode
            )
            suitcase_builder.set_scene_idxs(scene_idxs=[i])
            # random new base pose
            new_base = sapien.Pose(
                p=[-0.1 + np.random.uniform(-1, 1) * self.random_thresh,
                   0 + np.random.uniform(-1, 1) * self.random_thresh,
                   self.suitcase_half_size + np.random.uniform(-1, 1) * self.random_thresh],
                q=euler2quat(0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             np.pi / 2 + np.random.uniform(-1, 1) * self.random_thresh * np.pi)
            )
            suitcase_builder.initial_pose = new_base
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
                qpos_max.append(qmax/4)

            # self.lid_links[0][0].joint.set_qpos(qpos_max)

            self.lid_link = Link.merge(
                [links[link_ids[i] % len(links)] for i, links in enumerate(self.lid_links)],
                name="lid_link",
            )

            if model_id == "103762":
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
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.483, 0.4],
                    q=[-0.0, 1.000, 0, 0.0]
                )
            )
            self.waypoint_pos2 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.32, 0.03],
                    q=[0.635, 0.772, 0.016, 0.015]
                )
            )
            if model_id == "100189":
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.32, 0.17],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            elif model_id == "102379":
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.25, 0.15],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            self.waypoint_pos3 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.23, 0.34],
                    q=[0.3, 0.953, 0.012, 0.013]
                )
            )
            self.waypoint_pos4 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.019, 0.34],
                    q=[-0.086, 0.996, 0.004, 0.026]
                )
            )
            self.waypoint_pos5 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.16, 0.23],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )
            self.waypoint_pos6 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.331, 0.347],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )

    def _load_high_box(self, origin_base, joint_types: List[str], model_ids, link_ids):
        # we sample random suitcase model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.

        self._suitcases = []
        self.lid_links: List[List[Link]] = []
        self.lid_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            suitcase_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode=self.mode
            )
            suitcase_builder.set_scene_idxs(scene_idxs=[i])
            # random new base pose
            new_base = sapien.Pose(
                p=[-0.1 + np.random.uniform(-1, 1) * self.random_thresh,
                   -0. + np.random.uniform(-1, 1) * self.random_thresh,
                   self.suitcase_half_size/2 + np.random.uniform(-1, 1) * self.random_thresh],
                q=euler2quat(0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             0 + np.random.uniform(-1, 1) * self.random_thresh * np.pi,
                             np.pi / 2 + np.random.uniform(-1, 1) * self.random_thresh * np.pi)
            )
            suitcase_builder.initial_pose = new_base
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
                qpos_max.append(qmax/2)

            # self.lid_links[0][0].joint.set_qpos(qpos_max)

            self.lid_link = Link.merge(
                [links[link_ids[i] % len(links)] for i, links in enumerate(self.lid_links)],
                name="lid_link",
            )

            if model_id == "103762":
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
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.483, 0.4],
                    q=[-0.0, 1.000, 0, 0.0]
                )
            )
            if model_id in ["100214", "100243"]:
                self.waypoint_pos1 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.183, 0.4],
                        q=[-0.0, 1.000, 0, 0.0]
                    )
                )
            self.waypoint_pos2 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.22, 0.2],
                    q=[0.635, 0.772, 0.016, 0.015]
                )
            )
            if model_id in ["48492", "100174"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.34, 0.16],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            elif model_id in ["100221"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.28, 0.07],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            elif model_id in ["100214", "100243"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.14, 0.26],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )
            elif model_id in ["100664"]:
                self.waypoint_pos2 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.18, 0.18],
                        q=[0.635, 0.772, 0.016, 0.015]
                    )
                )
            self.waypoint_pos3 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.23, 0.3],
                    q=[0.3, 0.953, 0.012, 0.013]
                )
            )
            if model_id in ["100214", "100243", "100664"]:
                self.waypoint_pos3 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.14, 0.26],
                        q=[0.3, 0.953, 0.012, 0.013]
                    )
                )
            self.waypoint_pos4 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, 0.019, 0.3],
                    q=[-0.086, 0.996, 0.004, 0.026]
                )
            )
            if model_id in ["100214", "100243"]:
                self.waypoint_pos4 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, 0.019, 0.26],
                        q=[-0.086, 0.996, 0.004, 0.026]
                    )
                )
            self.waypoint_pos5 = self._transform_point(
                origin_base,
                new_base,
                sapien.Pose(
                    p=[-0.2, -0.16, 0.23],
                    q=[-0.503, 0.864, -0.008, 0.018]
                )
            )
            if model_id in ["47645", "100214", "100243", "100664", "102456"]:
                self.waypoint_pos5 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, -0.1, 0.2],
                        q=[-0.503, 0.864, -0.008, 0.018]
                    )
                )
            elif model_id in ["100174"]:
                self.waypoint_pos5 = self._transform_point(
                    origin_base,
                    new_base,
                    sapien.Pose(
                        p=[-0.2, -0.05, 0.2],
                        q=[-0.503, 0.864, -0.008, 0.018]
                    )
                )
            self.waypoint_pos6 = self._transform_point(
                origin_base,
                new_base,
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
            # tactile=self.agent.tactile(self.lid_links[0][0])
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_lid_pos=info["lid_link_pos"] - self.agent.tcp.pose.p,
                # TODO(zxz): modify for PPO
                target_link_qpos=self.lid_link.joint.qpos,
                target_lid_pos=info["lid_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_to_lid_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["lid_link_pos"], axis=1
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

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    """
    Sensor Data
    """
    def get_obs(self, info: Optional[Dict] = None):
        """
        Return the current observation of the environment. User may call this directly to get the current observation
        as opposed to taking a step with actions in the environment.

        Note that some tasks use info of the current environment state to populate the observations to avoid having to
        compute slow operations twice. For example a state based observation may wish to include a boolean indicating
        if a robot is grasping an object. Computing this boolean correctly is slow, so it is preferable to generate that
        data in the info object by overriding the `self.evaluate` function.

        Args:
            info (Dict): The info object of the environment. Generally should always be the result of `self.get_info()`.
                If this is None (the default), this function will call `self.get_info()` itself
        """
        if info is None:
            info = self.get_info()
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return dict()
        elif self._obs_mode == "state":
            state_dict = self._get_obs_state_dict(info)
            obs = common.flatten_state_dict(state_dict, use_torch=True, device=self.device)
        elif self._obs_mode == "state_dict":
            obs = self._get_obs_state_dict(info)
        elif self._obs_mode == "pointcloud":
            # TODO support more flexible pcd obs mode with new render system
            obs = self._get_obs_with_sensor_data(info)
            obs = sensor_data_to_pointcloud(obs, self._sensors)
        elif self._obs_mode == "sensor_data":
            # return raw texture data dependent on choice of shader
            obs = self._get_obs_with_sensor_data(info, apply_texture_transforms=False)
        else:
            obs = self._get_obs_with_sensor_data(info)
            if 'hand_camera' in obs['sensor_data'] and self.model is not None:
                import numpy as np
                import cv2
                import torch
                import clip
                from torch import nn
                def cnn_register_hooks(model):
                    activations = []
                    def hook_fn(module, input, output):
                        activations.append(output)
                    hooks = []
                    for layer in model.cnn:
                        if isinstance(layer, nn.Conv2d):
                            hooks.append(layer.register_forward_hook(hook_fn))
                    return hooks, activations

                def cv2_visualization(model, activations, image, layer_idx=0):
                    image = image.cuda() / 255.0
                    origin_image = np.transpose(image[0].cpu().numpy(), (1, 2, 0)).copy()
                    model(image)
                    feature_map = activations[layer_idx].detach().cpu().numpy()
                    feature_map = np.squeeze(feature_map, axis=0)
                    feature_map = feature_map[0, :, :]
                    feature_map = np.maximum(feature_map, 0)
                    feature_map = feature_map / np.max(feature_map)
                    heatmap = cv2.applyColorMap(np.uint8(255 * feature_map), cv2.COLORMAP_JET)
                    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
                    if origin_image.shape[0] != heatmap.shape[0] or origin_image.shape[1] != heatmap.shape[1]:
                        heatmap = cv2.resize(heatmap, (origin_image.shape[1], origin_image.shape[0]))
                    origin_image = np.uint8(origin_image * 255.0)
                    overlay = cv2.addWeighted(origin_image, 0.7, heatmap, 0.5, 0)
                    # overlay = origin_image
                    return overlay


                # =====================================================================

                from pytorch_grad_cam import (
                    GradCAM,
                )
                from transformers import CLIPProcessor, CLIPModel

                from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
                from pytorch_grad_cam.ablation_layer import AblationLayerVit

                def reshape_transform(tensor, height=7, width=7):
                    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                    result = result.transpose(2, 3).transpose(1, 2)
                    return result

                class ImageClassifier(nn.Module):
                    def __init__(self, labels):
                        super(ImageClassifier, self).__init__()
                        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                        self.labels = labels

                    def forward(self, x):
                        text_inputs = self.preprocess(text=labels, return_tensors="pt", padding=True)
                        outputs = self.clip_model(pixel_values=x, input_ids=text_inputs['input_ids'],
                                            attention_mask=text_inputs['attention_mask'])
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                        # for label, prob in zip(self.labels, probs[0]):
                        #     print(f"{label}: {prob:.4f}")
                        return probs

                model = self.model.visual_encoder
                image = obs['sensor_data']['hand_camera']['rgb'].permute(0, 3, 1, 2).float()

                with torch.enable_grad():
                    if isinstance(model, PlainConv):
                        hooks, activations = cnn_register_hooks(model)
                        overlay = cv2_visualization(model, activations, image, layer_idx=3)
                        for hook in hooks:
                            hook.remove()
                    elif isinstance(model, CLIPEncoder):
                        labels = ["a suitcase", "a laptop", "a box"]
                        model = ImageClassifier(labels)
                        model.eval()
                        target_layers = [model.clip_model.vision_model.encoder.layers[-1].layer_norm1]
                        rgb_img = image[0].cpu().numpy().transpose(1, 2, 0)
                        rgb_img = cv2.resize(rgb_img, (224, 224))
                        rgb_img = np.float32(rgb_img) / 255
                        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
                        targets = None
                        cam.batch_size = 32
                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets,
                                            eigen_smooth=True, aug_smooth=True)
                        grayscale_cam = grayscale_cam[0, :]
                        overlay = show_cam_on_image(rgb_img, grayscale_cam)
                        # cv2.imwrite(f'grad_cam.jpg', overlay)

                # plt.figure(figsize=(10, 10))
                # plt.imshow(overlay)
                # plt.axis('off')
                # plt.title('Overlay Heatmap on Image')
                # plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
                self.observations.append(overlay)
        return obs

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        """get only data from sensors. Auto hides any objects that are designated to be hidden"""
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render()
        self.capture_sensor_data()
        sensor_obs = dict()
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera):
                if self.obs_mode in ["state", "state_dict"]:
                    # normally in non visual observation modes we do not render sensor observations. But some users may want to render sensor data for debugging or various algorithms
                    sensor_obs[name] = sensor.get_obs(position=False, segmentation=False, apply_texture_transforms=apply_texture_transforms)
                else:
                    sensor_obs[name] = sensor.get_obs(
                        rgb=self.obs_mode_struct.visual.rgb,
                        depth=self.obs_mode_struct.visual.depth,
                        position=self.obs_mode_struct.visual.position,
                        segmentation=self.obs_mode_struct.visual.segmentation,
                        normal=self.obs_mode_struct.visual.normal,
                        albedo=self.obs_mode_struct.visual.albedo,
                        apply_texture_transforms=apply_texture_transforms,
                        fisheye=sensor.camera.name=='hand_camera',
                    )
        # explicitly synchronize and wait for cuda kernels to finish
        # this prevents the GPU from making poor scheduling decisions when other physx code begins to run
        torch.cuda.synchronize()
        return sensor_obs

    def _get_obs_with_sensor_data(self, info: Dict, apply_texture_transforms: bool = True) -> dict:
        """Get the observation with sensor data"""
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
            sensor_param=self.get_sensor_params(),
            sensor_data=self._get_obs_sensor_data(apply_texture_transforms),
        )