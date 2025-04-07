ALGO_NAME = "BC_hypernet_step_encoder"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import tyro
import h5py
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from .hyper_net.evaluate_diffusion import evaluate
from .hyper_net.make_env import make_eval_envs
from .hyper_net.utils import (IterationBasedBatchSampler, build_state_obs_extractor,
                                    convert_obs, worker_init_fn)
from .hyper_net.hypernetwork_diffusion import UNetPolicy
from .hyper_net.hypernetwork import Hypernet
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.encoders.plain_conv import PlainConv


# class VideoEncoder(nn.Module):
#     def __init__(self, output_dim=64):
#         super().__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
#         self.fc = nn.Linear(512, output_dim)
#
#     def forward(self, video):
#         batch, T, H, W, C = video.shape
#         video = video.permute(0, 1, 4, 2, 3)
#         video = video.view(batch * T, C, H, W)
#         features = self.resnet(video)
#         features = features.view(batch, T, 512)
#         features = features.mean(dim=1)
#         task_feature = self.fc(features)
#         return task_feature

class VideoEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        # Replace ResNet with PlainConv
        self.cnn = PlainConv(
            in_channels=3,
            out_dim=512,  # Match the dimension of ResNet18 features
            pool_feature_map=True,
            last_act=True
        )
        # Final FC layer similar to original
        self.fc = nn.Linear(512, output_dim)

    def forward(self, video):
        # Handle the same input format as the original encoder
        batch, T, H, W, C = video.shape
        video = video.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        video = video.reshape(batch * T, C, H, W)
        # Extract features using PlainConv instead of ResNet
        features = self.cnn(video)
        # Reshape and pool temporal dimension, same as original
        features = features.view(batch, T, 512)
        features = features.mean(dim=1)
        # Final projection
        task_feature = self.fc(features)
        return task_feature

# 配置参数
@dataclass
class Args:
    video_path: str= "processed_data"
    """Where the prompt video at"""
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 32  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [32, 64, 128]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        4  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    shader: str = "default"
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. 
    Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    visual_encoder: str = "plain_conv"
    """Vision encoder. can be "plain_conv", "clip", "dinov2", "resnet"""
    prompt: str=""
    """Prompt for language condition"""


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if k not in ['prompt']:
            if isinstance(v, dict) or isinstance(v, spaces.Dict):
                out[k] = reorder_keys(d[k], ref_dict[k])
            else:
                out[k] = d[k]
    return out


prompt2task_dict = {
    "pick red cube and place on plate.": "human_pick_red_cube_place_plate",
    "pick blue cube and place on plate.": "human_pick_blue_cube_place_plate",
    "pick yellow cup and place on plate.": "human_pick_cup_place_plate",
    "stack red cube on blue cube.": "human_stack_red_cube_on_blue_cube",
    "stack blue cube on red cube.": "human_stack_blue_cube_on_red_cube",
    "pick red cube and place on yellow cup.": "human_pick_red_cube_place_cup",
    "pick blue cube and place on yellow cup .": "human_pick_blue_cube_place_cup",
    "pick yellow cup and pour and place on plate.": "human_pour_cup",
    "": "human_pick_red_cube_place_plate"
}

# 数据集类
class HypernetDataset(Dataset):
    def __init__(self, data_path, videos, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj=None,
                 use_language=True):
        self.device = device
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.videos = videos
        self.use_language = use_language
        obs_process_fn = obs_process_fn
        obs_space = obs_space

        # Load real robot demonstration data using Diffusion Policy's utility
        from diffusion_policy.utils import load_demo_dataset_with_lan
        trajectories = load_demo_dataset_with_lan(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process observations to align with the environment's observation space
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float16))
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"])
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())

        # Process language descriptions if available
        if self.use_language and "language" in trajectories:
            self.trajectory_language = trajectories["language"]
        else:
            # Create empty language placeholders if not available in dataset
            self.trajectory_language = [args.prompt] * len(trajectories["actions"])

        # Pre-process actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

        # Define horizons (assuming these are passed via args or another mechanism)
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            ).to(self.device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon = obs_horizon = args.obs_horizon  # e.g., 2
        self.pred_horizon = pred_horizon = args.pred_horizon  # e.g., 16

        # Pre-compute all possible (traj_idx, start, end) tuples
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start):start + self.obs_horizon
            ].to(self.device)
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0).to(self.device)

        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end].to(self.device)
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)

        assert (
                obs_seq["state"].shape[0] == self.obs_horizon
                and act_seq.shape[0] == self.pred_horizon
        )

        prompt = self.trajectory_language[traj_idx]
        task_name = prompt2task_dict[prompt]
        video_idx = random.randint(0, len(self.videos[task_name])-1)
        video = self.videos[task_name][video_idx].to(self.device, dtype=torch.float32) / 255.0

        return {
            "video": video,
            "observations": obs_seq,
            "actions": act_seq,
            "language": prompt
        }

    def __len__(self):
        return len(self.slices)


# Agent 类
class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args, device):
        super().__init__()
        self.device = device

        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
                len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
                env.single_action_space.low == -1
        ).all()

        # 从环境获取动作维度
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]

        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        # 初始化你的网络组件
        fobs_dim = 256
        ftask_dim = 256
        weight_dim = 128
        deriv_hidden_dim = 64
        driv_num_layers = 4
        codec_hidden_dim = 128
        codec_num_layers = 4
        num_layers = 8
        if args.visual_encoder == 'plain_conv':
            self.obs_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=fobs_dim, pool_feature_map=True
            ).to(device)
        elif args.visual_encoder == 'clip':
            from diffusion_policy.encoders.clip import CLIPEncoder
            self.obs_encoder = CLIPEncoder(
                out_dim=fobs_dim
            ).to(device)
        elif args.visual_encoder == 'dinov2':
            from diffusion_policy.encoders.dinov2 import DINOv2Encoder
            self.obs_encoder = DINOv2Encoder(
                out_dim=fobs_dim
            ).to(device)
        elif args.visual_encoder == 'resnet':
            from diffusion_policy.encoders.resnet import ResNetEncoder
            self.obs_encoder = ResNetEncoder(
                out_dim=fobs_dim, pool_feature_map=True
            ).to(device)
        elif args.visual_encoder == 'siglip':
            from diffusion_policy.encoders.siglip import SigLIP2Encoder
            self.obs_encoder = SigLIP2Encoder(
                out_dim=fobs_dim
            ).to(device)
        # Noise scheduler
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Video encoder
        self.video_encoder = VideoEncoder(output_dim=ftask_dim).to(device)

        # Define TargetNets
        self.noise_pred_net = UNetPolicy(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (fobs_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        ).to(device)
        self.down_path_target = self.noise_pred_net.unet.down_path_target
        self.up_path_target = self.noise_pred_net.unet.up_path_target

        # Define Hypernets for each TargetNet
        self.hypernet_down_path = Hypernet(
            self.down_path_target, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers,
            codec_hidden_dim, codec_num_layers, num_layers
        ).to(device)
        self.hypernet_up_path = Hypernet(
            self.up_path_target, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers,
            codec_hidden_dim, codec_num_layers, num_layers
        ).to(device)

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.obs_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, data_batch):
        videos = data_batch["video"]
        obs_seq = data_batch["observations"]
        action_seq = data_batch["actions"]
        B = obs_seq["state"].shape[0]

        ftask = self.video_encoder(videos)

        # Generate weights for each TargetNet
        down_path_params = self.hypernet_down_path.forward_blocks(ftask)[-1]
        up_path_params = self.hypernet_down_path.forward_blocks(ftask)[-1]

        obs_cond = self.encode_obs(obs_seq, eval_mode=False)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(
            noisy_action_seq,
            timesteps,
            global_cond=obs_cond.detach(),
            down_path_params=down_path_params,
            up_path_params=up_path_params
        )
        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq, val_videos, prompts):
        videos = []
        for prompt in prompts:
            task_name = prompt2task_dict[prompt]
            video_idx = random.randint(0, len(val_videos[task_name])-1)
            video = val_videos[task_name][video_idx].unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
            videos.append(video)
        videos = torch.cat(videos, dim=0)

        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            ftask = self.video_encoder(videos)

            # Generate weights for each TargetNet
            down_path_params = self.hypernet_down_path.forward_blocks(ftask)[-1]
            up_path_params = self.hypernet_down_path.forward_blocks(ftask)[-1]

            obs_cond = self.encode_obs(obs_seq, eval_mode=False)
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    noisy_action_seq,
                    k,
                    global_cond=obs_cond.detach(),
                    down_path_params=down_path_params,
                    up_path_params=up_path_params
                )
                noisy_action_seq = self.noise_scheduler.step(noise_pred, k, noisy_action_seq).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    # 设置实验名称
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 确定设备
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")

    # 创建环境参数
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack=args.shader),
        # mode="eval"
    )
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    # 创建评估环境
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper]
    )
    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="HyperNet",
            tags=["hyper_net"],
        )

    # 初始化 TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # 定义观察处理函数
    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth="rgbd" in args.demo_path
    )

    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
    tmp_env = gym.make(args.env_id, **env_kwargs)
    original_obs_space = tmp_env.observation_space
    # determine whether the env will return rgb and/or depth data
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    # Load video data from HDF5 files
    videos = {}
    train_videos = {}
    val_videos = {}
    val_num_per_task = 5
    video_files = [f for f in os.listdir(args.video_path) if f.endswith(".h5")]
    num_tasks = len(video_files)
    print(f"Detected {num_tasks} tasks from HDF5 files.")

    for video_file in video_files:
        task_name = video_file.replace(".h5", "")
        hdf5_path = os.path.join(args.video_path, video_file)
        with h5py.File(hdf5_path, 'r') as h5f:
            num_videos = len(h5f)
            videos[task_name] = []
            for i in range(min(num_videos, 50)):  # videos_per_task=50
                group = h5f[str(i)]
                video = group["obs"][:]
                videos[task_name].append(torch.tensor(video, dtype=torch.uint8))

            # 随机划分 train/val
            indices = list(range(len(videos[task_name])))
            random.shuffle(indices)
            val_indices = indices[:val_num_per_task]
            train_indices = indices[val_num_per_task:]
            train_videos[task_name] = [videos[task_name][i] for i in train_indices]
            val_videos[task_name] = ([videos[task_name][i] for i in val_indices])

    dataset = HypernetDataset(
        data_path=args.demo_path,
        videos=train_videos,
        obs_process_fn=obs_process_fn,
        obs_space=original_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device,
        num_traj=args.num_demos
    )

    # 设置数据加载器
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    # 初始化代理
    agent = Agent(envs, args, device=device)
    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters
    )

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def evaluate_and_save_best(iteration, val_videos):
        if iteration % args.eval_freq == 0 and iteration != 0:
            last_tick = time.time()
            eval_metrics = evaluate(
                args.num_eval_episodes, agent, envs, device, args.sim_backend, val_videos=val_videos
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # 训练循环
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # 前向传播和损失计算
        last_tick = time.time()
        total_loss = agent.compute_loss(data_batch)
        timings["forward"] += time.time() - last_tick

        # 反向传播
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - last_tick

        # 评估和日志记录
        evaluate_and_save_best(iteration, val_videos)
        log_metrics(iteration)

        # 定期保存检查点
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))

        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    # 最终评估和日志记录
    evaluate_and_save_best(args.total_iters, val_videos)
    log_metrics(args.total_iters)

    envs.close()
    writer.close()