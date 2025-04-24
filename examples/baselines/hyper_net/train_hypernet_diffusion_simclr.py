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
from tqdm import tqdm
import tyro
import h5py
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from .hyper_net.evaluate_diffusion import evaluate
from .hyper_net.make_env import make_eval_envs
from .hyper_net.utils import (IterationBasedBatchSampler, build_state_obs_extractor,
                                    convert_obs, worker_init_fn)
from .hyper_net.hypernetwork_diffusion import UNetPolicy
from .hyper_net.hypernetwork_v0 import Hypernet
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.encoders.plain_conv import PlainConv

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

prompt2label_dict = {}
label_counter = 0
for prompt in prompt2task_dict.keys():
    if prompt not in prompt2label_dict:
        prompt2label_dict[prompt] = label_counter
        label_counter += 1


class VideoTemporalAugmenter:
    def __init__(self, temporal_mask_ratio=0.2, frame_dropout_ratio=0.2,
                 temporal_shuffle_prob=0.5):
        self.temporal_mask_ratio = temporal_mask_ratio
        self.frame_dropout_ratio = frame_dropout_ratio
        self.temporal_shuffle_prob = temporal_shuffle_prob

    def __call__(self, video):
        """
        Apply semantic-preserving temporal augmentations
        video: [B, T, H, W, C]
        """
        B, T, H, W, C = video.shape
        augmented_video = video.clone()

        # Apply one augmentation randomly - this preserves semantic content better
        aug_choice = random.randint(0, 3)

        if aug_choice == 0 and T > 2:
            # Temporal masking - mask random frames (self-supervision task)
            mask_size = max(1, int(T * self.temporal_mask_ratio))
            start_idx = random.randint(0, T - mask_size)
            # Instead of zeroing, we can use a more subtle masking
            mask = torch.ones_like(augmented_video)
            mask[:, start_idx:start_idx + mask_size] = 0.3  # Dim but not completely zero
            augmented_video = augmented_video * mask

        elif aug_choice == 1 and T > 2:
            # Frame dropout - drop frames and repeat previous
            # This preserves semantic content by copying from neighboring frames
            drop_frames = torch.rand(T) < self.frame_dropout_ratio
            if drop_frames.sum() > 0:  # Only if we have frames to drop
                for t in range(1, T):
                    if drop_frames[t]:
                        augmented_video[:, t] = augmented_video[:, t - 1]

        elif aug_choice == 2 and T > 4:
            # Temporal shuffling of small chunks
            # Keep chunks small to preserve local temporal structures
            chunk_size = max(2, T // 5)
            num_chunks = T // chunk_size
            if num_chunks > 1:
                chunks = [augmented_video[:, i * chunk_size:(i + 1) * chunk_size].clone()
                          for i in range(num_chunks)]
                # Only shuffle some chunks, not all
                num_to_shuffle = max(2, num_chunks // 2)
                indices_to_shuffle = random.sample(range(num_chunks), num_to_shuffle)
                shuffle_indices = torch.randperm(num_to_shuffle).tolist()

                for i, idx in enumerate(indices_to_shuffle):
                    new_idx = indices_to_shuffle[shuffle_indices[i]]
                    augmented_video[:, idx * chunk_size:(idx + 1) * chunk_size] = chunks[new_idx]

        elif aug_choice == 3 and T > 2:
            # Temporal subsampling and interpolation
            # This preserves semantic meaning while changing temporal resolution
            keep_fraction = random.uniform(0.6, 0.9)  # Keep 60-90% of frames
            keep_count = max(2, int(T * keep_fraction))
            keep_indices = sorted(random.sample(range(T), keep_count))

            sampled_frames = augmented_video[:, keep_indices]

            # Reshape for interpolation
            B_new, T_new, H, W, C = sampled_frames.shape
            sampled_frames = sampled_frames.reshape(B_new, T_new, C, H, W).permute(0, 2, 1, 3, 4)

            # Interpolate back to original length using trilinear interpolation
            interpolated = F.interpolate(
                sampled_frames,
                size=(T, H, W),
                mode='trilinear',
                align_corners=False
            )

            # Reshape back to original format
            augmented_video = interpolated.permute(0, 2, 3, 4, 1)

        return augmented_video


class VideoEncoder(nn.Module):
    def __init__(self, output_dim=128, projection_dim=64, temperature=0.07):
        super().__init__()

        # Backbone - keeping the lighter MobileNetV3 architecture
        self.backbone = models.mobilenet_v3_small(pretrained=True).features[:8]
        self.feature_dim = 40  # MobileNetV3 small layer 8 output channels

        # Temporal feature extractor using 1D convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # Feature processor
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim)
        )

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(output_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

        # Video augmenter
        self.augmenter = VideoTemporalAugmenter()
        self.temperature = temperature

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Video Encoder parameters: {n_params / 1e6:.2f}M")

    def encode_video(self, video):
        batch, T, H, W, C = video.shape

        # Process frames
        video = video.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        video_features = []
        for i in range(batch):
            single_video = video[i]
            video_feature = self.backbone(single_video)
            video_features.append(video_feature)
        features = torch.cat(video_features, dim=0)  # [B*T, feature_dim, h, w]

        # Get the actual output feature dimension from the backbone
        _, C_out, h, w = features.shape

        # Global pooling on spatial dimensions
        features = F.adaptive_avg_pool2d(features, (1, 1))  # [B*T, C_out, 1, 1]
        features = features.view(batch, T, -1)  # [B, T, C_out]

        # Check if the feature dimension matches what we expect
        actual_feature_dim = features.shape[2]

        # If there's a mismatch, adjust the temporal_conv to match the actual feature dimension
        if actual_feature_dim != self.feature_dim:
            # Create a new 1D convolution to match the dimensions
            device = features.device
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(actual_feature_dim, 32, kernel_size=3, padding=1).to(device),
                nn.BatchNorm1d(32).to(device),
                nn.ReLU(inplace=True)
            )
            # Update the feature_dim to match the actual dimension
            self.feature_dim = actual_feature_dim
            print(f"Adjusted temporal_conv to match feature dimension: {actual_feature_dim}")

        # Permute for temporal convolution
        features = features.permute(0, 2, 1)  # [B, C_out, T]

        # Apply 1D convolution for temporal modeling
        features = self.temporal_conv(features)  # [B, 32, T]

        # Process features
        features = self.feature_processor(features)  # [B, output_dim]

        return features

    def forward(self, videos, labels=None, train=True, num_augs=2):
        if train and labels is not None:
            batch_size = videos.shape[0]

            # Create augmented versions of each video
            all_videos = [videos]  # Original videos
            for _ in range(num_augs):
                augmented = self.augmenter(videos.clone())  # Apply temporal augmentation
                all_videos.append(augmented)

            # Stack all versions: [B*(num_augs+1), T, H, W, C]
            all_videos_stacked = torch.cat(all_videos, dim=0)

            # Encode all videos
            all_features = self.encode_video(all_videos_stacked)  # [B*(num_augs+1), output_dim]
            all_projections = F.normalize(self.projector(all_features), dim=1)  # [B*(num_augs+1), projection_dim]

            # Reshape to group by original video
            features_grouped = all_features.view(num_augs + 1, batch_size, -1)  # [num_augs+1, B, output_dim]
            projections_grouped = all_projections.view(num_augs + 1, batch_size, -1)  # [num_augs+1, B, projection_dim]

            # Original features for task representation
            task_features = features_grouped[0]  # [B, output_dim]

            # All projections for contrastive learning (transpose to get [B, num_augs+1, dim])
            projections_for_contrast = projections_grouped.transpose(0, 1)  # [B, num_augs+1, projection_dim]

            return task_features, projections_for_contrast, labels
        else:
            # Inference mode - just return features of original video
            task_features = self.encode_video(videos)
            projections = F.normalize(self.projector(task_features), dim=1)
            return task_features, projections, None

    def supervised_contrastive_loss(self, projections_grouped, labels):
        """
        Compute supervised contrastive loss
        - Positive pairs: Different augmentations of the same video (same instance)
                          + Different videos with the same label (same task)
        - Negative pairs: Videos with different labels (different tasks)

        projections_grouped: [batch_size, num_views, projection_dim]
        labels: [batch_size]
        """
        batch_size, num_views, projection_dim = projections_grouped.shape
        device = projections_grouped.device
        labels = labels.to(device)

        # Reshape to [batch_size*num_views, projection_dim]
        projections = projections_grouped.reshape(-1, projection_dim)

        # Create expanded labels to match each augmentation
        expanded_labels = labels.repeat_interleave(num_views)

        # Compute similarity matrix
        similarity = torch.matmul(projections, projections.T) / self.temperature  # [B*num_views, B*num_views]

        # Create mask for positive pairs
        # 1. Different augmentations of the same video
        mask_same_instance = torch.zeros((batch_size * num_views, batch_size * num_views), device=device)
        for i in range(batch_size):
            for a in range(num_views):
                for b in range(num_views):
                    if a != b:  # Different views of same video
                        mask_same_instance[i * num_views + a, i * num_views + b] = 1

        # 2. Different videos with the same label (same task)
        mask_same_label = (expanded_labels.unsqueeze(0) == expanded_labels.unsqueeze(1)).float()
        # Remove self-similarity
        mask_self = torch.eye(batch_size * num_views, device=device)
        mask_same_label = mask_same_label * (1 - mask_self)
        # Remove same instance pairs (already counted in mask_same_instance)
        mask_same_label = mask_same_label - mask_same_instance
        mask_same_label = torch.clamp(mask_same_label, min=0.0)

        # Combined mask: both same instance and same label are positive pairs
        mask_positives = mask_same_instance + mask_same_label

        # For numerical stability
        sim_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_max.detach()

        # Compute exp(similarity)
        exp_sim = torch.exp(similarity)

        # Compute positive and negative similarities
        pos_sim = torch.sum(exp_sim * mask_positives, dim=1)
        neg_sim = torch.sum(exp_sim * (1 - mask_self), dim=1)

        # Compute final loss
        loss = -torch.log(pos_sim / neg_sim)

        # Average over non-zero elements
        n_pos = torch.sum(mask_positives, dim=1)
        n_pos = torch.clamp(n_pos, min=1.0)  # Avoid division by zero
        loss = torch.sum(loss / n_pos) / batch_size

        return loss

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
    act_horizon: int = 4  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 32  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [32, 48, 64]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        4  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are similar
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
        label = prompt2label_dict[prompt] if prompt in prompt2label_dict else -1

        return {
            "video": video,
            "observations": obs_seq,
            "actions": act_seq,
            "language": prompt,
            "label": label,
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
        ftask_dim = 512
        weight_dim = 128
        deriv_hidden_dim = 48
        driv_num_layers = 3
        codec_hidden_dim = 96
        codec_num_layers = 3
        num_layers = 3
        if args.visual_encoder == 'plain_conv':
            self.obs_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=fobs_dim, pool_feature_map=True
            ).to(device)
        elif args.visual_encoder == 'resnet':
            from diffusion_policy.encoders.resnet import ResNetEncoder
            self.obs_encoder = ResNetEncoder(
                in_channels=total_visual_channels, out_dim=fobs_dim, pool_feature_map=True
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
            # ftask_dim=ftask_dim,
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

        # Add He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.loss_dict = {}

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
        labels = data_batch["label"]
        B = obs_seq["state"].shape[0]

        # Get task features and contrastive features from video encoder with labels
        task_features, projections_for_contrast, _ = self.video_encoder(
            videos,
            labels=labels,
            train=True,
            num_augs=2
        )

        # 1. Supervised Contrastive Loss - ensures task-level similarity
        # This is now a label-aware contrastive loss that considers tasks
        contrastive_loss = self.video_encoder.supervised_contrastive_loss(
            projections_for_contrast,
            labels
        )

        # 2. Diffusion model loss - original noise prediction loss
        # Generate weights for each TargetNet using the task features
        down_path_params = self.hypernet_down_path.forward_blocks(task_features)[-1]
        up_path_params = self.hypernet_up_path.forward_blocks(task_features)[-1]

        obs_cond = self.encode_obs(obs_seq, eval_mode=False)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(
            noisy_action_seq,
            timesteps,
            global_cond=obs_cond,
            down_path_params=down_path_params,
            up_path_params=up_path_params,
            # ftask=contrastive_features,
        )
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # Combine losses - no need for a separate diversity loss as it's built into the
        # supervised contrastive loss (pushing different tasks apart)
        total_loss = diffusion_loss + 0.1 * contrastive_loss

        # Track individual losses for monitoring
        self.loss_dict = {
            'diffusion_loss': diffusion_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss

    def get_action(self, obs_seq, val_videos, prompts):
        videos = []
        labels = []

        for prompt in prompts:
            task_name = prompt2task_dict[prompt]
            label = prompt2label_dict[prompt] if prompt in prompt2label_dict else -1
            video_idx = random.randint(0, len(val_videos[task_name]) - 1)
            video = val_videos[task_name][video_idx].unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
            videos.append(video)
            labels.append(label)

        videos = torch.cat(videos, dim=0)
        labels = torch.tensor(labels, device=self.device)

        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            # During inference, we don't need augmentation
            task_features, _, _ = self.video_encoder(videos, labels=labels, train=False)

            # Use learned features for hypernetwork
            down_path_params = self.hypernet_down_path.forward_blocks(task_features)[-1]
            up_path_params = self.hypernet_up_path.forward_blocks(task_features)[-1]

            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    noisy_action_seq,
                    k,
                    global_cond=obs_cond,
                    down_path_params=down_path_params,
                    up_path_params=up_path_params,
                    # ftask=contrastive_features
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
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

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

    # val_videos = train_videos

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
    # Replace the optimizer initialization with this code in the main script
    # Group parameters by component
    diffusion_params = list(agent.noise_pred_net.parameters()) + \
                       list(agent.hypernet_down_path.parameters()) + \
                       list(agent.hypernet_up_path.parameters())

    video_encoder_params = list(agent.video_encoder.parameters())
    obs_encoder_params = list(agent.obs_encoder.parameters())

    # Configure parameter groups with different learning rates
    param_groups = [
        {"params": diffusion_params, "lr": args.lr, "name": "diffusion"},
        {"params": video_encoder_params, "lr": args.lr * 0.5, "name": "video_encoder"},
        {"params": obs_encoder_params, "lr": args.lr, "name": "obs_encoder"}
    ]
    # Initialize optimizer with parameter groups
    optimizer = optim.AdamW(
        params=param_groups,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=args.save_freq
    )

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def evaluate_and_save_best(iteration, val_videos):
        if iteration % args.eval_freq == 0 and iteration != 0:
            last_tick = time.time()
            eval_metrics = evaluate(
                10, agent, envs, device, args.sim_backend, val_videos=val_videos
            )
            if np.mean(eval_metrics['success_at_end']) >= 0.5:
                print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
                for k in eval_metrics.keys():
                    eval_metrics[k] = np.mean(eval_metrics[k])
                    writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                    print(f"{k}: {eval_metrics[k]:.4f}")
                print("Small scaled success_at_end >= 0.5, no re-evaluate with more episodes...")
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
            # Log these losses if desired
            for k, v in agent.loss_dict.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
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
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
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