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

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.encoders.plain_conv import PlainConv
from diffusion_policy.evaluate_video import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)

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

# Enhanced Video Encoder with Task-specific Feature Extraction
class VideoEncoder(nn.Module):
    def __init__(self, output_dim=64, backbone='resnet18', dropout_rate=0.3):
        super().__init__()
        # Select backbone architecture
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512

        # Remove the final FC layer from backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 3D Conv for capturing temporal patterns
        self.conv3d = nn.Sequential(
            nn.Conv3d(feature_dim, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Transformer encoder for capturing long-range dependencies
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout_rate
            ),
            num_layers=2
        )

        # Task-specific feature extraction
        self.task_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )

        # Auxiliary classifier for task discrimination
        self.task_classifier = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, len(prompt2task_dict))
        )

        self.feature_norm = nn.LayerNorm(128)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, video):
        # video shape: [B, T, H, W, C]
        batch, T, H, W, C = video.shape

        # Prepare storage for batch results
        all_task_features = []
        all_task_logits = []
        all_contrastive_features = []

        # Process each video sequence independently
        for b in range(batch):
            single_video = video[b]  # [T, H, W, C]
            single_video = single_video.permute(0, 3, 1, 2)

            # Extract features through backbone
            frame_features = self.backbone(single_video).squeeze(-1).squeeze(-1)  # [T, feature_dim]

            # Combine all frame features
            video_features = frame_features.unsqueeze(0)  # [1, T, feature_dim]

            # Apply 3D convolution to video features
            # Adjust dimensions for 3D convolution input format
            features_3d = video_features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [1, feature_dim, T, 1, 1]
            conv_features = self.conv3d(features_3d)  # [1, 128, T, 1, 1]
            conv_features = conv_features.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [1, T, 128]

            # Apply temporal attention
            attn_weights = self.temporal_attention(conv_features)  # [1, T, 1]
            attended_features = conv_features * attn_weights  # [1, T, 128]

            # Use Transformer for temporal modeling
            # Adjust to Transformer input format
            trans_input = attended_features.permute(1, 0, 2)  # [T, 1, 128]
            trans_output = self.transformer(trans_input)  # [T, 1, 128]
            trans_output = trans_output.permute(1, 0, 2)  # [1, T, 128]

            # Global feature
            global_feature = self.feature_norm(torch.mean(trans_output, dim=1))  # [1, 128]

            # Task feature
            task_feature = self.task_fc(global_feature)  # [1, output_dim]

            # Auxiliary classifier
            task_logit = self.task_classifier(task_feature)  # [1, num_classes]

            # Contrastive learning projection
            contrastive_feature = self.contrastive_projector(task_feature)  # [1, output_dim]
            contrastive_feature = F.normalize(contrastive_feature, p=2, dim=1)  # Normalize

            # Store results
            all_task_features.append(task_feature)
            all_task_logits.append(task_logit)
            all_contrastive_features.append(contrastive_feature)

        # Combine results from all videos
        batch_task_features = torch.cat(all_task_features, dim=0)  # [B, output_dim]
        batch_task_logits = torch.cat(all_task_logits, dim=0)  # [B, num_classes]
        batch_contrastive_features = torch.cat(all_contrastive_features, dim=0)  # [B, output_dim]

        return batch_task_features, batch_task_logits, batch_contrastive_features


# Data augmentation function for videos
def augment_video_batch(videos, strength=0.2):
    """
    Apply consistent augmentations to each video in the batch.

    Args:
        videos: Tensor of shape [B, T, H, W, C]
        strength: Augmentation strength factor

    Returns:
        Augmented videos tensor of same shape
    """
    B, T, H, W, C = videos.shape
    augmented = videos.clone()

    for b in range(B):
        # Apply consistent augmentation for each video
        if torch.rand(1).item() < 0.5:  # 50% chance of horizontal flip
            augmented[b] = torch.flip(augmented[b], [2])

        # Random brightness and contrast
        if torch.rand(1).item() < 0.8:  # 80% chance of adjustment
            brightness = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
            contrast = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
            augmented[b] = torch.clamp(contrast * (augmented[b] - 0.5) + 0.5 + brightness - 1, 0, 1)

    return augmented


# Feature visualization function
def visualize_features(features, labels, step, writer):
    """
    Create t-SNE visualization of feature embeddings and log to TensorBoard.

    Args:
        features: Feature embeddings tensor
        labels: Corresponding class labels tensor
        step: Current training step
        writer: TensorBoard SummaryWriter instance
    """
    try:
        from sklearn.manifold import TSNE
        import numpy as np
        import matplotlib.pyplot as plt

        # Convert to CPU numpy arrays
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_np)

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels_np):
            mask = labels_np == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f'Task {label}')

        plt.legend()
        plt.title(f'Feature Space Visualization - Step {step}')

        # Save to TensorBoard
        writer.add_figure('feature_visualization', plt.gcf(), step)
    except Exception as e:
        print(f"Failed to generate feature visualization: {e}")


# Dynamic loss weighting based on training progress
def get_loss_weights(current_step, total_steps):
    """
    Dynamically adjust loss weights based on training progress.

    Args:
        current_step: Current training step
        total_steps: Total number of training steps

    Returns:
        Dictionary of loss component weights
    """
    progress = current_step / total_steps

    if progress < 0.2:  # First 20% emphasizes feature learning
        return {
            'diffusion': 0.5,
            'cls': 0.2,
            'contrastive': 1.0,
            'center': 0.5,
            'inter_class': 0.5
        }
    elif progress < 0.5:  # Middle phase balances components
        return {
            'diffusion': 0.8,
            'cls': 0.1,
            'contrastive': 0.5,
            'center': 0.3,
            'inter_class': 0.3
        }
    else:  # Later phase emphasizes diffusion model
        return {
            'diffusion': 1.0,
            'cls': 0.05,
            'contrastive': 0.2,
            'center': 0.1,
            'inter_class': 0.1
        }


# Advanced NT-Xent contrastive loss with label information
def nt_xent_loss(features, labels, temperature=0.1):
    """
    Compute NT-Xent contrastive loss using label information.

    Args:
        features: Feature embeddings tensor [B, D]
        labels: Class labels tensor [B]
        temperature: Temperature scaling factor

    Returns:
        Computed contrastive loss
    """
    batch_size = features.shape[0]
    device = features.device

    # Create similarity matrix
    features_norm = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features_norm, features_norm.T) / temperature

    # Create mask for positive pairs (same label)
    positive_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
    # Remove self-comparisons
    identity_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - identity_mask

    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Compute log probabilities
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)

    # Final loss
    loss = -mean_log_prob_pos.mean()
    return loss


# Center loss for minimizing intra-class variance
def center_loss(features, labels):
    """
    Compute center loss to minimize intra-class variance.

    Args:
        features: Feature embeddings tensor [B, D]
        labels: Class labels tensor [B]

    Returns:
        Computed center loss
    """
    unique_labels = torch.unique(labels)
    device = features.device

    # Calculate per-class centroids
    centers = {}
    for label in unique_labels:
        mask = (labels == label)
        if mask.sum() > 0:
            centers[label.item()] = features[mask].mean(0, keepdim=True)

    # Calculate distances to centroids
    loss = torch.tensor(0., device=device)
    sample_count = 0

    for label in unique_labels:
        mask = (labels == label)
        if mask.sum() > 0:
            center = centers[label.item()]
            class_features = features[mask]
            # Squared Euclidean distance to center
            dist = torch.sum((class_features - center) ** 2, dim=1)
            loss += torch.sum(dist)
            sample_count += mask.sum()

    if sample_count > 0:
        loss = loss / sample_count

    return loss


# Inter-class distance loss to maximize separation between classes
def inter_class_distance_loss(features, labels):
    """
    Compute inter-class distance loss to maximize class separation.

    Args:
        features: Feature embeddings tensor [B, D]
        labels: Class labels tensor [B]

    Returns:
        Computed inter-class distance loss
    """
    unique_labels = torch.unique(labels)
    device = features.device

    if len(unique_labels) <= 1:
        return torch.tensor(0., device=device)

    # Calculate class centroids
    centers = []
    for label in unique_labels:
        mask = (labels == label)
        if mask.sum() > 0:
            center = features[mask].mean(0)
            # Normalize centroid
            center = F.normalize(center, p=2, dim=0)
            centers.append(center)

    centers = torch.stack(centers)

    # Calculate centroid similarity matrix
    similarity = torch.mm(centers, centers.t())

    # Create mask to exclude self-similarity
    mask = 1.0 - torch.eye(len(centers), device=device)

    # Calculate average similarity (to be minimized)
    loss = (similarity * mask).sum() / (len(centers) * (len(centers) - 1))

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
        default_factory=lambda: [48, 72, 96]
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

# 数据集类
class SmallDemoDataset_DiffusionPolicy(Dataset):
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
        label = prompt2label_dict[prompt]

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
        self.ftask_dim = 256
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
        self.video_encoder = VideoEncoder(output_dim=self.ftask_dim).to(device)

        # Define TargetNets
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (fobs_dim + obs_state_dim + self.ftask_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        ).to(device)
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

        self.loss_dict = {}
        # Add TensorBoard writer to the agent for feature visualization
        self.writer = None

        # Feature memory bank for tracking class centroids
        self.feature_memory = {}
        self.memory_momentum = 0.9  # For EMA updates of class centroids

        # Initialize class centers if any exist
        self.class_centers = {}

    def encode_obs(self, obs_seq, ftask, eval_mode):
        B = obs_seq["rgb"].shape[0]
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

        ftask = ftask.unsqueeze(1).repeat(1, self.obs_horizon, 1)  # (B, obs_horizon, ftask_dim)

        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        feature = torch.cat((feature, ftask), dim=-1) # (B, obs_horizon, D+obs_state_dim+ftask_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, data_batch, current_step=0, total_steps=100000):
        """
        Compute combined loss for training with dynamic loss weighting.

        Args:
            data_batch: Dictionary containing training batch data
            current_step: Current training iteration
            total_steps: Total training iterations

        Returns:
            Total combined loss
        """
        videos = data_batch["video"]
        obs_seq = data_batch["observations"]
        action_seq = data_batch["actions"]
        labels = data_batch["label"].to(self.device)
        B = obs_seq["state"].shape[0]

        # Apply data augmentation during training
        if self.training:
            videos = augment_video_batch(videos)

        # Get task features and task classification logits from video encoder
        ftask, task_logits, contrastive_features = self.video_encoder(videos)

        # Get dynamic loss weights based on training progress
        weights = get_loss_weights(current_step, total_steps)

        # 1. Classification Loss
        cls_loss = F.cross_entropy(task_logits, labels)

        # 2. Contrastive Loss using NT-Xent
        contrastive_loss = nt_xent_loss(contrastive_features, labels)

        # 3. Center Loss - minimize intra-class variance
        intra_class_loss = center_loss(contrastive_features, labels)

        # 4. Inter-class Distance Loss - maximize class separation
        inter_class_loss = inter_class_distance_loss(contrastive_features, labels)

        # 5. Diffusion model noise prediction loss
        # Generate weights for each TargetNet

        obs_cond = self.encode_obs(obs_seq, ftask, eval_mode=False)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(
            noisy_action_seq,
            timesteps,
            global_cond=obs_cond,
        )
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # Combine all losses with dynamic weighting
        total_loss = (
                weights['diffusion'] * diffusion_loss +
                weights['cls'] * cls_loss +
                weights['contrastive'] * contrastive_loss +
                weights['center'] * intra_class_loss +
                weights['inter_class'] * inter_class_loss
        )

        # Track individual losses for monitoring
        self.loss_dict = {
            'diffusion_loss': diffusion_loss.item(),
            'classification_loss': cls_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'center_loss': intra_class_loss.item(),
            'inter_class_loss': inter_class_loss.item(),
            'total_loss': total_loss.item()
        }

        # Periodically visualize feature space (every 1000 steps)
        if current_step % 1000 == 0 and hasattr(self, 'writer'):
            visualize_features(contrastive_features.detach(), labels.detach(), current_step, self.writer)

        return total_loss

    def get_action(self, obs_seq, val_videos, prompts):
        """
        Generate action based on observation sequence, video demonstrations and prompts.

        Args:
            obs_seq: Current observation sequence
            val_videos: Dictionary of video demonstrations for each task
            prompts: Task prompts corresponding to each environment in the batch

        Returns:
            Predicted actions
        """
        videos = []
        for prompt in prompts:
            task_name = prompt2task_dict[prompt]
            # Select a random video from available demonstrations for this task
            video_idx = random.randint(0, len(val_videos[task_name]) - 1)
            video = val_videos[task_name][video_idx].unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
            videos.append(video)
        videos = torch.cat(videos, dim=0)

        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            # Extract features from demonstration videos
            ftask, task_logits, contrastive_features = self.video_encoder(videos)

            # Encode current observations
            obs_cond = self.encode_obs(obs_seq, ftask, eval_mode=True)

            # Initialize noisy action sequence
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

            # Progressive denoising
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    noisy_action_seq,
                    k,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(noise_pred, k, noisy_action_seq).prev_sample

            # Extract the relevant action horizon
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

    dataset = SmallDemoDataset_DiffusionPolicy(
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
    diffusion_params = list(agent.noise_pred_net.parameters())
    video_encoder_params = list(agent.video_encoder.parameters())
    obs_encoder_params = list(agent.obs_encoder.parameters())

    # Configure parameter groups with different learning rates
    param_groups = [
        {"params": diffusion_params, "lr": args.lr, "name": "diffusion"},
        {"params": video_encoder_params, "lr": args.lr, "name": "video_encoder"},
        {"params": obs_encoder_params, "lr": args.lr, "name": "obs_encoder"}
    ]
    # Initialize optimizer with parameter groups
    optimizer = optim.AdamW(
        params=param_groups,
        betas=(0.95, 0.999),
        weight_decay=1e-6
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,
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
            if np.mean(eval_metrics['success_at_end']) >= 0.4:
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

    # Training loop
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()

    # Assign writer to agent for visualization
    agent.writer = writer

    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # Forward pass and loss calculation
        last_tick = time.time()
        total_loss = agent.compute_loss(data_batch, current_step=iteration, total_steps=args.total_iters)
        timings["forward"] += time.time() - last_tick

        # Backward pass
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - last_tick

        # Evaluation and logging
        evaluate_and_save_best(iteration, val_videos)
        log_metrics(iteration)

        # Periodically update feature memory bank
        if iteration % 100 == 0 and hasattr(agent, 'feature_memory'):
            # Extract features from current batch
            with torch.no_grad():
                _, _, features = agent.video_encoder(data_batch["video"])
                labels = data_batch["label"].to(device)

                # Update feature memory bank
                for i in range(features.shape[0]):
                    label = labels[i].item()
                    if label not in agent.feature_memory:
                        agent.feature_memory[label] = features[i].detach()
                    else:
                        # Exponential moving average update
                        agent.feature_memory[label] = (
                                agent.memory_momentum * agent.feature_memory[label] +
                                (1 - agent.memory_momentum) * features[i].detach()
                        )

            # Log feature statistics
            if iteration % 1000 == 0:
                # Calculate and log inter-class distances
                if len(agent.feature_memory) > 1:
                    centers = torch.stack(list(agent.feature_memory.values()))
                    centers_norm = F.normalize(centers, p=2, dim=1)
                    similarity = torch.mm(centers_norm, centers_norm.t())

                    # Mask out diagonal
                    mask = 1.0 - torch.eye(len(centers), device=device)
                    avg_sim = (similarity * mask).sum() / (len(centers) * (len(centers) - 1))

                    writer.add_scalar("metrics/inter_class_similarity", avg_sim.item(), iteration)

        # Periodic checkpoint saving
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))

        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    # Final evaluation and logging
    evaluate_and_save_best(args.total_iters, val_videos)
    log_metrics(args.total_iters)

    envs.close()
    writer.close()