ALGO_NAME = "BC_Diffusion_rgbd_video_UNet"

import os
import random
import time
import h5py
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate_video import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.encoders.plain_conv import PlainConv
import torchvision.models as models

# Import the mapping dictionaries from train_rgbd_video.py
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


# Enhanced Video Encoder (copied from train_rgbd_video.py)
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
            contrastive_feature = torch.nn.functional.normalize(contrastive_feature, p=2, dim=1)  # Normalize

            # Store results
            all_task_features.append(task_feature)
            all_task_logits.append(task_logit)
            all_contrastive_features.append(contrastive_feature)

        # Combine results from all videos
        batch_task_features = torch.cat(all_task_features, dim=0)  # [B, output_dim]
        batch_task_logits = torch.cat(all_task_logits, dim=0)  # [B, num_classes]
        batch_contrastive_features = torch.cat(all_contrastive_features, dim=0)  # [B, output_dim]

        return batch_task_features, batch_task_logits, batch_contrastive_features


@dataclass
class Args:
    exp_name: Optional[str] = None
    ckpt_exp_name: Optional[str] = None
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

    video_path: str = "processed_data"
    """Where the prompt video is located"""

    prompt: str = ""
    """Prompt for language condition"""

    # Diffusion Policy specific arguments
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 4  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 32  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [48, 72, 96]
    )  # default setting is about ~4.5M params
    n_groups: int = 4  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    shader: str = "default"
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. 
    Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    visual_encoder: str = "plain_conv"
    """Vision encoder. can be "plain_conv", "clip", "dinov2", "resnet"""
    depth: bool = False
    """use depth to eval"""


class Agent(nn.Module):
    def __init__(self, env, args, device):
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

        # Action dimension
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]

        # Determine visual channels
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        # Initialize network components
        fobs_dim = 256
        self.ftask_dim = 512

        # Observation encoder
        if args.visual_encoder == 'plain_conv':
            self.obs_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=fobs_dim, pool_feature_map=True
            ).to(device)
        elif args.visual_encoder == 'resnet':
            from diffusion_policy.encoders.resnet import ResNetEncoder
            self.obs_encoder = ResNetEncoder(
                in_channels=total_visual_channels, out_dim=fobs_dim, pool_feature_map=True
            ).to(device)

        # Video encoder
        self.video_encoder = VideoEncoder(output_dim=self.ftask_dim).to(device)

        # Noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * (fobs_dim + obs_state_dim + self.ftask_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        ).to(device)

        # Noise scheduler
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

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
        feature = torch.cat((feature, ftask), dim=-1)  # (B, obs_horizon, D+obs_state_dim+ftask_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim+ftask_dim))

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


def load_ckpt(run_name, tag, agent):
    checkpoint = torch.load(f"runs/{run_name}/checkpoints/{tag}.pt")
    agent.load_state_dict(checkpoint["agent"])
    return agent


def load_videos(video_path):
    """
    Load demonstration videos from HDF5 files.

    Args:
        video_path: Path to the directory containing video HDF5 files

    Returns:
        Dictionary of videos indexed by task name
    """
    videos = {}
    video_files = [f for f in os.listdir(video_path) if f.endswith(".h5")]

    for video_file in video_files:
        task_name = video_file.replace(".h5", "")
        hdf5_path = os.path.join(video_path, video_file)

        with h5py.File(hdf5_path, 'r') as h5f:
            num_videos = len(h5f)
            videos[task_name] = []
            for i in range(min(num_videos, 20)):  # Limit to 20 videos per task for evaluation
                group = h5f[str(i)]
                video = group["obs"][:]
                videos[task_name].append(torch.tensor(video, dtype=torch.uint8))

    return videos


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.ckpt_exp_name is None:
        args.ckpt_exp_name = os.path.basename(__file__)[: -len(".py")]
        ckpt_name = f"{args.env_id}__{args.ckpt_exp_name}__{args.seed}__{int(time.time())}"
    else:
        ckpt_name = args.ckpt_exp_name

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load demonstration videos
    eval_videos = load_videos(args.video_path)
    print(f"Loaded demonstration videos for {len(eval_videos)} tasks")

    # Create evaluation environments
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
    )

    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    other_kwargs = dict(obs_horizon=args.obs_horizon)

    # Create eval environment with appropriate wrappers
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[partial(FlattenRGBDObservationWrapper, sep_depth=True)],
    )

    # Initialize tensorboard writer
    writer = SummaryWriter(f"runs/{ckpt_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Initialize agent
    agent = Agent(envs, args, device)

    # Load checkpoint
    agent = load_ckpt(ckpt_name, "best_eval_success_at_end", agent)
    print(f"Loaded checkpoint from runs/{ckpt_name}/checkpoints/best_eval_success_at_end.pt")

    # Evaluate agent
    print("==================== Eval Begin ====================")
    last_tick = time.time()

    # Set evaluation prompt if provided
    eval_prompt = args.prompt
    if not eval_prompt:
        # Use default prompt if none provided
        eval_prompt = list(prompt2task_dict.keys())[0]

    # Create prompts for each environment
    prompts = [eval_prompt] * args.num_eval_envs

    # Run evaluation
    eval_metrics = evaluate(
        args.num_eval_episodes, agent, envs, device, args.sim_backend, val_videos=eval_videos
    )

    eval_time = time.time() - last_tick

    print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        writer.add_scalar(f"eval/{k}", eval_metrics[k], 1)
        print(f"{k}: {eval_metrics[k]:.4f}")

    print(f"Evaluation time: {eval_time:.2f} seconds")
    print("==================== Eval End ====================")

    envs.close()
    writer.close()