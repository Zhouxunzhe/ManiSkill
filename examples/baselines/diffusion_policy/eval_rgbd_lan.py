ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)


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
    """Vision encoder. can be "plain_conv", "clip", "dinov2", "resnet", "siglip"""
    lan_encoder: str = "encoder_only"
    """Language encoder. can be "encoder_only", "tokenizer_only", "encoder_decoder", "tokenizer_decoder", "encoder_ffn", "tokenizer_ffn"""
    language_condition_type: str = "concat"
    """How language as condition, can be "concat", "adapter", "sparse_actions"""
    sparse_steps: int=4
    """Sparse action steps, no larger than 4"""
    prompt: str=""
    """Prompt for language condition"""
    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj,
                 use_language=False):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.use_language = use_language

        from diffusion_policy.utils import load_demo_dataset_with_lan
        trajectories = load_demo_dataset_with_lan(data_path, num_traj=num_traj, concat=False)

        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                    device
                )  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                device
            )
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())

        # Process language descriptions if available
        if self.use_language and "language" in trajectories:
            self.trajectory_language = trajectories["language"]
        else:
            # Create empty language placeholders if not available in dataset
            self.trajectory_language = [args.prompt] * len(trajectories["actions"])

        # Pre-process the actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(
                device=device
            )
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
                "delta_pos" in args.control_mode
                or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print(
                "Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                         max(0, start): start + self.obs_horizon
                         ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start): end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
                obs_seq["state"].shape[0] == self.obs_horizon
                and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
            "language": self.trajectory_language[traj_idx]
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args, device="cuda:0"):
        super().__init__()
        self.device = device
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.sparse_steps = args.sparse_steps if hasattr(args, 'sparse_steps') else 4

        # Check if language encoder should be used
        self.use_language = hasattr(args, 'lan_encoder') and args.lan_encoder != ""
        self.language_condition_type = args.language_condition_type if hasattr(args,
                                                                               'language_condition_type') else "concat"

        assert (
                len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
                env.single_action_space.low == -1
        ).all()

        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        language_feature_dim = 512

        self.vision_model = None
        self.processor = None

        # Visual encoder
        if args.visual_encoder == 'plain_conv':
            from diffusion_policy.encoders.plain_conv import PlainConv
            self.visual_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
            ).to(device)
        elif args.visual_encoder == 'clip':
            from diffusion_policy.encoders.clip import CLIPEncoder
            self.visual_encoder = CLIPEncoder(
                out_dim=visual_feature_dim
            ).to(device)
        elif args.visual_encoder == 'dinov2':
            from diffusion_policy.encoders.dinov2 import DINOv2Encoder
            self.visual_encoder = DINOv2Encoder(
                out_dim=visual_feature_dim
            ).to(device)
        elif args.visual_encoder == 'resnet':
            from diffusion_policy.encoders.resnet import ResNetEncoder
            self.visual_encoder = ResNetEncoder(
                in_channels=total_visual_channels, out_dim=visual_feature_dim
            ).to(device)
        elif args.visual_encoder == 'siglip':
            from diffusion_policy.encoders.siglip import SigLIP2Encoder
            self.visual_encoder = SigLIP2Encoder(
                out_dim=visual_feature_dim
            ).to(device)
        elif args.visual_encoder == "shared":
            from transformers import SiglipVisionModel, AutoProcessor
            if self.vision_model is  None:
                self.vision_model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
            from diffusion_policy.encoders.vis_encoder import VisionEncoder
            self.visual_encoder = VisionEncoder(
                vision_model=self.vision_model,
                processor=self.processor,
                out_dim=visual_feature_dim,
                encoder_type=getattr(args, 'visual_encoder_type', "encoder_only"),
                device=self.device
            )

        # Language encoder setup (new)
        if self.use_language:
            from transformers import SiglipVisionModel, AutoProcessor
            from diffusion_policy.encoders.lan_encoder import LanguageEncoder
            if self.vision_model is None:
                self.vision_model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
            self.language_encoder = LanguageEncoder(
                vision_model=self.vision_model,
                processor=self.processor,
                encoder_type=args.lan_encoder,
                output_dim=language_feature_dim,
                device=self.device
            )

            # Vision-language adapter
            if self.language_condition_type == "adapter":
                self.vision_adapter = nn.Sequential(
                    nn.Linear(visual_feature_dim, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(),
                    nn.Linear(512, language_feature_dim)
                ).to(self.device)

            # Setup for sparse prediction
            if self.language_condition_type == "sparse_actions":
                # Sparse action predictor (for predicting waypoints)
                self.sparse_action_predictor = ConditionalUnet1D(
                    input_dim=self.act_dim,
                    global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim) + language_feature_dim,
                    diffusion_step_embed_dim=args.diffusion_step_embed_dim,
                    down_dims=[args.unet_dims[0] // 2] + args.unet_dims[:-1],  # Smaller network
                    n_groups=args.n_groups,
                ).to(self.device)

            # Adjust condition dimension based on language condition type
            if self.language_condition_type == "concat":
                global_cond_dim = self.obs_horizon * (visual_feature_dim + obs_state_dim) + language_feature_dim
            elif self.language_condition_type == "adapter":
                global_cond_dim = self.obs_horizon * (language_feature_dim + obs_state_dim) + language_feature_dim
            elif self.language_condition_type == "sparse_actions":
                global_cond_dim = self.obs_horizon * (
                            visual_feature_dim + obs_state_dim) + self.sparse_steps * self.act_dim + language_feature_dim
            else:
                # Vision-only case
                global_cond_dim = self.obs_horizon * (visual_feature_dim + obs_state_dim)
        else:
            # Original condition dimension (vision only)
            global_cond_dim = self.obs_horizon * (visual_feature_dim + obs_state_dim)

        # Main diffusion model
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        ).to(self.device)

        # Diffusion setup
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Sparse diffusion setup (if using language)
        if self.use_language and self.language_condition_type == "sparse_actions":
            self.sparse_noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            # rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            rgb = obs_seq["rgb"].float()  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            # depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            depth = obs_seq["depth"].float()  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k

        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)

        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)

        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)

        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)

        return visual_feature, feature.flatten(start_dim=1)  # Return both visual features and combined features

    def encode_language(self, text_instructions, obs_seq=None, eval_mode=False):
        """Encode language instructions with optional visual features"""
        if not self.use_language:
            return None
        if self.include_rgb:
            # rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            rgb = obs_seq["rgb"].float()  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            # depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            depth = obs_seq["depth"].float()  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k

        return self.language_encoder(text_instructions, obs_horizon=self.obs_horizon, image=img_seq)  # (B, language_feature_dim), (B, visual_feature_dim)

    def predict_sparse_actions(self, obs_seq, language_feature, eval_mode=False):
        """Predict sparse waypoint actions using language and vision"""
        B = obs_seq["state"].shape[0]

        # Get visual features
        visual_features, obs_cond = self.encode_obs(obs_seq, eval_mode=eval_mode)

        # Combine with language features
        combined_cond = torch.cat([obs_cond, language_feature], dim=1)

        # Initialize from noise
        noisy_sparse_actions = torch.randn(
            (B, self.sparse_steps, self.act_dim), device=obs_seq["state"].device
        )

        # Run diffusion process
        if eval_mode:
            with torch.no_grad():
                for k in self.sparse_noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = self.sparse_action_predictor(
                        sample=noisy_sparse_actions,
                        timestep=k,
                        global_cond=combined_cond,
                    )

                    # Inverse diffusion step
                    noisy_sparse_actions = self.sparse_noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=noisy_sparse_actions,
                    ).prev_sample
        else:
            # For training, just return the noisy actions (will be denoised during loss computation)
            pass

        return noisy_sparse_actions

    def compute_loss(self, obs_seq, action_seq, text_instructions=None):
        B = obs_seq["state"].shape[0]

        # Get visual features and observation conditioning
        visual_features, obs_cond = self.encode_obs(obs_seq, eval_mode=False)

        # Handle language condition if available
        if self.use_language and text_instructions is not None:
            language_feature = self.encode_language(text_instructions, obs_seq=obs_seq, eval_mode=False)

            if self.language_condition_type == "concat":
                # Simple concatenation of features
                obs_cond = torch.cat([obs_cond, language_feature], dim=1)

            elif self.language_condition_type == "adapter":
                # Adapt visual features to language space
                batch_size = visual_features.shape[0]
                adapted_visual = self.vision_adapter(visual_features.reshape(-1, visual_features.shape[-1]))
                adapted_visual = adapted_visual.reshape(batch_size, self.obs_horizon, -1)

                # Create new observation condition with adapted features
                adapted_obs_cond = torch.cat(
                    (adapted_visual, obs_seq["state"]), dim=-1
                ).flatten(start_dim=1)

                # Combine with language feature
                # print(adapted_obs_cond.shape, language_feature.shape)
                obs_cond = torch.cat([adapted_obs_cond, language_feature], dim=1)

            elif self.language_condition_type == "sparse_actions":
                # First, compute loss for sparse action prediction
                sparse_loss = 0.0

                # Sample sparse waypoints from the full action sequence
                sparse_indices = torch.linspace(
                    self.obs_horizon,
                    self.pred_horizon - 1,
                    steps=self.sparse_steps,
                    dtype=torch.long,
                    device=action_seq.device
                )
                sparse_target_actions = action_seq[:, sparse_indices]

                # Sample noise for sparse actions
                sparse_noise = torch.randn_like(sparse_target_actions)
                sparse_timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action_seq.device
                ).long()

                # Add noise to sparse actions
                noisy_sparse_actions = self.sparse_noise_scheduler.add_noise(
                    sparse_target_actions, sparse_noise, sparse_timesteps
                )

                # Get condition for sparse action prediction
                sparse_cond = torch.cat([obs_cond, language_feature], dim=1)

                # Predict noise for sparse actions
                sparse_noise_pred = self.sparse_action_predictor(
                    noisy_sparse_actions, sparse_timesteps, global_cond=sparse_cond
                )

                # Compute sparse prediction loss
                sparse_loss = F.mse_loss(sparse_noise_pred, sparse_noise)

                # Now use predicted sparse actions as condition for dense action prediction
                # For training, we use teacher forcing (ground truth sparse actions)
                obs_cond = torch.cat([sparse_cond, sparse_target_actions.flatten(start_dim=1)], dim=1)

        # Sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=action_seq.device)

        # Sample diffusion timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action_seq.device
        ).long()

        # Add noise to actions
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # Predict noise
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        # Compute main diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # Return combined loss if using sparse actions, otherwise just diffusion loss
        if self.use_language and self.language_condition_type == "sparse_actions":
            return diffusion_loss + sparse_loss
        else:
            return diffusion_loss

    def get_action(self, obs_seq, text_instructions=None):
        B = obs_seq["state"].shape[0]

        with torch.no_grad():
            # Prepare image sequences
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            # Get visual features and observation conditioning
            visual_features, obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            # Handle language condition if available
            if self.use_language and text_instructions is not None:
                text_input = [text_instructions[0]] * len(obs_seq['rgb'])
                language_feature = self.encode_language(text_input, obs_seq=obs_seq, eval_mode=True)

                if self.language_condition_type == "concat":
                    # Simple concatenation of features
                    obs_cond = torch.cat([obs_cond, language_feature], dim=1)

                elif self.language_condition_type == "adapter":
                    # Adapt visual features to language space
                    batch_size = visual_features.shape[0]
                    adapted_visual = self.vision_adapter(visual_features.reshape(-1, visual_features.shape[-1]))
                    adapted_visual = adapted_visual.reshape(batch_size, self.obs_horizon, -1)

                    # Create new observation condition with adapted features
                    adapted_obs_cond = torch.cat(
                        (adapted_visual, obs_seq["state"]), dim=-1
                    ).flatten(start_dim=1)

                    # Combine with language feature
                    obs_cond = torch.cat([adapted_obs_cond, language_feature], dim=1)

                elif self.language_condition_type == "sparse_actions":
                    # First predict sparse waypoint actions using language + vision
                    sparse_cond = torch.cat([obs_cond, language_feature], dim=1)

                    # Initialize sparse actions from noise
                    noisy_sparse_actions = torch.randn(
                        (B, self.sparse_steps, self.act_dim), device=obs_seq["state"].device
                    )

                    # Run sparse diffusion process
                    for k in self.sparse_noise_scheduler.timesteps:
                        # Predict noise
                        noise_pred = self.sparse_action_predictor(
                            sample=noisy_sparse_actions,
                            timestep=k,
                            global_cond=sparse_cond,
                        )

                        # Inverse diffusion step
                        noisy_sparse_actions = self.sparse_noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=noisy_sparse_actions,
                        ).prev_sample

                    # Use predicted sparse actions as condition for dense action prediction
                    obs_cond = torch.cat([sparse_cond, noisy_sparse_actions.flatten(start_dim=1)], dim=1)

            # Initialize dense action sequence from noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            # Run diffusion process
            for k in self.noise_scheduler.timesteps:
                # Predict noise
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x626 and 882x128)
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # Inverse diffusion step
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

            # Only take act_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )

def load_ckpt(ckpt_name, tag):
    checkpoint = torch.load(f"runs/{ckpt_name}/checkpoints/{tag}.pt")
    agent.load_state_dict(checkpoint["agent"])
    ema_agent.load_state_dict(checkpoint["ema_agent"])
    return agent, ema_agent


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack=args.shader),
        # mode="eval"
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    agent = Agent(envs, args, device=device)
    ema_agent = Agent(envs, args)
    # Load checkpoint
    agent, ema_agent = load_ckpt(ckpt_name, "best_eval_success_at_end")
    ema = EMAModel(parameters=agent.parameters(), power=0.75)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best():
        ema.copy_to(ema_agent.parameters())
        eval_metrics = evaluate(
            args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
        )

        print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
        for k in eval_metrics.keys():
            eval_metrics[k] = np.mean(eval_metrics[k])
            writer.add_scalar(f"eval/{k}", eval_metrics[k], 1)
            print(f"{k}: {eval_metrics[k]:.4f}")

    evaluate_and_save_best()

    envs.close()
    writer.close()
