ALGO_NAME = "BC_Hypernet_Diffusion"

import os
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
import torchvision.models as models
from tqdm import tqdm
import tyro
import h5py
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from hypernetwork import Hypernet, TargetNet, MLP
from hyper_net.evaluate import evaluate
from hyper_net.make_env import make_eval_envs
from hyper_net.utils import (IterationBasedBatchSampler, build_state_obs_extractor,
                                    convert_obs, worker_init_fn)


# 保留你的神经网络结构
class VideoEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)

    def forward(self, video):
        batch, T, H, W, C = video.shape
        video = video.permute(0, 1, 4, 2, 3)
        video = video.view(batch * T, C, H, W)
        features = self.resnet(video)
        features = features.view(batch, T, 512)
        features = features.mean(dim=1)
        task_feature = self.fc(features)
        return task_feature


class RobotPolicy(nn.Module):
    def __init__(self, mlp_in_dim, mlp_out_dim, mlp_hidden_dim, mlp_num_layers, in_channels):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.mlp = MLP(mlp_in_dim, mlp_out_dim, mlp_hidden_dim, mlp_num_layers)

    def forward(self, obs, robot_state, mlp_params=None):
        features = self.resnet(obs)
        features = features.view(features.size(0), -1)
        mlp_input = torch.cat([features, robot_state], dim=1)
        if mlp_params is not None:
            x = mlp_input
            batch_size = x.shape[0]
            for i, fc in enumerate(self.mlp.fcs):
                weight = mlp_params[f'fcs.{i}.weight']
                bias = mlp_params[f'fcs.{i}.bias']
                x = torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1) + bias
                if i < len(self.mlp.fcs) - 1:
                    x = self.mlp.activation(x)
            output = x
        else:
            output = self.mlp(mlp_input)
        return output


class MLPTargetNet(TargetNet):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(x)

    def get_in_dims(self):
        return [self.mlp.fcs[0].in_features]

    def get_out_dims(self):
        return [self.mlp.fcs[-1].out_features]

    def get_submodules(self):
        return [self.mlp]

    def get_submodule_names(self):
        return ['mlp']

    def merge_submodule_weights(self, weight_dicts):
        return weight_dicts[0]


# 配置参数
@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "PegInsertionSide-v1"
    demo_path: str = ("")
    video_path: str= "processed_data"
    num_demos: Optional[int] = None
    total_iters: int = 1000000
    batch_size: int = 256

    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    n_groups: int = 8

    obs_mode: str = "rgb+depth"
    max_episode_steps: Optional[int] = 200
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    num_dataload_workers: int = 0
    control_mode: str = "pd_joint_delta_pos"
    shader: str = "default"

    # Hypernet 参数
    batch_size = 128
    T = 10
    obs_shape = (3, 128, 128)
    robot_state_dim = 127
    action_dim = 8
    ftask_dim = 64
    mlp_in_dim = 512 + robot_state_dim
    mlp_out_dim = action_dim
    mlp_hidden_dim = 256
    mlp_num_layers = 2
    weight_dim = 128
    deriv_hidden_dim = 32
    driv_num_layers = 2
    codec_hidden_dim = 64
    codec_num_layers = 2
    num_layers = 8
    val_num_per_task = 5


# 数据集类
class HypernetDataset(Dataset):
    def __init__(self, data_path, videos, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj=None):
        self.device = device
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.videos = videos
        obs_process_fn = obs_process_fn
        obs_space = obs_space

        # Load real robot demonstration data using Diffusion Policy's utility
        from hyper_net.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process observations to align with the environment's observation space
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = self._reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float32)).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(device)
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(device)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())

        # Pre-process actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

        # Define horizons (assuming these are passed via args or another mechanism)
        self.obs_horizon = args.obs_horizon  # e.g., 2
        self.pred_horizon = args.pred_horizon  # e.g., 16

        # Pre-compute all possible (traj_idx, start, end) tuples
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L
            pad_before = self.obs_horizon - 1
            pad_after = self.pred_horizon - self.obs_horizon
            self.slices += [
                (traj_idx, start, start + self.pred_horizon)
                for start in range(-pad_before, L - self.pred_horizon + pad_after)
            ]
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories
        self.task_names = list(self.videos.keys())

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        # 获取观察序列（保持原始逻辑，但只使用最后一个）
        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        action = act_seq[-1]

        task_name = 'human_pickplace_redcube_plate_the_red_cube_the_plate_with_the_right_hand'
        video_idx = random.randint(0, len(self.videos[task_name])-1)
        video = self.videos[task_name][video_idx].to(self.device, dtype=torch.float32) / 255.0

        return {
            "video": video,
            "observations": obs_seq,
            "actions": action
        }

    def __len__(self):
        return len(self.slices)

    def _reorder_keys(self, obs_dict, obs_space):
        # Helper function to reorder dictionary keys based on obs_space
        return {k: obs_dict[k] for k in obs_space.keys()}


# Agent 类
class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args, device):
        super().__init__()
        self.device = device

        # 从环境获取动作维度
        self.act_dim = env.single_action_space.shape[0]

        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        # 初始化你的网络组件
        self.video_encoder = VideoEncoder(output_dim=args.ftask_dim).to(device)
        self.policy = RobotPolicy(args.mlp_in_dim, args.mlp_out_dim, args.mlp_hidden_dim, args.mlp_num_layers, in_channels=total_visual_channels).to(device)
        self.target_net = MLPTargetNet(self.policy.mlp)
        self.hypernet = Hypernet(self.target_net, args.ftask_dim, args.weight_dim, args.deriv_hidden_dim, args.driv_num_layers,
                            args.codec_hidden_dim, args.codec_num_layers, args.num_layers).to(device)

    def compute_loss(self, data_batch):
        videos = data_batch["video"].to(self.device)
        if self.include_rgb:
            rgb = data_batch["observations"]["rgb"][:, -1].to(self.device).float() / 255.0
            obs_seq = rgb
        if self.include_depth:
            depth = data_batch["observations"]["depth"][:, -1].to(self.device).float() / 1024.0
            obs_seq = depth
        if self.include_rgb and self.include_depth:
            obs_seq = torch.cat([rgb, depth], dim=1)
        robot_states = data_batch["observations"]["state"][:, -1].to(self.device)
        actions = data_batch["actions"].to(self.device)  # 现在是单个动作

        ftask = self.video_encoder(videos)
        final_weight_dicts = self.hypernet.forward_blocks(ftask)
        generated_weight = final_weight_dicts[-1]

        pred_actions = self.policy(obs_seq, robot_states, mlp_params=generated_weight)
        loss = F.mse_loss(pred_actions, actions)
        return loss

    def get_action(self, obs, val_videos):
        task_name = 'human_pickplace_redcube_plate_the_red_cube_the_plate_with_the_right_hand'
        video_idx = random.randint(0, len(val_videos[task_name])-1)
        video = val_videos[task_name][video_idx].unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0

        with torch.no_grad():
            if self.include_rgb:
                rgb = (obs["rgb"][:, -1].to(self.device).float() / 255.0).permute(0, 3, 1, 2)
                obs_seq = rgb
            if self.include_depth:
                depth = (obs["depth"][:, -1].to(self.device).float() / 1024.0).permute(0, 3, 1, 2)
                obs_seq = depth
            if self.include_rgb and self.include_depth:
                obs_seq = torch.cat([rgb, depth], dim=1)
            robot_state = obs["state"][:, -1].to(self.device)

            # 生成权重
            ftask = self.video_encoder(video)
            final_weight_dicts = self.hypernet.forward_blocks(ftask)
            generated_weight = final_weight_dicts[-1]

            actions = self.policy(obs_seq, robot_state, mlp_params=generated_weight)
            return actions


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict()},
        f"runs/{run_name}/checkpoints/{tag}.pt"
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
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

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

    # 初始化数据集

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
                videos[task_name].append(torch.tensor(video, dtype=torch.uint8).to(device))

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
    agent = Agent(envs, args, device=device).to(device)

    # 设置优化器（仅优化 video_encoder 和 hypernet）
    optimizer = optim.AdamW(
        list(agent.video_encoder.parameters()) + list(agent.hypernet.parameters()),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6
    )

    # 设置学习率调度器（可选）
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters
    )

    # 设置 EMA（可选，与原始代码无此功能，可移除）
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args, device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def evaluate_and_save_best(iteration, val_videos):
        if iteration % args.eval_freq == 0 and iteration != 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, val_videos, args.sim_backend
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

        # EMA 更新（可选）
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

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