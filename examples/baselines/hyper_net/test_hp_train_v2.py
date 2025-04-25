import torch
import torch.nn as nn
import torchvision.models as models
from hyper_net.hypernetwork import Hypernet, TargetNet, MLP
from torch.utils.data import Dataset, DataLoader
import random
import os
import h5py
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 视频编码器
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

# Robot Policy
class RobotPolicy(nn.Module):
    def __init__(self, mlp_in_dim, mlp_out_dim, mlp_hidden_dim, mlp_num_layers):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.mlp = MLP(mlp_in_dim, mlp_out_dim, mlp_hidden_dim, mlp_num_layers)

    def forward(self, obs, robot_state, mlp_params=None):
        features = self.resnet(obs)
        features = features.view(features.size(0), -1)
        mlp_input = torch.cat([features, robot_state], dim=1)
        if mlp_params is not None:
            x = mlp_input
            batch_size = x.shape[0]  # Fixed: changed x.shape(0) to x.shape[0]
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

# TargetNet 定义（仅包含 MLP）
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

# 加载视频数据
def load_video_data(processed_dir="processed_data", videos_per_task=50):
    videos = {}
    hdf5_files = [f for f in os.listdir(processed_dir) if f.endswith(".h5")]
    num_tasks = len(hdf5_files)
    print(f"Detected {num_tasks} tasks from HDF5 files.")

    for hdf5_file in hdf5_files:
        task_name = hdf5_file.replace(".h5", "")
        hdf5_path = os.path.join(processed_dir, hdf5_file)
        with h5py.File(hdf5_path, 'r') as h5f:
            num_videos = len(h5f)
            if num_videos < videos_per_task:
                print(f"Warning: Task '{task_name}' has only {num_videos} videos, expected {videos_per_task}.")
            videos[task_name] = []
            for i in range(min(num_videos, videos_per_task)):
                group = h5f[str(i)]
                video = group["obs"][:]
                videos[task_name].append(torch.tensor(video, dtype=torch.uint8))
            print(f"Task '{task_name}' video shape: {videos[task_name][0].shape}, loaded {len(videos[task_name])} videos.")

    return videos, num_tasks

# 生成合成 demo 数据
def generate_demo_data(num_tasks, task_names, demos_per_task=100):
    obs_shape = (3, 128, 128)
    robot_state_dim = 8
    action_dim = 8

    demos = {}
    for task_name in task_names:
        demos[task_name] = []
        for _ in range(demos_per_task):
            length = random.randint(100, 200)
            obs = torch.randn(length, *obs_shape)
            robot_state = torch.randn(length, robot_state_dim)
            action = torch.randn(length, action_dim)
            demos[task_name].append((obs, robot_state, action))
        print(f"Task '{task_name}' demo shape: obs {demos[task_name][0][0].shape}, "
              f"robot_state {demos[task_name][0][1].shape}, action {demos[task_name][0][2].shape}")

    return demos

# 自定义 Dataset（每隔 K 个 timestep 采样）
class TaskDataset(Dataset):
    def __init__(self, videos, demos, K=2):
        self.videos = videos
        self.demos = demos
        self.task_names = list(videos.keys())
        self.videos_per_task = 50
        self.demos_per_task = 100
        self.K = K

        self.samples = []
        for task_name in self.task_names:
            for video_idx in range(len(self.videos[task_name])):
                for demo_idx in range(self.demos_per_task):
                    obs, robot_state, action = self.demos[task_name][demo_idx]
                    for t in range(0, len(obs), self.K):
                        self.samples.append((task_name, video_idx, demo_idx, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task_name, video_idx, demo_idx, timestep = self.samples[idx]
        video = self.videos[task_name][video_idx]
        obs, robot_state, action = self.demos[task_name][demo_idx]
        return video, obs[timestep], robot_state[timestep], action[timestep]

# 训练函数（带进度条）
def train_model(video_encoder, hypernet, policy, dataset, num_epochs=10, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam(
        list(video_encoder.parameters()) + list(hypernet.parameters()),
        lr=1e-3
    )

    # # 如果也需要更新policy的resnet参数
    # optimizer = torch.optim.Adam(
    #     list(video_encoder.parameters()) + list(hypernet.parameters()) + list(policy.resnet.parameters()),
    #     lr=1e-3
    # )

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(dataloader)

        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_videos, batch_obs, batch_robot_states, batch_actions in dataloader:
                batch_videos = batch_videos.to(device, dtype=torch.float32) / 255.0
                batch_obs = batch_obs.to(device)
                batch_robot_states = batch_robot_states.to(device)
                batch_actions = batch_actions.to(device)

                optimizer.zero_grad()
                ftask = video_encoder(batch_videos)
                final_weight_dicts = hypernet.forward_blocks(ftask)
                generated_weight = final_weight_dicts[-1]
                pred_actions = policy(batch_obs, batch_robot_states, mlp_params=generated_weight)

                loss = criterion(pred_actions, batch_actions)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

# 主程序
if __name__ == "__main__":
    batch_size = 128
    T = 10
    obs_shape = (3, 128, 128)
    robot_state_dim = 8
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

    print("Loading video data from HDF5 files...")
    videos, num_tasks = load_video_data()
    task_names = list(videos.keys())

    print("Generating synthetic demo data...")
    demos = generate_demo_data(num_tasks, task_names)

    dataset = TaskDataset(videos, demos, K=2)
    print(f"Dataset created with {len(dataset)} samples.")

    video_encoder = VideoEncoder(output_dim=ftask_dim).to(device)
    policy = RobotPolicy(mlp_in_dim, mlp_out_dim, mlp_hidden_dim, mlp_num_layers).to(device)
    target_net = MLPTargetNet(policy.mlp)
    hypernet = Hypernet(target_net, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers,
                        codec_hidden_dim, codec_num_layers, num_layers).to(device)

    print("Starting training...")
    train_model(video_encoder, hypernet, policy, dataset, num_epochs=10, batch_size=batch_size)
    print("Training complete.")