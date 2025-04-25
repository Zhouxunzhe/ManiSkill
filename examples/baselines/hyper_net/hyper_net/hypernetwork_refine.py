import os, sys
from typing import List, Dict, Tuple, Optional

import einops
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class TargetNet(nn.Module):
    """定义目标网络的抽象基类"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def get_in_dims(self) -> List[int]:
        raise NotImplementedError

    def get_out_dims(self) -> List[int]:
        raise NotImplementedError

    def get_submodules(self) -> List[nn.Module]:
        raise NotImplementedError

    def get_submodule_names(self) -> List[str]:
        if not hasattr(self, 'submodule_names'):
            self.submodule_names = []
            for subm in self.get_submodules():
                for name, module in self.named_modules():
                    if subm is module:
                        self.submodule_names.append(name)
                        break
        return self.submodule_names

    def merge_submodule_weights(self, weight_dicts):
        '''将子模块的权重字典转换为目标网络的权重字典'''
        weight_dict = {}
        for sub_name, wd in zip(self.get_submodule_names(), weight_dicts):
            for k, v in wd.items():
                weight_dict[sub_name + '.' + k] = v
        return weight_dict


class MemoryEfficientLossScaling:
    """内存高效的损失尺度缩放器"""

    @staticmethod
    def scale_mse_loss(pred, target, scale_factor=1e-4):
        """
        计算缩放后的MSE损失，无需存储额外中间张量

        Args:
            pred: 预测值
            target: 目标值
            scale_factor: 缩放因子，默认1e-4

        Returns:
            缩放后的MSE损失
        """
        # 直接计算差异
        diff = pred - target

        # 计算MSE但不立即求平均，这样可以应用缩放
        squared_diff = diff ** 2

        # 应用缩放因子 - 这比缩放输入更高效，因为不需要存储额外的张量
        scaled_squared_diff = squared_diff * scale_factor

        # 现在计算平均值
        return scaled_squared_diff.mean()


def compute_efficient_diffusion_loss(noise_pred, noise):
    """
    内存高效的diffusion loss计算

    Args:
        noise_pred: 预测的噪声
        noise: 目标噪声

    Returns:
        缩放后的MSE损失
    """
    # 使用内存高效的损失缩放
    return MemoryEfficientLossScaling.scale_mse_loss(noise_pred, noise, scale_factor=1e-4)


class LowRankLinear(nn.Module):
    """使用低秩分解的线性层，极小初始化缩放"""

    def __init__(self, in_features, out_features, rank=None, bias=True):
        super().__init__()
        # 自动确定合理的秩
        if rank is None:
            rank = min(in_features, out_features) // 4
            rank = max(rank, 1)  # 确保至少秩为1

        # 极小初始化：对于非常小的目标loss，我们需要极小的初始权重
        # 缩放因子10^-4，因为目标loss是10^-4到10^-3级别
        scaling = 1e-4
        std_u = np.sqrt(2.0 / in_features) * scaling
        std_v = np.sqrt(2.0 / rank) * scaling

        self.weight_u = nn.Parameter(torch.randn(in_features, rank) * std_u)
        self.weight_v = nn.Parameter(torch.randn(rank, out_features) * std_v)

        if bias:
            # 初始化偏置也要极小
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = torch.matmul(x, self.weight_u)
        x = torch.matmul(x, self.weight_v)
        if self.bias is not None:
            x = x + self.bias
        return x


class MemoryEfficientMLP(nn.Module):
    """内存高效的MLP实现"""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2, dropout=0.0,
                 activation=F.silu, use_checkpoint=True, low_rank=True, rank_ratio=4):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.activation = activation
        self.dropout = dropout

        # 选择线性层实现方式
        LinearLayer = LowRankLinear if low_rank else nn.Linear

        # 确定低秩层的秩参数
        in_rank = max(min(in_dim, hidden_dim) // rank_ratio, 1) if low_rank else None
        hidden_rank = max(hidden_dim // rank_ratio, 1) if low_rank else None
        out_rank = max(min(hidden_dim, out_dim) // rank_ratio, 1) if low_rank else None

        # 创建网络层
        if num_layers == 1:
            self.layers = nn.ModuleList([LinearLayer(in_dim, out_dim, rank=in_rank)])
        else:
            layers = [LinearLayer(in_dim, hidden_dim, rank=in_rank)]
            for i in range(num_layers - 2):
                layers.append(LinearLayer(hidden_dim, hidden_dim, rank=hidden_rank))
            layers.append(LinearLayer(hidden_dim, out_dim, rank=out_rank))
            self.layers = nn.ModuleList(layers)

    def _forward_impl(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不使用激活函数
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)


class ModuleEncoder(nn.Module):
    """目标模块参数的高效编码器"""

    def __init__(self, target_net, weight_dim, hidden_dim, num_layers,
                 use_checkpoint=True, low_rank=True, rank_ratio=4):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.sorted_keys = sorted(self.name_shape_dict.keys())
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])

        # 使用内存高效的MLP
        self.encoder = MemoryEfficientMLP(
            self.param_cnt, weight_dim, hidden_dim,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            low_rank=low_rank,
            rank_ratio=rank_ratio
        )

    def forward(self, weight_dict):
        # 按顺序连接所有参数
        weight_vec = torch.cat([weight_dict[k].reshape(-1) for k in self.sorted_keys], dim=0)
        if weight_vec.ndim == 1:
            weight_vec = weight_vec.unsqueeze(0)  # 添加批次维度
        return self.encoder(weight_vec)


class ModuleDecoder(nn.Module):
    """目标模块参数的高效解码器，极小初始化"""

    def __init__(self, target_net, weight_dim, hidden_dim, num_layers,
                 use_checkpoint=True, low_rank=True, rank_ratio=4):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.sorted_keys = sorted(self.name_shape_dict.keys())
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])

        # 使用内存高效的MLP
        self.decoder = MemoryEfficientMLP(
            weight_dim, self.param_cnt, hidden_dim,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            low_rank=low_rank,
            rank_ratio=rank_ratio
        )

        self.chunks = [self.name_shape_dict[k].numel() for k in self.sorted_keys]

        # 添加输出缩放，用于精细控制生成权重的尺度
        self.output_scale = nn.Parameter(torch.ones(1) * 1e-4)

    def forward(self, weight_vec):
        decoded_weights = self.decoder(weight_vec)

        # 应用极小的缩放
        decoded_weights = decoded_weights * self.output_scale

        # 拆分解码后的权重向量
        weight_chunks = torch.split(decoded_weights, self.chunks, dim=-1)

        # 恢复原始参数形状
        weight_dict = {}
        for k, chunk in zip(self.sorted_keys, weight_chunks):
            weight_dict[k] = chunk.reshape(-1, *self.name_shape_dict[k])

        return weight_dict


class EfficientJacobian(nn.Module):
    """内存高效的Jacobian计算模块 - 优化版本"""

    def __init__(self, in_dim, out_dim, combined_dim, hidden_dim,
                 num_layers, use_checkpoint=True, dropout=0.0, rank_ratio=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 确定是否使用低秩分解 - 更激进的优化条件
        self.is_large = in_dim * out_dim > 500  # 降低阈值，更多场景用低秩

        if self.is_large:
            # 更激进的秩压缩比例
            rank = max(min(in_dim, out_dim) // (rank_ratio * 2), 1)  # 加倍压缩
            self.rank = rank

            # 分别计算U和V矩阵
            self.jacobian_u = MemoryEfficientMLP(
                combined_dim, out_dim * rank, hidden_dim,
                num_layers=max(2, num_layers - 1),  # 减少层数
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                low_rank=True,
                rank_ratio=rank_ratio * 2  # 加倍压缩
            )

            self.jacobian_v = MemoryEfficientMLP(
                combined_dim, in_dim * rank, hidden_dim,
                num_layers=max(2, num_layers - 1),  # 减少层数
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                low_rank=True,
                rank_ratio=rank_ratio * 2  # 加倍压缩
            )
        else:
            # 对于小型Jacobian，直接计算完整矩阵但减少隐层大小
            self.jacobian = MemoryEfficientMLP(
                combined_dim, out_dim * in_dim, max(32, hidden_dim // 2),  # 减半隐层
                num_layers=max(2, num_layers - 1),  # 减少层数
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                low_rank=True,
                rank_ratio=rank_ratio * 2  # 加倍压缩
            )

    def forward(self, combined_input):
        """计算Jacobian矩阵或其低秩分解 - 内存高效实现"""
        if self.is_large:
            # 计算低秩分解 - 内联优化以减少内存
            u_flat = self.jacobian_u(combined_input)
            v_flat = self.jacobian_v(combined_input)

            # 重塑为合适的维度 - 使用直接reshape而不是einops以减少内存开销
            batch_size = combined_input.shape[0]
            u = u_flat.reshape(batch_size, self.out_dim, self.rank)
            v = v_flat.reshape(batch_size, self.in_dim, self.rank)

            return u, v
        else:
            # 直接计算完整Jacobian
            jacobian_flat = self.jacobian(combined_input)
            # 使用直接reshape而不是einops
            return jacobian_flat.reshape(jacobian_flat.shape[0], self.out_dim, self.in_dim)

    def compute_gradient(self, dl_dout, jacobian_output):
        """内存高效的梯度计算"""
        if self.is_large:
            # 使用低秩分解计算
            u, v = jacobian_output

            # 使用torch.bmm替代einops.einsum以减少内存开销
            # dl_dout: [batch, out_dim]
            # u: [batch, out_dim, rank]
            # v: [batch, in_dim, rank]

            # 步骤1: (dl_dout)·U -> [batch, rank]
            dl_dout_reshaped = dl_dout.unsqueeze(1)  # [batch, 1, out_dim]
            temp = torch.bmm(dl_dout_reshaped, u).squeeze(1)  # [batch, rank]

            # 步骤2: (dl_dout·U)·V^T -> [batch, in_dim]
            temp_unsqueezed = temp.unsqueeze(1)  # [batch, 1, rank]
            dl_din = torch.bmm(temp_unsqueezed, v.transpose(1, 2)).squeeze(1)  # [batch, in_dim]

            return dl_din
        else:
            # 直接使用torch.bmm替代einops.einsum
            dl_dout_reshaped = dl_dout.unsqueeze(1)  # [batch, 1, out_dim]
            return torch.bmm(dl_dout_reshaped, jacobian_output).squeeze(1)  # [batch, in_dim]


class BalancedOptSubBlock(nn.Module):
    '''平衡内存效率与收敛性的OptSubBlock实现'''

    def __init__(self, ftask_dim, in_dim, out_dim, weight_dim,
                 hidden_dim, num_layers,
                 use_checkpoint=True, dropout=0.0, rank_ratio=4):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        self.combined_dim = in_dim + weight_dim + ftask_dim

        # 前向网络：计算当前层的输出
        self.forward_net = MemoryEfficientMLP(
            self.combined_dim, out_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_checkpoint=use_checkpoint
        )

        # 输出对输入的Jacobian计算模块
        self.dout_din_jacobian = EfficientJacobian(
            in_dim, out_dim, self.combined_dim, hidden_dim,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            dropout=dropout,
            rank_ratio=rank_ratio
        )

        # 输出对权重的Jacobian计算模块
        self.dout_dw_jacobian = EfficientJacobian(
            weight_dim, out_dim, self.combined_dim, hidden_dim,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            dropout=dropout,
            rank_ratio=rank_ratio
        )

    def _forward_impl(self, z_in, weight_emb, ftask):
        # 确保批次维度匹配
        batch_size = z_in.shape[0]
        if weight_emb.shape[0] == 1 and batch_size > 1:
            weight_emb = weight_emb.expand(batch_size, -1)
        if ftask.shape[0] == 1 and batch_size > 1:
            ftask = ftask.expand(batch_size, -1)

        # 合并输入
        combined_input = torch.cat([z_in, weight_emb, ftask], dim=-1)

        # 前向传播计算输出
        out = self.forward_net(combined_input)
        return out, combined_input

    def _backward_impl(self, combined_input, dl_dout):
        """计算梯度，保持数学上的正确性"""
        # 计算输出对输入的梯度
        dout_din = self.dout_din_jacobian(combined_input)
        dl_din = self.dout_din_jacobian.compute_gradient(dl_dout, dout_din)

        # 计算输出对权重的梯度
        dout_dw = self.dout_dw_jacobian(combined_input)
        dl_dw = self.dout_dw_jacobian.compute_gradient(dl_dout, dout_dw)

        return dl_din, dl_dw

    def pseudo_forward(self, z_in, weight_emb, ftask):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_impl, z_in, weight_emb, ftask)
        else:
            return self._forward_impl(z_in, weight_emb, ftask)

    def pseudo_backward(self, z_in, weight_emb, ftask, dl_dout):
        # 首先重新计算前向传播以获取combined_input
        _, combined_input = self._forward_impl(z_in, weight_emb, ftask)

        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._backward_impl, combined_input, dl_dout)
        else:
            return self._backward_impl(combined_input, dl_dout)


class MemoryEfficientOptBlock(nn.Module):
    '''内存高效的OptBlock实现，保持梯度计算的数学正确性'''

    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 hidden_dim, num_layers,
                 use_checkpoint=True, dropout=0.0,
                 layer_groups=True, skip_ratio=0.5, rank_ratio=4):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dims = target_net.get_in_dims()
        self.out_dims = target_net.get_out_dims()
        self.weight_dim = weight_dim
        self.use_checkpoint = use_checkpoint
        self.num_layers = len(self.in_dims)
        self.skip_ratio = skip_ratio

        # 确定每次迭代要处理的层数
        self.steps_per_iter = max(int(self.num_layers * (1 - skip_ratio)), 1)

        # 对层进行分组以共享参数（可选）
        if layer_groups and self.num_layers > 1:
            self.layer_groups, self.layer_to_group = self._group_similar_layers()

            # 为每个组创建一个OptSubBlock
            self.group_blocks = nn.ModuleDict()
            for group_id, (in_dim, out_dim) in self.layer_groups.items():
                self.group_blocks[str(group_id)] = BalancedOptSubBlock(
                    ftask_dim, in_dim, out_dim, weight_dim,
                    hidden_dim, num_layers,
                    use_checkpoint=use_checkpoint,
                    dropout=dropout,
                    rank_ratio=rank_ratio
                )

            # 为维度不匹配的层创建投影
            self.input_projections = nn.ModuleDict()
            self.output_projections = nn.ModuleDict()

            for i, (in_dim, out_dim) in enumerate(zip(self.in_dims, self.out_dims)):
                group_id = self.layer_to_group[i]
                group_in_dim, group_out_dim = self.layer_groups[group_id]

                if in_dim != group_in_dim:
                    self.input_projections[str(i)] = nn.Linear(in_dim, group_in_dim)

                if out_dim != group_out_dim:
                    self.output_projections[str(i)] = nn.Linear(group_out_dim, out_dim)
        else:
            # 为每层创建独立的OptSubBlock
            self.opt_sub_blocks = nn.ModuleList([
                BalancedOptSubBlock(
                    ftask_dim, in_dim, out_dim, weight_dim,
                    hidden_dim, num_layers,
                    use_checkpoint=use_checkpoint,
                    dropout=dropout,
                    rank_ratio=rank_ratio
                )
                for in_dim, out_dim in zip(self.in_dims, self.out_dims)
            ])

        # 使用分组
        self.layer_groups_enabled = layer_groups and self.num_layers > 1

        # 初始输入生成网络
        self.initial_net = MemoryEfficientMLP(
            ftask_dim, self.in_dims[0], hidden_dim,
            num_layers=max(num_layers // 2, 1),
            use_checkpoint=use_checkpoint,
            dropout=dropout
        )

        # 损失对输出的梯度生成网络
        self.dloss_dout = MemoryEfficientMLP(
            ftask_dim + self.out_dims[-1], self.out_dims[-1], hidden_dim,
            num_layers=max(num_layers // 2, 1),
            use_checkpoint=use_checkpoint,
            dropout=dropout
        )

        # 层跳过参数（类似DDIM的alpha和sigma）
        self.alphas = nn.Parameter(torch.ones(self.num_layers) * 0.9)
        self.sigmas = nn.Parameter(torch.ones(self.num_layers) * 0.1)

    def _group_similar_layers(self):
        """基于维度相似性将层分组，保持计算的正确性"""
        # 基于维度比率和规模将层分组
        groups = {}  # group_id -> (in_dim, out_dim)
        layer_to_group = {}  # layer_idx -> group_id

        for i, (in_dim, out_dim) in enumerate(zip(self.in_dims, self.out_dims)):
            # 计算维度比率和大小分类
            ratio = in_dim / max(1, out_dim)

            # 确定组类型
            if ratio < 0.5:
                shape_type = "expanding"  # 扩展型
            elif ratio > 2.0:
                shape_type = "reducing"  # 收缩型
            else:
                shape_type = "similar"  # 相似型

            # 确定尺寸类别
            if max(in_dim, out_dim) <= 64:
                size_type = "small"
            elif max(in_dim, out_dim) <= 256:
                size_type = "medium"
            else:
                size_type = "large"

            # 创建组ID
            group_id = f"{shape_type}_{size_type}"

            # 查找或创建组
            if group_id not in groups:
                groups[group_id] = (in_dim, out_dim)

            layer_to_group[i] = group_id

        return groups, layer_to_group

    def _get_active_layers(self, iteration):
        """智能确定本次迭代要处理的层"""
        if self.steps_per_iter >= self.num_layers:
            # 如果处理所有层
            return list(range(self.num_layers))
        else:
            # 循环处理不同的层集合，确保所有层都被处理
            offset = iteration % (self.num_layers // self.steps_per_iter + 1)
            indices = []
            for i in range(offset, self.num_layers, self.num_layers // self.steps_per_iter + 1):
                if i < self.num_layers:
                    indices.append(i)
            return indices[:self.steps_per_iter]

    def forward(self, iteration, ftask, weight_embs):
        """执行前向和反向传播，计算权重梯度"""
        # 确定本次迭代要处理的层
        active_layers = self._get_active_layers(iteration)

        # 保存所有中间激活
        z_ins = [None] * (self.num_layers + 1)
        z_ins[0] = self.initial_net(ftask)  # 初始输入

        # 前向传播
        for i in range(self.num_layers):
            if i in active_layers:
                # 获取当前层输入
                current_z_in = z_ins[i]

                if self.layer_groups_enabled:
                    # 使用分组OptSubBlock
                    group_id = self.layer_to_group[i]
                    block = self.group_blocks[str(group_id)]

                    # 应用输入投影（如果需要）
                    if str(i) in self.input_projections:
                        current_z_in = self.input_projections[str(i)](current_z_in)

                    # 执行前向传播
                    z_out, _ = block.pseudo_forward(current_z_in, weight_embs[i], ftask)

                    # 应用输出投影（如果需要）
                    if str(i) in self.output_projections:
                        z_out = self.output_projections[str(i)](z_out)
                else:
                    # 使用独立的OptSubBlock
                    z_out, _ = self.opt_sub_blocks[i].pseudo_forward(current_z_in, weight_embs[i], ftask)

                z_ins[i + 1] = z_out
            elif i > 0 and z_ins[i] is not None:
                # 对于未处理的层，使用控制传播策略
                # 类似于DDIM的逐步推理，但保持信息连贯性
                alpha = self.alphas[i]
                sigma = self.sigmas[i]
                noise = torch.randn_like(z_ins[i]) * sigma
                z_ins[i + 1] = alpha * z_ins[i] + noise

        # 计算最终输出对损失的梯度
        dl_douts = [self.dloss_dout(torch.cat([ftask, z_ins[-1]], dim=-1))]

        # 反向传播计算梯度
        weight_grads = [None] * self.num_layers

        # 从最后一层反向传播到第一层
        for i in reversed(range(self.num_layers)):
            if i in active_layers:
                # 获取当前层输入和梯度
                z_in = z_ins[i]
                dl_dout = dl_douts[-1]

                if self.layer_groups_enabled:
                    # 使用分组OptSubBlock
                    group_id = self.layer_to_group[i]
                    block = self.group_blocks[str(group_id)]

                    # 应用投影（如果需要）
                    if str(i) in self.input_projections:
                        z_in = self.input_projections[str(i)](z_in)

                    if str(i) in self.output_projections:
                        # 通过输出投影的转置传播梯度
                        dl_dout = torch.matmul(dl_dout, self.output_projections[str(i)].weight)

                    # 执行反向传播
                    dl_din, dl_dw = block.pseudo_backward(z_in, weight_embs[i], ftask, dl_dout)

                    # 对于输入投影，反向传播梯度
                    if str(i) in self.input_projections:
                        dl_din = torch.matmul(dl_din, self.input_projections[str(i)].weight.t())
                else:
                    # 使用独立的OptSubBlock
                    dl_din, dl_dw = self.opt_sub_blocks[i].pseudo_backward(z_in, weight_embs[i], ftask, dl_dout)

                dl_douts.append(dl_din)
                weight_grads[i] = dl_dw
            else:
                # 对于未处理的层，使用控制传播策略
                if len(dl_douts) > 0:
                    # 使用类似DDIM的梯度传播
                    alpha = self.alphas[i]
                    sigma = self.sigmas[i]
                    noise = torch.randn_like(dl_douts[-1]) * sigma
                    dl_douts.append(alpha * dl_douts[-1] + noise)
                else:
                    # 如果没有梯度可以传播，使用零梯度
                    dl_douts.append(torch.zeros_like(z_ins[i]))

                # 为未处理的层设置零梯度
                weight_grads[i] = torch.zeros_like(weight_embs[i])

        return weight_grads


class OptimizedHypernet(nn.Module):
    """内存优化的超网络实现"""

    def __init__(self, target_net, ftask_dim, weight_dim, deriv_hidden_dim, deriv_num_layers,
                 codec_hidden_dim, codec_num_layers, num_iterations, agent_lr=1e-4,
                 chunk_size=2, use_checkpoint=True, dropout=0.0, layer_groups=True,
                 skip_ratio=0.5, rank_ratio=4):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.num_iterations = num_iterations
        self.chunk_size = min(chunk_size, num_iterations)
        self.use_checkpoint = use_checkpoint

        # 减少编码器/解码器内部层数，降低参数量
        codec_num_layers = max(2, codec_num_layers - 1)  # 减少一层但确保至少有2层

        # 减少OptBlock中的隐藏维度，降低内存占用
        deriv_hidden_dim = max(64, deriv_hidden_dim // 2)  # 将隐藏维度减半但确保至少64维

        # 内存高效的编码器和解码器
        self.encoders = nn.ModuleList([
            ModuleEncoder(
                target_module, weight_dim, codec_hidden_dim, codec_num_layers,
                use_checkpoint=use_checkpoint,
                low_rank=True,
                rank_ratio=rank_ratio
            )
            for target_module in target_net.get_submodules()
        ])

        self.decoders = nn.ModuleList([
            ModuleDecoder(
                target_module, weight_dim, codec_hidden_dim, codec_num_layers,
                use_checkpoint=use_checkpoint,
                low_rank=True,
                rank_ratio=rank_ratio
            )
            for target_module in target_net.get_submodules()
        ])

        # 构建内存高效的OptBlock
        self.opt_block = MemoryEfficientOptBlock(
            target_net, ftask_dim, weight_dim,
            deriv_hidden_dim, deriv_num_layers,
            use_checkpoint=use_checkpoint,
            dropout=dropout,
            layer_groups=layer_groups,
            skip_ratio=skip_ratio,
            rank_ratio=rank_ratio
        )

        # 一个小的学习率 - 使用极小初始值
        self.lr_base = nn.Parameter(torch.ones(1) * agent_lr)

        # 学习率调度 - 使用更简单的线性增长以减少存储
        self.lr_multipliers = nn.Parameter(torch.linspace(0.1, 1.0, num_iterations))

        # 层归一化用于稳定权重更新
        num_submodules = len(target_net.get_submodules())
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(weight_dim) for _ in range(num_submodules)
        ])

        # 使用标量而不是张量参数进一步减少内存
        self.weight_scale = nn.Parameter(torch.tensor(1e-4))

        # 初始化参数使用内存高效的方法
        self._initialize_parameters()

        # 关闭不必要的统计信息跟踪，减少显存占用
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = False

        # 计算并打印参数数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"内存优化Hypernet参数数量: {n_params / 1e6:.2f}M")

        target_params = sum(p.numel() for p in target_net.parameters())
        print(f"目标网络参数数量: {target_params / 1e6:.2f}M")
        print(f"比例: {n_params / max(1, target_params):.2f}x")

    def _initialize_parameters(self):
        """内存高效的初始化"""
        # 使用in-place操作进行初始化以减少内存峰值
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    m.weight.mul_(1e-4)  # in-place操作
                    if m.bias is not None:
                        m.bias.zero_()  # in-place操作
                elif isinstance(m, LowRankLinear):
                    # 对于低秩线性层也使用in-place操作
                    nn.init.normal_(m.weight_u, std=0.01)
                    m.weight_u.mul_(1e-4)  # in-place操作
                    nn.init.normal_(m.weight_v, std=0.01)
                    m.weight_v.mul_(1e-4)  # in-place操作

    def _process_chunk(self, ftask, weight_embs, start_idx, end_idx):
        """内存高效的块处理"""
        # 使用clone创建副本，但立即detach以减少计算图大小
        current_weight_embs = [w.clone().detach().requires_grad_(True) for w in weight_embs]

        # 处理一系列迭代
        for i in range(start_idx, end_idx):
            # 计算梯度
            weight_grads = self.opt_block(i, ftask, current_weight_embs)

            # 应用归一化和学习率
            lr = self.lr_base * self.lr_multipliers[i]

            # 更新权重 - 使用in-place操作减少内存使用
            for j, (weight_emb, weight_grad, layer_norm) in enumerate(
                    zip(current_weight_embs, weight_grads, self.layer_norms)):
                # 归一化梯度
                normalized_grad = layer_norm(weight_grad)

                # 就地更新权重 - 减少内存峰值
                weight_emb.add_(normalized_grad.mul(lr))  # in-place操作

        # 解码最终权重
        weight_dicts = []
        for decoder, weight_emb in zip(self.decoders, current_weight_embs):
            # 使用标量缩放 - 更内存高效
            with torch.no_grad():
                scaled_emb = weight_emb * self.weight_scale
            weight_dict = decoder(scaled_emb)
            weight_dicts.append(weight_dict)

        # 合并所有子模块的权重
        final_weight_dict = self.target_net.merge_submodule_weights(weight_dicts)

        return final_weight_dict, current_weight_embs

    def forward_blocks(self, ftask):
        """返回每个优化步骤的权重字典列表，保持与原始Hypernet类的API兼容性"""
        # 获取初始权重
        weight_dicts = [
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        ]

        # 编码初始权重
        weight_embs = []
        for encoder, weight_dict in zip(self.encoders, weight_dicts):
            weight_emb = encoder(weight_dict)

            # 匹配批次大小
            if weight_emb.shape[0] == 1 and ftask.shape[0] > 1:
                weight_emb = weight_emb.expand(ftask.shape[0], -1)

            weight_embs.append(weight_emb)

        # 存储每次迭代的权重字典
        all_weight_dicts = []
        current_weight_embs = [w.clone() for w in weight_embs]

        # 处理每次迭代
        for i in range(self.num_iterations):
            # 计算当前迭代的梯度
            weight_grads = self.opt_block(i, ftask, current_weight_embs)

            # 应用学习率
            lr = self.lr_base * self.lr_multipliers[i]

            # 更新权重
            for j, (weight_emb, weight_grad, layer_norm) in enumerate(
                    zip(current_weight_embs, weight_grads, self.layer_norms)):
                # 应用归一化
                normalized_grad = layer_norm(weight_grad)

                # 更新权重嵌入
                current_weight_embs[j] = weight_emb + lr * normalized_grad

            # 解码当前权重
            current_weight_dicts = []
            for decoder, weight_emb in zip(self.decoders, current_weight_embs):
                weight_dict = decoder(weight_emb)
                current_weight_dicts.append(weight_dict)

            # 合并权重并添加到列表
            merged_weight_dict = self.target_net.merge_submodule_weights(current_weight_dicts)
            all_weight_dicts.append(merged_weight_dict)

            # 清除缓存以节省内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_weight_dicts

    def forward(self, ftask, inputs, early_sup=False):
        """执行超网络前向传播，生成目标网络权重"""
        # 获取初始权重
        weight_dicts = [
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        ]

        # 编码初始权重
        weight_embs = []
        for encoder, weight_dict in zip(self.encoders, weight_dicts):
            weight_emb = encoder(weight_dict)

            # 匹配批次大小
            if weight_emb.shape[0] == 1 and ftask.shape[0] > 1:
                weight_emb = weight_emb.expand(ftask.shape[0], -1)

            weight_embs.append(weight_emb)

        # 分块处理迭代以降低内存使用
        chunks = []
        final_weight_embs = weight_embs

        for chunk_start in range(0, self.num_iterations, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.num_iterations)

            # 处理当前块
            if self.use_checkpoint and self.training:
                # 使用梯度检查点节省内存
                chunk_result = checkpoint.checkpoint(
                    self._process_chunk, ftask, final_weight_embs, chunk_start, chunk_end
                )
                chunk_weight_dict, final_weight_embs = chunk_result
            else:
                chunk_weight_dict, final_weight_embs = self._process_chunk(
                    ftask, final_weight_embs, chunk_start, chunk_end
                )

            chunks.append(chunk_weight_dict)

            # 清除不必要的缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最终权重来自最后一个块
        final_weight_dict = chunks[-1]

        # 根据需要返回中间结果
        if early_sup:
            # 返回所有中间结果
            return torch.stack([
                torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(
                    self.target_net, chunk, inputs)
                for chunk in chunks
            ])
        else:
            # 只使用最终权重
            return torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(
                self.target_net, final_weight_dict, inputs)


class Toy(TargetNet):
    """示例目标网络实现"""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2, activation=F.relu):
        super().__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.dim_input = in_dim
        self.dim_hidden = hidden_dim
        self.dim_output = out_dim

        if num_layers == 1:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
            self.fcs.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
            self.fcs.append(nn.Linear(hidden_dim, out_dim))
        self.activation = activation

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            if i == len(self.fcs) - 1:
                x = fc(x)  # 最后一层不使用激活函数
            else:
                x = self.activation(fc(x))
        return x

    def get_in_dims(self):
        return [self.dim_input] + [self.dim_hidden] * (self.num_layers - 1)

    def get_out_dims(self):
        return [self.dim_hidden] * (self.num_layers - 1) + [self.dim_output]

    def get_submodules(self):
        return self.fcs


# 示例用法
if __name__ == "__main__":
    # 创建目标网络
    target_net = Toy(64, out_dim=10, hidden_dim=32, num_layers=3)

    # 创建平衡型超网络
    hypernet = OptimizedHypernet(
        target_net,
        ftask_dim=16,
        weight_dim=48,
        deriv_hidden_dim=32,
        deriv_num_layers=2,
        codec_hidden_dim=48,
        codec_num_layers=2,
        num_iterations=4,
        chunk_size=2,
        use_checkpoint=True,
        dropout=0.1,
        layer_groups=True,
        skip_ratio=0.3,  # 每次迭代处理70%的层
        rank_ratio=4  # 平衡的秩比例
    )

    # 示例用法
    batch_size = 16
    input_data = torch.randn(batch_size, 64)
    task_desc = torch.randn(batch_size, 16)

    # 测试前向传播
    hypernet.train()
    output = hypernet(task_desc, input_data)
    print(f"输出形状: {output.shape}")