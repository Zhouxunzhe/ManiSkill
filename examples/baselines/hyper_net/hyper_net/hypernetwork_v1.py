import os, sys
from typing import List

import einops
import torch
from torch import nn
import torch.nn.functional as F


class TargetNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    # in_dims and out_dims are not necessarily the same with actual model input/output dims.
    # but prev output dim must be the same with next input dim.
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
        '''
        convert the weight dict of the submodules to the weight dict of the target net.
        '''
        weight_dict = {}
        for sub_name, wd in zip(self.get_submodule_names(), weight_dicts):
            for k, v in wd.items():
                weight_dict[sub_name + '.' + k] = v

        return weight_dict


# 将MLP替换为Conv1D网络
class Conv1dBlock(nn.Module):
    '''
    A lightweight block consisting of:
    Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        # Ensure n_groups is compatible with out_channels
        n_groups = min(n_groups, out_channels)

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Conv1DNet(nn.Module):
    '''
    A lightweight 1D convolution network to replace MLP for parameter efficiency
    Uses GroupNorm and Mish activation for better performance
    Handles extremely large parameter counts by using dimensionality reduction
    '''

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, kernel_size=3, n_groups=8, max_projection_dim=1024):
        super().__init__()

        # Ensure input dimension is correctly processed
        in_dim = in_dim if isinstance(in_dim, int) else in_dim[0]

        # Check if dimensions are extremely large and apply dimension reduction if needed
        self.use_projection = in_dim > max_projection_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        if self.use_projection:
            # Use a more efficient approach for very large dimensions
            # Instead of a full Linear layer which would require in_dim×in_dim parameters,
            # we use a low-rank approximation with a bottleneck dimension
            projection_dim = min(max_projection_dim, in_dim // 1000 + 1)
            self.projection_dim = projection_dim

            # Create efficient projection matrices
            # Input projection: in_dim -> projection_dim
            self.proj_in_weight = nn.Parameter(torch.randn(projection_dim) * 0.02)
            self.proj_in_bias = nn.Parameter(torch.zeros(projection_dim))

            # Output projection: projection_dim -> out_dim
            # Correct shape for matrix multiplication: projection_dim x out_dim
            self.proj_out_weight = nn.Parameter(torch.randn(projection_dim, out_dim) * 0.02)
            self.proj_out_bias = nn.Parameter(torch.zeros(out_dim))
        else:
            # Use regular Linear layers for reasonable dimensions
            self.proj_in = nn.Linear(in_dim, in_dim)
            self.proj_out = nn.Linear(in_dim, out_dim)

        if num_layers == 1:
            # For single layer, use a simple Conv1d without the block
            self.layers = nn.ModuleList([nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)])
        else:
            self.layers = nn.ModuleList()

            # First layer: 1 -> hidden_dim
            self.layers.append(Conv1dBlock(1, hidden_dim, kernel_size, n_groups))

            # Middle layers: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                self.layers.append(Conv1dBlock(hidden_dim, hidden_dim, kernel_size, n_groups))

            # Last layer: hidden_dim -> 1 (no activation)
            self.layers.append(nn.Conv1d(hidden_dim, 1, kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        # Apply input projection based on dimension handling approach
        if self.use_projection:
            # For extremely large dimensions, use efficient projection
            # Instead of matmul, we compute a mean weighted by proj_in_weight
            batch_size = x.shape[0]

            # Use chunking to process large vectors in smaller pieces
            chunk_size = 10000  # Adjust based on memory constraints
            num_chunks = (self.in_dim + chunk_size - 1) // chunk_size

            # Initialize result tensor for accumulated projection
            result = torch.zeros(batch_size, self.projection_dim, device=x.device)

            # Process in chunks to avoid OOM
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, self.in_dim)

                # Process this chunk and add its contribution
                # We use a statistical approach (mean) to reduce dimensionality
                chunk = x[:, start_idx:end_idx]
                chunk_mean = chunk.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1]

                # Apply weights to the mean (broadcasting)
                for j in range(self.projection_dim):
                    result[:, j] += chunk_mean[:, 0] * self.proj_in_weight[j]

            # Add bias
            x = result + self.proj_in_bias
        else:
            # Regular projection for normal-sized dimensions
            x = self.proj_in(x)

        # Convert to convolution format (B, C, L)
        x = x.unsqueeze(1)

        # Apply convolution layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Convert back to original shape
        x = x.squeeze(1)

        # Apply output projection based on dimension handling approach
        if self.use_projection:
            # Efficient output projection for large dimensions
            # Perform the matrix multiplication correctly:
            # [batch_size, projection_dim] @ [projection_dim, out_dim]
            x = torch.matmul(x, self.proj_out_weight) + self.proj_out_bias
        else:
            # Regular output projection
            x = self.proj_out(x)

        return x


class ModuleEncoder(nn.Module):
    def __init__(self, target_net, weight_dim, hidden_dim, num_layers):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.encoder = Conv1DNet(self.param_cnt, weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, weight_dict):
        weight_vec = torch.cat([weight_dict[k].reshape(-1, s.numel()) for k, s in self.name_shape_dict.items()], dim=-1)
        if weight_vec.ndim == 1:
            weight_vec = weight_vec[None]
        return self.encoder(weight_vec)


class ModuleDecoder(nn.Module):
    def __init__(self, target_net, weight_dim, hidden_dim, num_layers):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.decoder = Conv1DNet(weight_dim, self.param_cnt, hidden_dim=hidden_dim, num_layers=num_layers)
        self.chunks = [v.numel() for v in self.name_shape_dict.values()]

    def forward(self, weight_vec):
        decoded_weights = self.decoder(weight_vec)
        weight_chunks = torch.split(
            decoded_weights, self.chunks, dim=-1
        )
        weight_dict = {
            k: chunk.reshape(-1, *v)
            for chunk, (k, v) in zip(weight_chunks, self.name_shape_dict.items())
        }
        return weight_dict


class OptSubBlock(nn.Module):
    '''
    a OptSubBlock is responsible for estimating the derivative of output w.r.t. the inputs, and the target net parameters.
    Now includes ftask in forward and backward passes to prevent task information loss.
    '''

    def __init__(self, ftask_dim, in_dim, out_dim, weight_dim,
                 hidden_dim, num_layers,
                 dl_din_way='direct',
                 dl_dw_way='slice',
                 **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_dim = weight_dim  # Store weight_dim as an instance variable
        self.hidden_dim = hidden_dim

        # 修改网络接受ftask作为额外输入
        self.forward_net = Conv1DNet(in_dim + weight_dim + ftask_dim, out_dim, hidden_dim=hidden_dim,
                                     num_layers=num_layers)

        self.dout_din = Conv1DNet(in_dim + weight_dim + ftask_dim, out_dim * in_dim, hidden_dim=hidden_dim,
                                  num_layers=num_layers)
        self.dout_dw = Conv1DNet(in_dim + weight_dim + ftask_dim, out_dim * weight_dim, hidden_dim=hidden_dim,
                                 num_layers=num_layers)

        self.dl_din_way = dl_din_way
        self.dl_dw_way = dl_dw_way

        if dl_din_way == 'slice':
            self.mm_mlp_in = Conv1DNet(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_din_way == 'full':
            self.mm_mlp_in = Conv1DNet(out_dim * (in_dim + 1), in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            assert dl_din_way == 'direct'

        if dl_dw_way == 'slice':
            self.mm_mlp_w = Conv1DNet(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_dw_way == 'full':
            self.mm_mlp_w = Conv1DNet(out_dim * (weight_dim + 1), weight_dim, hidden_dim=hidden_dim,
                                      num_layers=num_layers)
        else:
            assert dl_dw_way == 'direct'

    def pseudo_forward(self, z_in, weight_emb, ftask):
        # 添加ftask作为输入
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])
        if z_in.shape[0] != ftask.shape[0]:
            ftask = einops.repeat(ftask, '1 i -> n i', n=z_in.shape[0])

        out = self.forward_net(torch.cat([z_in, weight_emb, ftask], dim=-1))
        return out

    def pseudo_backward(self, z_in, weight_emb, dl_dout, ftask):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])
        if z_in.shape[0] != ftask.shape[0]:
            ftask = einops.repeat(ftask, '1 i -> n i', n=z_in.shape[0])

        dout_din = einops.rearrange(
            self.dout_din(torch.cat([z_in, weight_emb, ftask], dim=-1)),
            'n (o i) -> n o i', i=self.in_dim
        )
        dout_dw = einops.rearrange(
            self.dout_dw(torch.cat([z_in, weight_emb, ftask], dim=-1)),
            'n (o i) -> n o i', i=self.weight_dim  # Now self.weight_dim is properly defined
        )

        if self.dl_din_way == 'direct':
            dl_din = einops.einsum(dl_dout, dout_din, 'n o, n o i -> n i')
        elif self.dl_din_way == 'slice':
            dl_din = self.mm_mlp_in(
                torch.cat([einops.repeat(dl_dout, 'n o -> n i o', i=dout_din.shape[1]), dout_din], dim=2),
            )[..., 0]
        elif self.dl_din_way == 'full':
            dl_din = self.mm_mlp_in(
                torch.cat([dl_dout, dout_din.flatten(1)], dim=1)
            )

        if self.dl_dw_way == 'direct':
            dl_dw = einops.einsum(dl_dout, dout_dw, 'n o, n o i -> n i')
        elif self.dl_dw_way == 'slice':
            # Get the dimensions
            batch_size = dl_dout.shape[0]
            out_dim = dl_dout.shape[1]
            weight_dim = self.weight_dim
            products = torch.zeros(batch_size, 2 * out_dim, device=dl_dout.device)
            products[:, :out_dim] = dl_dout
            for i in range(out_dim):
                products[:, out_dim + i] = torch.mean(dout_dw[:, i, :], dim=1)
            dl_dw = self.mm_mlp_w(products)
        elif self.dl_dw_way == 'full':
            dl_dw = self.mm_mlp_w(
                torch.cat([dl_dout, dout_dw.flatten(1)], dim=1)
            )

        return dl_din, dl_dw


# 用一个共享的OptBlock替代原来的多个OptBlock
class SharedOptBlock(nn.Module):
    '''
    A shared OptBlock that can be applied iteratively to save parameters.
    Also passes ftask through to all sub-components.
    '''

    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 *args, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dims = target_net.get_in_dims()
        self.out_dims = target_net.get_out_dims()
        self.weight_dim = weight_dim

        # 创建共享的OptSubBlocks
        self.opt_sub_blocks = nn.ModuleList([
            OptSubBlock(ftask_dim, in_dim, out_dim, weight_dim,
                        deriv_hidden_dim, driv_num_layers,
                        *args, **kwargs)
            for in_dim, out_dim in zip(self.in_dims, self.out_dims)
        ])

        self.forward_in = Conv1DNet(ftask_dim, self.in_dims[0], hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
        self.dloss_dout = Conv1DNet(ftask_dim + self.out_dims[-1], self.out_dims[-1], hidden_dim=deriv_hidden_dim,
                                    num_layers=driv_num_layers)

    def forward(self, ftask, weight_embs, encoders: List[ModuleEncoder], decoders: List[ModuleDecoder]):
        z_ins = [self.forward_in(ftask)]
        for weight_emb, opt_sub_block in zip(weight_embs, self.opt_sub_blocks):
            # 传递ftask到所有子块
            z_ins.append(opt_sub_block.pseudo_forward(z_ins[-1], weight_emb, ftask))

        dl_douts = [self.dloss_dout(torch.cat([ftask, z_ins[-1]], dim=-1))]

        dw_dicts = []
        for opt_sub_block, z_in, weight_emb, decoder in reversed(
                list(zip(self.opt_sub_blocks, z_ins, weight_embs, decoders))):
            # 传递ftask到所有子块
            dl_dout, dl_dw = opt_sub_block.pseudo_backward(z_in, weight_emb, dl_douts[-1], ftask)
            dl_douts.append(dl_dout)
            dw_dicts.append(dl_dw)

        dw_dicts = list(reversed(dw_dicts))

        return dw_dicts


class ParamLN(nn.Module):
    def __init__(self, weight_dim):
        super().__init__()
        self.ln = nn.LayerNorm(weight_dim)

    def forward(self, weight_dict):
        return self.ln(weight_dict)


class SharedParamHypernet(nn.Module):
    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 codec_hidden_dim, codec_num_layers,
                 num_layers, *args, **kwargs):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.num_layers = num_layers

        self.encoders = nn.ModuleList([
            ModuleEncoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])
        self.decoders = nn.ModuleList([
            ModuleDecoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])

        # 使用一个共享的OptBlock，而不是多个
        self.shared_opt_block = SharedOptBlock(
            target_net, ftask_dim, weight_dim,
            deriv_hidden_dim, driv_num_layers,
            *args, **kwargs
        )

        # 学习更新步长
        self.dynamic_lrs = nn.Parameter(torch.zeros(num_layers).fill_(-1e-2))
        self.layer_norms = nn.ModuleList(
            [ParamLN(weight_dim) for submodule in self.target_net.get_submodules()]
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"HyperNet parameters: {n_params / 1e6:.2f}M")

    def forward_blocks(self, ftask):
        weight_dicts = list(
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        )
        weight_embs = list(
            encoder(weight_dict)
            for encoder, weight_dict in zip(self.encoders, weight_dicts)
        )

        weight_embs = [weight_emb.repeat(ftask.shape[0], 1) for weight_emb in weight_embs]
        final_weight_dicts = []

        # 循环使用同一个共享OptBlock多次
        for i in range(self.num_layers):
            weight_upd_embs = self.shared_opt_block(ftask, weight_embs, self.encoders, self.decoders)
            weight_embs = [v + self.dynamic_lrs[i] * v_upd for v, v_upd in zip(weight_embs, weight_upd_embs)]

            weight_dicts = [decoder(w) for (decoder, w) in zip(self.decoders, weight_embs)]
            weight_dict = self.target_net.merge_submodule_weights(weight_dicts)
            final_weight_dicts.append(weight_dict)

        return final_weight_dicts

    def forward(self, ftask, inputs, early_sup=False):
        final_weight_dicts = self.forward_blocks(ftask)
        if early_sup:
            return torch.stack([
                torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(self.target_net, generated_weight, inputs)
                for generated_weight in final_weight_dicts])
        else:
            return torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(self.target_net, final_weight_dicts[-1],
                                                                                inputs)


class Toy(TargetNet):
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
                x = fc(x)  # Remove the activation function from the last layer
            else:
                x = self.activation(fc(x))
        return x

    def get_in_dims(self):
        return [self.dim_input] + [self.dim_hidden] * (self.num_layers - 1)

    def get_out_dims(self):
        return [self.dim_hidden] * (self.num_layers - 1) + [self.dim_output]

    def get_submodules(self):
        return self.fcs


if __name__ == "__main__":
    model = Hypernet(Toy(64, out_dim=7, hidden_dim=16, num_layers=2), ftask_dim=10, weight_dim=128, deriv_hidden_dim=32,
                     driv_num_layers=2,
                     codec_hidden_dim=64, codec_num_layers=2, num_layers=8)
    input = torch.randn(128, 64)
    f_input = torch.randn(128, 10)
    output = model(f_input, input)