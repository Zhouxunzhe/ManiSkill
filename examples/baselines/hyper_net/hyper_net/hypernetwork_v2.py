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


# Replace MLP with Conv1D for better parameter efficiency
class Conv1DMLP(nn.Module):
    '''
    A Conv1D-based MLP with a variable number of layers and hidden dimensions.
    More parameter efficient than fully-connected layers.
    '''

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()
        if num_layers == 1:
            self.convs = nn.ModuleList([nn.Conv1d(in_dim, out_dim, kernel_size=1)])
        else:
            self.convs = nn.ModuleList([nn.Conv1d(in_dim, hidden_dim, kernel_size=1)])
            self.convs.extend([nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(num_layers - 2)])
            self.convs.append(nn.Conv1d(hidden_dim, out_dim, kernel_size=1))
        self.activation = activation
        self.norm_layers = nn.ModuleList([nn.GroupNorm(min(32, hidden_dim), hidden_dim)
                                          if i < num_layers - 1 else None
                                          for i in range(num_layers)])

    def forward(self, x):
        # Reshape for Conv1D: [batch, features] -> [batch, features, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = self.norm_layers[i](x)
                x = self.activation(x)

        # Reshape back: [batch, features, 1] -> [batch, features]
        return x.squeeze(-1)


class ModuleEncoder(nn.Module):
    def __init__(self, target_net, weight_dim, hidden_dim, num_layers):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.encoder = Conv1DMLP(self.param_cnt, weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)

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
        self.decoder = Conv1DMLP(weight_dim, self.param_cnt, hidden_dim=hidden_dim, num_layers=num_layers)
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
    Now including ftask to prevent task information loss
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
        self.hidden_dim = hidden_dim

        # Updated to include ftask in inputs
        self.forward_net = Conv1DMLP(in_dim + weight_dim + ftask_dim, out_dim,
                                     hidden_dim=hidden_dim, num_layers=num_layers)

        # Updated to include ftask in inputs
        self.dout_din = Conv1DMLP(in_dim + weight_dim + ftask_dim, out_dim * in_dim,
                                  hidden_dim=hidden_dim, num_layers=num_layers)
        self.dout_dw = Conv1DMLP(in_dim + weight_dim + ftask_dim, out_dim * weight_dim,
                                 hidden_dim=hidden_dim, num_layers=num_layers)

        self.dl_din_way = dl_din_way
        self.dl_dw_way = dl_dw_way

        if dl_din_way == 'slice':
            self.mm_mlp_in = Conv1DMLP(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_din_way == 'full':
            self.mm_mlp_in = Conv1DMLP(out_dim * (in_dim + 1), in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            assert dl_din_way == 'direct'

        if dl_dw_way == 'slice':
            self.mm_mlp_w = Conv1DMLP(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_dw_way == 'full':
            self.mm_mlp_w = Conv1DMLP(out_dim * (weight_dim + 1), weight_dim, hidden_dim=hidden_dim,
                                      num_layers=num_layers)
        else:
            assert dl_dw_way == 'direct'

    def pseudo_forward(self, z_in, weight_emb, ftask):
        # Added ftask as input
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])
        if ftask.shape[0] != z_in.shape[0]:
            ftask = einops.repeat(ftask, '1 i -> n i', n=z_in.shape[0])

        out = self.forward_net(torch.cat([z_in, weight_emb, ftask], dim=-1))
        return out

    def pseudo_backward(self, z_in, weight_emb, dl_dout, ftask):
        # Added ftask as input
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])
        if ftask.shape[0] != z_in.shape[0]:
            ftask = einops.repeat(ftask, '1 i -> n i', n=z_in.shape[0])

        dout_din = einops.rearrange(
            self.dout_din(torch.cat([z_in, weight_emb, ftask], dim=-1)),
            'n (o i) -> n o i', i=self.in_dim
        )
        dout_dw = einops.rearrange(
            self.dout_dw(torch.cat([z_in, weight_emb, ftask], dim=-1)),
            'n (o i) -> n o i', i=self.weight_dim
        )

        if self.dl_din_way == 'direct':
            dl_din = einops.einsum(dl_dout, dout_din, 'n o, n o i -> n i')
        elif self.dl_din_way == 'slice':
            dl_din = self.mm_mlp_in(
                torch.cat([einops.repeat(dl_dout, 'n o -> n o i', i=dout_din.shape[2]), dout_din], dim=1)
            )[..., 0]
        elif self.dl_din_way == 'full':
            dl_din = self.mm_mlp_in(
                torch.cat([dl_dout, dout_din.flatten(1)], dim=1)
            )

        if self.dl_dw_way == 'direct':
            dl_dw = einops.einsum(dl_dout, dout_dw, 'n o, n o i -> n i')
        elif self.dl_dw_way == 'slice':
            dl_dw = self.mm_mlp_w(
                torch.cat([einops.repeat(dl_dout, 'n o -> n o i', i=dout_dw.shape[2]), dout_dw], dim=1)
            )[..., 0]
        elif self.dl_dw_way == 'full':
            dl_dw = self.mm_mlp_w(
                torch.cat([dl_dout, dout_dw.flatten(1)], dim=1)
            )

        return dl_din, dl_dw


class SharedOptBlock(nn.Module):
    '''
    A weight-shared version of OptBlock that uses the same parameters for multiple iterations
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

        # Create just one set of OptSubBlocks
        self.opt_sub_blocks = nn.ModuleList([
            OptSubBlock(ftask_dim, in_dim, out_dim, weight_dim,
                        deriv_hidden_dim, driv_num_layers,
                        *args, **kwargs)
            for in_dim, out_dim in zip(self.in_dims, self.out_dims)
        ])

        self.forward_in = Conv1DMLP(ftask_dim, self.in_dims[0], hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
        self.dloss_dout = Conv1DMLP(ftask_dim + self.out_dims[-1], self.out_dims[-1], hidden_dim=deriv_hidden_dim,
                                    num_layers=driv_num_layers)

    def forward(self, ftask, weight_embs, encoders, decoders):
        z_ins = [self.forward_in(ftask)]
        for weight_emb, opt_sub_block in zip(weight_embs, self.opt_sub_blocks):
            z_ins.append(opt_sub_block.pseudo_forward(z_ins[-1], weight_emb, ftask))

        dl_douts = [self.dloss_dout(torch.cat([ftask, z_ins[-1]], dim=-1))]

        dw_dicts = []
        for opt_sub_block, z_in, weight_emb, decoder in reversed(
                list(zip(self.opt_sub_blocks, z_ins[:-1], weight_embs, decoders))):
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


class EfficientHypernet(nn.Module):
    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 codec_hidden_dim, codec_num_layers,
                 num_iterations, *args, **kwargs):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.num_iterations = num_iterations

        self.encoders = nn.ModuleList([
            ModuleEncoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])
        self.decoders = nn.ModuleList([
            ModuleDecoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])

        # Use a single shared OptBlock instead of multiple blocks
        self.opt_block = SharedOptBlock(target_net, ftask_dim, weight_dim,
                                        deriv_hidden_dim, driv_num_layers,
                                        *args, **kwargs)

        # Dynamic learning rates for each iteration
        self.dynamic_lrs = nn.Parameter(torch.zeros(num_iterations).fill_(-1e-2))
        self.layer_norms = nn.ModuleList(
            [ParamLN(weight_dim) for submodule in self.target_net.get_submodules()]
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"EfficientHyperNet parameters: {n_params / 1e6:.2f}M")

    def forward_iteration(self, ftask, weight_embs):
        """Single iteration of the optimization block"""
        weight_upd_dicts = self.opt_block(ftask, weight_embs, self.encoders, self.decoders)
        weight_upd_dicts = [
            ln(submodule)
            for ln, submodule in zip(self.layer_norms, weight_upd_dicts)
        ]
        return weight_upd_dicts

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

        # Reuse the same OptBlock for multiple iterations
        for i in range(self.num_iterations):
            weight_upd_embs = self.forward_iteration(ftask, weight_embs)
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


# Implement Skip-Connection version for DDIM-like prediction
class SkipOptBlock(nn.Module):
    '''
    An OptBlock with skip connections to allow DDIM-like layer skipping
    '''

    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 skip_factor=2, *args, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dims = target_net.get_in_dims()
        self.out_dims = target_net.get_out_dims()
        self.weight_dim = weight_dim
        self.skip_factor = skip_factor  # How many layers to skip per iteration

        self.num_layers = len(self.in_dims)

        # Create OptSubBlocks but potentially fewer of them
        self.opt_sub_blocks = nn.ModuleList([
            OptSubBlock(ftask_dim, in_dim, out_dim, weight_dim,
                        deriv_hidden_dim, driv_num_layers,
                        *args, **kwargs)
            for in_dim, out_dim in zip(self.in_dims[::skip_factor],
                                       [self.out_dims[min(i * skip_factor, len(self.out_dims) - 1)]
                                        for i in range(len(self.in_dims[::skip_factor]))])
        ])

        self.forward_in = Conv1DMLP(ftask_dim, self.in_dims[0], hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
        self.dloss_dout = Conv1DMLP(ftask_dim + self.out_dims[-1], self.out_dims[-1], hidden_dim=deriv_hidden_dim,
                                    num_layers=driv_num_layers)

        # For interpolating skipped layers
        self.interpolate_net = Conv1DMLP(weight_dim * 2, weight_dim, hidden_dim=deriv_hidden_dim,
                                         num_layers=driv_num_layers)

    def forward(self, ftask, weight_embs, encoders, decoders, layer_mask=None):
        # Implementation for layer skipping would go here
        # This is a conceptual example - full implementation would require careful handling of indices

        # Use layer_mask to determine which layers to process
        if layer_mask is None:
            active_indices = list(range(0, len(weight_embs), self.skip_factor))
        else:
            active_indices = [i for i, mask in enumerate(layer_mask) if mask]

        active_weight_embs = [weight_embs[i] for i in active_indices]

        # Forward pass through active layers only
        z_ins = [self.forward_in(ftask)]
        for weight_emb, opt_sub_block in zip(active_weight_embs, self.opt_sub_blocks):
            z_ins.append(opt_sub_block.pseudo_forward(z_ins[-1], weight_emb, ftask))

        # Backward pass
        dl_douts = [self.dloss_dout(torch.cat([ftask, z_ins[-1]], dim=-1))]
        dw_dicts = []

        for opt_sub_block, z_in, weight_emb in reversed(
                list(zip(self.opt_sub_blocks, z_ins[:-1], active_weight_embs))):
            dl_dout, dl_dw = opt_sub_block.pseudo_backward(z_in, weight_emb, dl_douts[-1], ftask)
            dl_douts.append(dl_dout)
            dw_dicts.append(dl_dw)

        dw_dicts = list(reversed(dw_dicts))

        # Expand back to full size with interpolation for skipped layers
        full_dw_dicts = [None] * len(weight_embs)
        for i, dw in zip(active_indices, dw_dicts):
            full_dw_dicts[i] = dw

        # Interpolate skipped layers
        for i in range(len(weight_embs)):
            if full_dw_dicts[i] is None:
                # Find nearest available layers
                prev_idx = max([j for j in active_indices if j < i], default=active_indices[0])
                next_idx = min([j for j in active_indices if j > i], default=active_indices[-1])

                # Interpolate between available dw values
                if prev_idx == next_idx:
                    full_dw_dicts[i] = full_dw_dicts[prev_idx]
                else:
                    # Linear interpolation based on position
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    full_dw_dicts[i] = self.interpolate_net(
                        torch.cat([full_dw_dicts[prev_idx], full_dw_dicts[next_idx]], dim=-1))

        return full_dw_dicts


class RNNHypernet(nn.Module):
    """
    A RNN-based Hypernet implementation where the OptBlock is implemented as
    a recurrent network to share weights across layers
    """

    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 hidden_dim, num_layers,
                 codec_hidden_dim, codec_num_layers,
                 num_iterations, *args, **kwargs):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations

        # Encoders/decoders as before
        self.encoders = nn.ModuleList([
            ModuleEncoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])
        self.decoders = nn.ModuleList([
            ModuleDecoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])

        # RNN for forward pass
        self.forward_rnn = nn.GRU(
            input_size=weight_dim + ftask_dim,  # weight embedding + task
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection for forward pass
        self.forward_proj = Conv1DMLP(hidden_dim, target_net.get_out_dims()[-1],
                                      hidden_dim=hidden_dim, num_layers=2)

        # RNN for backward pass
        self.backward_rnn = nn.GRU(
            input_size=hidden_dim + ftask_dim,  # hidden state from forward + task
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection for backward pass (dw prediction)
        self.dw_proj = Conv1DMLP(hidden_dim, weight_dim, hidden_dim=hidden_dim, num_layers=2)

        # Dynamic learning rates
        self.dynamic_lrs = nn.Parameter(torch.zeros(num_iterations).fill_(-1e-2))

        self.layer_norms = nn.ModuleList(
            [ParamLN(weight_dim) for submodule in self.target_net.get_submodules()]
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"RNNHyperNet parameters: {n_params / 1e6:.2f}M")

    def forward_blocks(self, ftask):
        batch_size = ftask.shape[0]
        num_modules = len(self.target_net.get_submodules())

        # Initial weights
        weight_dicts = list(
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        )
        weight_embs = [encoder(weight_dict) for encoder, weight_dict in zip(self.encoders, weight_dicts)]
        weight_embs = [weight_emb.repeat(batch_size, 1) for weight_emb in weight_embs]

        final_weight_dicts = []

        for iteration in range(self.num_iterations):
            # Prepare sequence for RNN
            # [batch_size, seq_len, features]
            weight_seq = torch.stack(weight_embs, dim=1)
            ftask_expanded = ftask.unsqueeze(1).expand(-1, num_modules, -1)
            rnn_input = torch.cat([weight_seq, ftask_expanded], dim=2)

            # Forward pass through RNN
            forward_outputs, _ = self.forward_rnn(rnn_input)

            # Process last hidden state for backward input
            backward_input = torch.cat([forward_outputs, ftask_expanded], dim=2)

            # Backward pass through RNN
            backward_outputs, _ = self.backward_rnn(backward_input)

            # Project to weight updates
            dw_embs = []
            for i in range(num_modules):
                dw = self.dw_proj(backward_outputs[:, i])
                dw = self.layer_norms[i](dw)
                dw_embs.append(dw)

            # Update weights
            weight_embs = [v + self.dynamic_lrs[iteration] * dw for v, dw in zip(weight_embs, dw_embs)]

            # Convert to weight dicts
            weight_dicts = [decoder(w) for decoder, w in zip(self.decoders, weight_embs)]
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
    # Test the original model
    model = Hypernet(Toy(64, out_dim=7, hidden_dim=16, num_layers=2), ftask_dim=10, weight_dim=128,
                     deriv_hidden_dim=32, driv_num_layers=2,
                     codec_hidden_dim=64, codec_num_layers=2, num_layers=8)
    input = torch.randn(128, 64)
    f_input = torch.randn(128, 10)
    output = model(f_input, input)

    # Test the efficient model
    print("\nTesting EfficientHypernet...")
    efficient_model = EfficientHypernet(Toy(64, out_dim=7, hidden_dim=16, num_layers=2),
                                        ftask_dim=10, weight_dim=128,
                                        deriv_hidden_dim=32, driv_num_layers=2,
                                        codec_hidden_dim=64, codec_num_layers=2,
                                        num_iterations=8)
    output = efficient_model(f_input, input)
    print(f"Output shape: {output.shape}")

    # Test the RNN model
    print("\nTesting RNNHypernet...")
    rnn_model = RNNHypernet(Toy(64, out_dim=7, hidden_dim=16, num_layers=2),
                            ftask_dim=10, weight_dim=128,
                            hidden_dim=64, num_layers=2,
                            codec_hidden_dim=64, codec_num_layers=2,
                            num_iterations=8)
    output = rnn_model(f_input, input)
    print(f"Output shape: {output.shape}")