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


class Conv1dMLP(nn.Module):
    '''
    A replacement for MLP using 1D convolutions for parameter efficiency
    '''

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, kernel_size=3, activation=F.relu):
        super().__init__()

        self.activation = activation
        self.in_proj = nn.Linear(in_dim, hidden_dim)  # Project to hidden dim

        # Middle layers use Conv1d with depthwise separable convolutions for efficiency
        if num_layers > 2:
            self.conv_layers = nn.ModuleList()
            for _ in range(num_layers - 2):
                # Depthwise convolution
                self.conv_layers.append(nn.Conv1d(
                    hidden_dim, hidden_dim, kernel_size,
                    padding=(kernel_size - 1) // 2,
                    groups=hidden_dim  # Depthwise conv (each channel separately)
                ))
                # Pointwise convolution (1x1)
                self.conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
        else:
            self.conv_layers = nn.ModuleList()

        self.out_proj = nn.Linear(hidden_dim, out_dim)  # Project to output dim

    def forward(self, x):
        batch_size = x.shape[0]

        # Project to hidden dimension
        x = self.activation(self.in_proj(x))

        if len(self.conv_layers) > 0:
            # Reshape for Conv1d: [B, C] -> [B, C, 1]
            x = x.unsqueeze(-1)

            # Apply conv layers
            for i, conv in enumerate(self.conv_layers):
                x = conv(x)
                if i % 2 == 1:  # Apply activation after each pointwise conv
                    x = self.activation(x)

            # Back to original shape
            x = x.squeeze(-1)

        # Project to output dimension
        x = self.out_proj(x)
        return x


class ModuleEncoder(nn.Module):
    def __init__(self, target_net, weight_dim, hidden_dim, num_layers):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.encoder = Conv1dMLP(self.param_cnt, weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)

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
        self.decoder = Conv1dMLP(weight_dim, self.param_cnt, hidden_dim=hidden_dim, num_layers=num_layers)
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
        self.weight_dim = weight_dim  # Ensure weight_dim is stored as an instance variable
        self.hidden_dim = hidden_dim

        # Modified to include ftask_dim and use Conv1dMLP
        self.forward_net = Conv1dMLP(in_dim + weight_dim + ftask_dim, out_dim, hidden_dim=hidden_dim,
                                     num_layers=num_layers)
        self.dout_din = Conv1dMLP(in_dim + weight_dim + ftask_dim, out_dim * in_dim, hidden_dim=hidden_dim,
                                  num_layers=num_layers)
        self.dout_dw = Conv1dMLP(in_dim + weight_dim + ftask_dim, out_dim * weight_dim, hidden_dim=hidden_dim,
                                 num_layers=num_layers)

        self.dl_din_way = dl_din_way
        self.dl_dw_way = dl_dw_way

        if dl_din_way == 'slice':
            self.mm_mlp_in = Conv1dMLP(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_din_way == 'full':
            self.mm_mlp_in = Conv1dMLP(out_dim * (in_dim + 1), in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            assert dl_din_way == 'direct'

        if dl_dw_way == 'slice':
            self.mm_mlp_w = Conv1dMLP(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_dw_way == 'full':
            self.mm_mlp_w = Conv1dMLP(out_dim * (weight_dim + 1), weight_dim, hidden_dim=hidden_dim,
                                      num_layers=num_layers)
        else:
            assert dl_dw_way == 'direct'

    def pseudo_forward(self, z_in, weight_emb, ftask):
        # Added ftask as input
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])

        # Repeat ftask if needed
        if ftask.shape[0] != z_in.shape[0]:
            ftask = einops.repeat(ftask, '1 i -> n i', n=z_in.shape[0])

        # Include ftask in concatenation
        out = self.forward_net(torch.cat([z_in, weight_emb, ftask], dim=-1))
        return out

    def pseudo_backward(self, z_in, weight_emb, dl_dout, ftask):
        # Added ftask as input
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])

        # Repeat ftask if needed
        if ftask.shape[0] != z_in.shape[0]:
            ftask = einops.repeat(ftask, '1 i -> n i', n=z_in.shape[0])

        # Include ftask in concatenation for both dout_din and dout_dw
        dout_din = einops.rearrange(
            self.dout_din(torch.cat([z_in, weight_emb, ftask], dim=-1)),
            'n (o i) -> n o i', o=self.out_dim
        )
        dout_dw = einops.rearrange(
            self.dout_dw(torch.cat([z_in, weight_emb, ftask], dim=-1)),
            'n (o i) -> n o i', o=self.out_dim
        )

        if self.dl_din_way == 'direct':
            dl_din = einops.einsum(dl_dout, dout_din, 'n o, n o i -> n i')
        elif self.dl_din_way == 'slice':
            # Using the approach from v1 for dl_din
            batch_size = dl_dout.shape[0]
            out_dim = dl_dout.shape[1]
            products = torch.zeros(batch_size, 2 * out_dim, device=dl_dout.device)
            products[:, :out_dim] = dl_dout
            for i in range(out_dim):
                products[:, out_dim + i] = torch.mean(dout_din[:, i, :], dim=1)
            dl_din = self.mm_mlp_in(products)
        elif self.dl_din_way == 'full':
            dl_din = self.mm_mlp_in(
                torch.cat([dl_dout, dout_din.flatten(1)], dim=1)
            )

        if self.dl_dw_way == 'direct':
            dl_dw = einops.einsum(dl_dout, dout_dw, 'n o, n o i -> n i')
        elif self.dl_dw_way == 'slice':
            # Using the approach from v1 for dl_dw
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


class SharedOptBlock(nn.Module):
    '''
    A shared OptBlock that can be applied multiple times with the same weights.
    '''

    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 layer_sampling_strategy='all',  # 'all', 'skip', 'adaptive'
                 skip_steps=1,  # For skip strategy
                 *args, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dims = target_net.get_in_dims()
        self.out_dims = target_net.get_out_dims()
        self.weight_dim = weight_dim
        self.layer_sampling_strategy = layer_sampling_strategy
        self.skip_steps = skip_steps

        # Create a single set of OptSubBlocks
        self.opt_sub_blocks = nn.ModuleList([
            OptSubBlock(ftask_dim, in_dim, out_dim, weight_dim,
                        deriv_hidden_dim, driv_num_layers,
                        *args, **kwargs)
            for in_dim, out_dim in zip(self.in_dims, self.out_dims)
        ])

        self.forward_in = Conv1dMLP(ftask_dim, self.in_dims[0], hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
        self.dloss_dout = Conv1dMLP(ftask_dim + self.out_dims[-1], self.out_dims[-1], hidden_dim=deriv_hidden_dim,
                                    num_layers=driv_num_layers)

        # For adaptive layer sampling
        if layer_sampling_strategy == 'adaptive':
            self.layer_mask_net = Conv1dMLP(ftask_dim + 1, len(self.in_dims),
                                            hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)

        # For interpolation between skipped layers
        if layer_sampling_strategy in ['skip', 'adaptive']:
            self.z_interpolation_nets = nn.ModuleList([
                Conv1dMLP(in_dim + weight_dim + ftask_dim, out_dim,
                          hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
                for in_dim, out_dim in zip(self.in_dims, self.out_dims)
            ])

            self.grad_interpolation_nets = nn.ModuleList([
                Conv1dMLP(out_dim + weight_dim + ftask_dim, in_dim,
                          hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
                for in_dim, out_dim in zip(self.in_dims, self.out_dims)
            ])

    def _get_layer_mask(self, ftask, iteration=0):
        if self.layer_sampling_strategy == 'all':
            # Process all layers
            return torch.ones(ftask.shape[0], len(self.in_dims)).to(ftask.device)

        elif self.layer_sampling_strategy == 'skip':
            # Process every skip_steps layers
            mask = torch.zeros(ftask.shape[0], len(self.in_dims)).to(ftask.device)

            # Always process first and last layers
            mask[:, 0] = 1
            mask[:, -1] = 1

            # Process intermediate layers based on skip_steps and iteration
            for i in range(1, len(self.in_dims) - 1):
                if (i + iteration) % self.skip_steps == 0:
                    mask[:, i] = 1

            return mask

        elif self.layer_sampling_strategy == 'adaptive':
            # Generate adaptive mask based on task and iteration
            iter_tensor = torch.ones(ftask.shape[0], 1).to(ftask.device) * (iteration / 10.0)
            mask_logits = self.layer_mask_net(torch.cat([ftask, iter_tensor], dim=-1))

            # Always process first and last layers
            mask = torch.sigmoid(mask_logits)
            mask[:, 0] = 1
            mask[:, -1] = 1

            return mask

        else:
            raise ValueError(f"Unknown layer sampling strategy: {self.layer_sampling_strategy}")

    def forward(self, ftask, weight_embs, encoders, decoders, iteration=0):
        # Get layer mask for this iteration
        layer_mask = self._get_layer_mask(ftask, iteration)

        # Initial z
        z_ins = [self.forward_in(ftask)]

        # Forward pass with selective layer processing
        processed_indices = []

        for i, (weight_emb, opt_sub_block) in enumerate(zip(weight_embs, self.opt_sub_blocks)):
            # If layer mask indicates this layer should be processed
            if layer_mask[:, i].mean() > 0.5:  # Process layer if mask value is high
                z_ins.append(opt_sub_block.pseudo_forward(z_ins[-1], weight_emb, ftask))
                processed_indices.append(i)
            else:
                # For skipped layers, use interpolation
                if i > 0:
                    # Find nearest processed layers for interpolation
                    prev_processed = max([idx for idx in processed_indices if idx < i], default=0)

                    # Use interpolation network
                    z_in = z_ins[-1]
                    interpolated_z = self.z_interpolation_nets[i](
                        torch.cat([z_in, weight_emb, ftask], dim=-1)
                    )
                    z_ins.append(interpolated_z)

        # Backward pass with selective layer processing
        dl_douts = [self.dloss_dout(torch.cat([ftask, z_ins[-1]], dim=-1))]
        dw_dicts = [None] * len(self.opt_sub_blocks)  # Initialize with None placeholders

        processed_indices.reverse()  # For backward pass

        for i, idx in enumerate(processed_indices):
            opt_sub_block = self.opt_sub_blocks[idx]
            z_in = z_ins[idx]
            weight_emb = weight_embs[idx]

            # Process this layer
            dl_dout, dl_dw = opt_sub_block.pseudo_backward(z_in, weight_emb, dl_douts[-1], ftask)
            dl_douts.append(dl_dout)
            dw_dicts[idx] = dl_dw

        # Fill in skipped layers with interpolated gradients
        for i in range(len(dw_dicts)):
            if dw_dicts[i] is None:
                # For skipped layers, approximate gradients using nearest processed layers
                processed_before = [idx for idx in processed_indices if idx < i]
                processed_after = [idx for idx in processed_indices if idx > i]

                if processed_before and processed_after:
                    # Interpolate between nearest processed layers
                    prev_idx = max(processed_before)
                    next_idx = min(processed_after)

                    weight_emb = weight_embs[i]
                    dw_dicts[i] = torch.zeros_like(weight_emb)  # Initialize with zeros

                    # Use a simple weighted average as default interpolation
                    total_dist = next_idx - prev_idx
                    if total_dist > 0:
                        weight_prev = (next_idx - i) / total_dist
                        weight_next = (i - prev_idx) / total_dist

                        dw_dicts[i] = weight_prev * dw_dicts[prev_idx] + weight_next * dw_dicts[next_idx]
                elif processed_before:
                    # Use the last processed layer's gradient
                    dw_dicts[i] = dw_dicts[max(processed_before)]
                elif processed_after:
                    # Use the next processed layer's gradient
                    dw_dicts[i] = dw_dicts[min(processed_after)]
                else:
                    # This shouldn't happen with our sampling strategy, but just in case
                    dw_dicts[i] = torch.zeros_like(weight_embs[i])

        return dw_dicts


class ParamLN(nn.Module):
    def __init__(self, weight_dim):
        super().__init__()
        self.ln = nn.LayerNorm(weight_dim)

    def forward(self, weight_dict):
        return self.ln(weight_dict)


class ImprovedHypernet(nn.Module):
    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 codec_hidden_dim, codec_num_layers,
                 num_steps=8,
                 layer_sampling_strategy='all',  # 'all', 'skip', 'adaptive'
                 skip_steps=1,  # For skip strategy
                 *args, **kwargs):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.num_steps = num_steps

        self.encoders = nn.ModuleList([
            ModuleEncoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])

        self.decoders = nn.ModuleList([
            ModuleDecoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])

        # Single shared OptBlock used for all optimization steps
        self.shared_opt_block = SharedOptBlock(
            target_net, ftask_dim, weight_dim,
            deriv_hidden_dim, driv_num_layers,
            layer_sampling_strategy=layer_sampling_strategy,
            skip_steps=skip_steps,
            *args, **kwargs
        )

        # Learnable step sizes
        self.dynamic_lrs = nn.Parameter(torch.zeros(num_steps).fill_(-1e-6))

        # Layer normalizations for weight updates
        self.layer_norms = nn.ModuleList([
            ParamLN(weight_dim) for submodule in self.target_net.get_submodules()
        ])

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Improved HyperNet parameters: {n_params / 1e6:.2f}M")

    def forward(self, ftask, inputs, early_sup=False):
        # Get initial weight dictionaries for each submodule
        weight_dicts = list(
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        )

        # Encode initial weights
        weight_embs = list(
            encoder(weight_dict)
            for encoder, weight_dict in zip(self.encoders, weight_dicts)
        )

        # Ensure weight embeddings match batch size
        weight_embs = [weight_emb.repeat(ftask.shape[0], 1) for weight_emb in weight_embs]

        # List to store final weight dictionaries for each step
        final_weight_dicts = []

        # Iteratively update weights using the shared OptBlock
        for i, lr in enumerate(self.dynamic_lrs):
            # Run the shared OptBlock with the current iteration
            weight_upd_embs = self.shared_opt_block(
                ftask, weight_embs, self.encoders, self.decoders, iteration=i
            )

            # Apply layer normalization and update weights
            weight_upd_embs = [
                ln(v_upd) for ln, v_upd in zip(self.layer_norms, weight_upd_embs)
            ]
            weight_embs = [
                v + lr * v_upd for v, v_upd in zip(weight_embs, weight_upd_embs)
            ]

            # Decode updated weights
            weight_dicts = [
                decoder(w) for decoder, w in zip(self.decoders, weight_embs)
            ]

            # Merge weights from submodules into a single dict for the target network
            weight_dict = self.target_net.merge_submodule_weights(weight_dicts)
            final_weight_dicts.append(weight_dict)

        # Return all intermediate results or just the final one
        if early_sup:
            return torch.stack([
                torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(
                    self.target_net, generated_weight, inputs
                ) for generated_weight in final_weight_dicts
            ])
        else:
            return torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(
                self.target_net, final_weight_dicts[-1], inputs
            )

    def forward_blocks(self, ftask):
        """
        Forward pass that returns the generated weight dictionaries for each optimization step.
        This method is needed for compatibility with SharedParamHypernet.

        Args:
            ftask: Task features tensor with shape [batch_size, ftask_dim]

        Returns:
            List of weight dictionaries for each optimization step
        """
        # Get initial weight dictionaries for each submodule
        weight_dicts = list(
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        )

        # Encode initial weights
        weight_embs = list(
            encoder(weight_dict)
            for encoder, weight_dict in zip(self.encoders, weight_dicts)
        )

        # Ensure weight embeddings match batch size
        weight_embs = [weight_emb.repeat(ftask.shape[0], 1) for weight_emb in weight_embs]

        # List to store final weight dictionaries for each step
        final_weight_dicts = []

        # Iteratively update weights using the shared OptBlock
        for i, lr in enumerate(self.dynamic_lrs):
            # Run the shared OptBlock with the current iteration
            weight_upd_embs = self.shared_opt_block(
                ftask, weight_embs, self.encoders, self.decoders, iteration=i
            )

            # Debug the shape of weight_upd_embs
            for j, v_upd in enumerate(weight_upd_embs):
                if v_upd.shape[-1] != self.weight_dim:
                    # Ensure the shape is correct before applying layer norm
                    # If it's a single value per batch item, expand it to match weight_dim
                    if len(v_upd.shape) == 2 and v_upd.shape[1] == 1:
                        weight_upd_embs[j] = v_upd.expand(-1, self.weight_dim)

                    # If it's a different incompatible shape, reshape it if possible
                    # or skip layer normalization for this component

            # Apply layer normalization with shape check
            normalized_upd_embs = []
            for ln, v_upd in zip(self.layer_norms, weight_upd_embs):
                # Skip layer norm if shapes don't match and apply a simple scaling instead
                if v_upd.shape[-1] != self.weight_dim:
                    # Apply simple mean-variance normalization manually
                    mean = v_upd.mean(dim=-1, keepdim=True)
                    var = v_upd.var(dim=-1, keepdim=True, unbiased=False)
                    normalized = (v_upd - mean) / torch.sqrt(var + 1e-5)
                    normalized_upd_embs.append(normalized)
                else:
                    normalized_upd_embs.append(ln(v_upd))

            # Update weights
            weight_embs = [
                v + lr * v_upd for v, v_upd in zip(weight_embs, normalized_upd_embs)
            ]

            # Decode updated weights
            weight_dicts = [
                decoder(w) for decoder, w in zip(self.decoders, weight_embs)
            ]

            # Merge weights from submodules into a single dict for the target network
            weight_dict = self.target_net.merge_submodule_weights(weight_dicts)
            final_weight_dicts.append(weight_dict)

        return final_weight_dicts


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
    print("Testing original Hypernet...")
    # Create and test the original model for comparison
    model_orig = Hypernet(
        Toy(64, out_dim=7, hidden_dim=16, num_layers=2),
        ftask_dim=10, weight_dim=128,
        deriv_hidden_dim=32, driv_num_layers=2,
        codec_hidden_dim=64, codec_num_layers=2,
        num_layers=8
    )

    print("Testing improved Hypernet...")
    # Create and test the improved model
    model_improved = ImprovedHypernet(
        Toy(64, out_dim=7, hidden_dim=16, num_layers=2),
        ftask_dim=10, weight_dim=128,
        deriv_hidden_dim=32, driv_num_layers=2,
        codec_hidden_dim=64, codec_num_layers=2,
        num_steps=4,
        layer_sampling_strategy='adaptive',  # Use adaptive layer sampling
        skip_steps=2  # Skip every other layer
    )

    # Test both models
    input_tensor = torch.randn(128, 64)
    ftask_tensor = torch.randn(128, 10)

    # Forward pass with original model
    output_orig = model_orig(ftask_tensor, input_tensor)
    print(f"Original model output shape: {output_orig.shape}")

    # Forward pass with improved model
    output_improved = model_improved(ftask_tensor, input_tensor)
    print(f"Improved model output shape: {output_improved.shape}")

    # Check parameter counts
    orig_params = sum(p.numel() for p in model_orig.parameters())
    improved_params = sum(p.numel() for p in model_improved.parameters())

    print(f"Original model parameters: {orig_params / 1e6:.2f}M")
    print(f"Improved model parameters: {improved_params / 1e6:.2f}M")
    print(f"Parameter reduction: {100 * (1 - improved_params / orig_params):.2f}%")