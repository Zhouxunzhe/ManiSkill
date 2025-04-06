import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, List, Dict
from .hypernetwork import TargetNet


# Keep the original helper modules
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# 1. Create Down Path Network (Encoder)
class DownPathNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 down_dims,
                 cond_dim,
                 kernel_size=5,
                 n_groups=8):
        """
        Encapsulates the down path (encoder) of the UNet

        input_dim: Dimension of input
        down_dims: Channel sizes for each level
        cond_dim: Dimension of conditioning (diffusion step + global cond)
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Create down modules
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # Create mid modules
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        self.dims = all_dims

        n_params = sum(p.numel() for p in self.parameters())
        print(f"DownPathNetwork parameters: {n_params / 1e6:.2f}M")

    def forward(self, x, global_feature):
        """
        x: Input tensor [B, C, T]
        global_feature: Conditioning information [B, cond_dim]

        Returns:
        - mid_output: Output from mid modules
        - h: List of intermediate outputs for skip connections
        """
        h = []

        # Down path
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Mid modules
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        return x, h

    def get_dims(self):
        """Return dimensions for connecting with up path"""
        return self.dims


# 2. Create Up Path Network (Decoder)
class UpPathNetwork(nn.Module):
    def __init__(self,
                 dims,
                 cond_dim,
                 kernel_size=5,
                 n_groups=8):
        """
        Encapsulates the up path (decoder) of the UNet

        dims: List of channel dimensions matching the down path
        cond_dim: Dimension of conditioning (diffusion step + global cond)
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        input_dim = dims[0]
        down_dims = dims[1:]
        all_dims = dims
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        start_dim = down_dims[0]

        # Create up modules
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        # Final conv
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"UpPathNetwork parameters: {n_params / 1e6:.2f}M")

    def forward(self, x, h, global_feature):
        """
        x: Output from mid modules
        h: List of intermediate outputs from down path
        global_feature: Conditioning information

        Returns: Final output tensor
        """
        # Up path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # Final conv
        x = self.final_conv(x)

        return x


# 3. Target Network classes
class DownPathTargetNet(TargetNet):
    def __init__(self, down_path):
        """
        Target network for the down path that can use external parameters

        down_path: Original DownPathNetwork
        """
        super().__init__()
        self.down_path = down_path

    def forward(self, x, global_feature, params=None):
        """
        Forward pass using optional external parameters

        x: Input tensor [B, C, T]
        global_feature: Conditioning information [B, cond_dim]
        params: Optional dictionary of parameters to use instead of self.down_path

        Returns:
        - mid_output: Output from mid modules
        - h: List of intermediate outputs for skip connections
        """
        if params is None:
            return self.down_path(x, global_feature)

        h = []

        # Down path with custom parameters
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_path.down_modules):
            # Apply first ResNet block with custom parameters
            prefix_r1 = f'down_modules.{idx}.0.'
            x = self._apply_conditional_resblock(x, global_feature, resnet, params, prefix_r1)

            # Apply second ResNet block with custom parameters
            prefix_r2 = f'down_modules.{idx}.1.'
            x = self._apply_conditional_resblock(x, global_feature, resnet2, params, prefix_r2)

            h.append(x)
            x = downsample(x)  # Downsample doesn't have parameters to replace

        # Mid modules with custom parameters
        for idx, mid_module in enumerate(self.down_path.mid_modules):
            prefix_mid = f'mid_modules.{idx}.'
            x = self._apply_conditional_resblock(x, global_feature, mid_module, params, prefix_mid)

        return x, h

    def _apply_conditional_resblock(self, x, cond, original_module, params, prefix):
        """
        Apply a conditional residual block with custom parameters
        """
        def single_conv(x, weight, bias, padding=None, stride=None):
            return F.conv1d(x, weight, bias, padding=padding, stride=stride)

        vmap_conv = torch.vmap(single_conv, in_dims=(0, 0, 0, None, None))

        # First Conv1dBlock
        if f'{prefix}blocks.0.block.0.weight' in params and f'{prefix}blocks.0.block.0.bias' in params:
            conv1_weight = params[f'{prefix}blocks.0.block.0.weight']
            conv1_bias = params[f'{prefix}blocks.0.block.0.bias']
            x_norm = vmap_conv(
                x,
                conv1_weight,
                conv1_bias,
                original_module.blocks[0].block[0].padding,
                original_module.blocks[0].block[0].stride
            )
            if f'{prefix}blocks.0.block.1.weight' in params and f'{prefix}blocks.0.block.1.bias' in params:
                norm_weight = params[f'{prefix}blocks.0.block.1.weight']
                norm_bias = params[f'{prefix}blocks.0.block.1.bias']
                # Ensure norm_weight and norm_bias are [num_channels] by taking the mean across batch
                if norm_weight.dim() > 1 and norm_weight.shape[0] == x.shape[0]:
                    norm_weight = norm_weight.mean(dim=0)  # [128, 64] -> [64]
                    norm_bias = norm_bias.mean(dim=0)      # [128, 64] -> [64]
                x_norm = F.group_norm(x_norm, original_module.blocks[0].block[1].num_groups,
                                      norm_weight, norm_bias)
            else:
                x_norm = original_module.blocks[0].block[1](x_norm)
            x_norm = original_module.blocks[0].block[2](x_norm)  # Apply Mish
        else:
            x_norm = original_module.blocks[0](x)

        # FiLM conditioning
        if f'{prefix}cond_encoder.1.weight' in params and f'{prefix}cond_encoder.1.bias' in params:
            linear_weight = params[f'{prefix}cond_encoder.1.weight']
            linear_bias = params[f'{prefix}cond_encoder.1.bias']
            embed = torch.bmm(cond.unsqueeze(1), linear_weight.transpose(1, 2)).squeeze(1) + linear_bias
            embed = original_module.cond_encoder[0](embed)  # Apply Mish
            embed = embed.unsqueeze(-1)  # Match Unflatten behavior
        else:
            embed = original_module.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, original_module.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        x_norm = scale * x_norm + bias

        # Second Conv1dBlock
        if f'{prefix}blocks.1.block.0.weight' in params and f'{prefix}blocks.1.block.0.bias' in params:
            conv2_weight = params[f'{prefix}blocks.1.block.0.weight']
            conv2_bias = params[f'{prefix}blocks.1.block.0.bias']
            x_norm = vmap_conv(
                x_norm,
                conv2_weight,
                conv2_bias,
                original_module.blocks[1].block[0].padding,
                original_module.blocks[1].block[0].stride
            )
            if f'{prefix}blocks.1.block.1.weight' in params and f'{prefix}blocks.1.block.1.bias' in params:
                norm_weight = params[f'{prefix}blocks.1.block.1.weight']
                norm_bias = params[f'{prefix}blocks.1.block.1.bias']
                # Ensure norm_weight and norm_bias are [num_channels] by taking the mean across batch
                if norm_weight.dim() > 1 and norm_weight.shape[0] == x.shape[0]:
                    norm_weight = norm_weight.mean(dim=0)  # [128, 64] -> [64]
                    norm_bias = norm_bias.mean(dim=0)      # [128, 64] -> [64]
                x_norm = F.group_norm(x_norm, original_module.blocks[1].block[1].num_groups,
                                      norm_weight, norm_bias)
            else:
                x_norm = original_module.blocks[1].block[1](x_norm)
            x_norm = original_module.blocks[1].block[2](x_norm)  # Apply Mish
        else:
            x_norm = original_module.blocks[1](x_norm)

        # Residual connection
        if f'{prefix}residual_conv.weight' in params and f'{prefix}residual_conv.bias' in params:
            res_weight = params[f'{prefix}residual_conv.weight']
            res_bias = params[f'{prefix}residual_conv.bias']
            x_shortcut = vmap_conv(
                x,
                res_weight,
                res_bias,
                0,
                original_module.residual_conv.stride
            )
        else:
            x_shortcut = original_module.residual_conv(x)

        return x_shortcut + x_norm

    def get_in_dims(self):
        """
        Return the input dimensions of the down path
        """
        return [self.down_path.dims[0]]  # Input dimension of the first layer

    def get_out_dims(self):
        """
        Return the output dimensions of the down path
        """
        return [self.down_path.dims[-1]]  # Output dimension of the last layer (mid output)

    def get_submodules(self):
        """
        Return the list of submodules (down_path in this case)
        """
        return [self.down_path]

    def get_submodule_names(self):
        """
        Return the names of submodules
        """
        return ['down_path']

    def merge_submodule_weights(self, weight_dicts):
        """
        Merge the weight dictionaries of submodules into a single weight dictionary
        """
        return weight_dicts[0]  # Since there's only one submodule (down_path)

class UpPathTargetNet(TargetNet):
    def __init__(self, up_path):
        """
        Target network for the up path that can use external parameters

        up_path: Original UpPathNetwork
        """
        super().__init__()
        self.up_path = up_path

    def forward(self, x, h, global_feature, params=None):
        """
        Forward pass using optional external parameters

        x: Mid output from down path
        h: List of skip connections from down path
        global_feature: Conditioning information
        params: Optional dictionary of parameters

        Returns: Output tensor
        """
        if params is None:
            return self.up_path(x, h, global_feature)

        # Make a copy of h since we'll be popping from it
        h_copy = h.copy()

        # Up path with custom parameters
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_path.up_modules):
            x = torch.cat((x, h_copy.pop()), dim=1)

            # Apply first ResNet block with custom parameters
            prefix_r1 = f'up_modules.{idx}.0.'
            x = self._apply_conditional_resblock(x, global_feature, resnet, params, prefix_r1)

            # Apply second ResNet block with custom parameters
            prefix_r2 = f'up_modules.{idx}.1.'
            x = self._apply_conditional_resblock(x, global_feature, resnet2, params, prefix_r2)

            x = upsample(x)  # Upsample doesn't have parameters to replace

        # Final conv
        if len(self.up_path.final_conv) >= 2:
            # Apply first conv block
            prefix_fc1 = 'final_conv.0.'
            if f'{prefix_fc1}block.0.weight' in params and f'{prefix_fc1}block.0.bias' in params:
                fc1_weight = params[f'{prefix_fc1}block.0.weight']
                fc1_bias = params[f'{prefix_fc1}block.0.bias']
                x = F.conv1d(x, fc1_weight, fc1_bias,
                             padding=self.up_path.final_conv[0].block[0].padding,
                             stride=self.up_path.final_conv[0].block[0].stride)
                if f'{prefix_fc1}block.1.weight' in params and f'{prefix_fc1}block.1.bias' in params:
                    norm_weight = params[f'{prefix_fc1}block.1.weight']
                    norm_bias = params[f'{prefix_fc1}block.1.bias']
                    x = F.group_norm(x, self.up_path.final_conv[0].block[1].num_groups,
                                   norm_weight, norm_bias)
                else:
                    x = self.up_path.final_conv[0].block[1](x)
                x = self.up_path.final_conv[0].block[2](x)  # Mish activation
            else:
                x = self.up_path.final_conv[0](x)

            # Apply final conv
            prefix_fc2 = 'final_conv.1.'
            if f'{prefix_fc2}weight' in params and f'{prefix_fc2}bias' in params:
                fc2_weight = params[f'{prefix_fc2}weight']
                fc2_bias = params[f'{prefix_fc2}bias']
                x = F.conv1d(x, fc2_weight, fc2_bias)
            else:
                x = self.up_path.final_conv[1](x)

        return x

    def _apply_conditional_resblock(self, x, cond, original_module, params, prefix):
        """
        Apply a conditional residual block with custom parameters
        """
        def single_conv(x, weight, bias, padding=None, stride=None):
            return F.conv1d(x, weight, bias, padding=padding, stride=stride)

        vmap_conv = torch.vmap(single_conv, in_dims=(0, 0, 0, None, None))

        # First Conv1dBlock
        if f'{prefix}blocks.0.block.0.weight' in params and f'{prefix}blocks.0.block.0.bias' in params:
            conv1_weight = params[f'{prefix}blocks.0.block.0.weight']
            conv1_bias = params[f'{prefix}blocks.0.block.0.bias']
            x_norm = vmap_conv(
                x,
                conv1_weight,
                conv1_bias,
                original_module.blocks[0].block[0].padding,
                original_module.blocks[0].block[0].stride
            )
            if f'{prefix}blocks.0.block.1.weight' in params and f'{prefix}blocks.0.block.1.bias' in params:
                norm_weight = params[f'{prefix}blocks.0.block.1.weight']
                norm_bias = params[f'{prefix}blocks.0.block.1.bias']
                # Ensure norm_weight and norm_bias are [num_channels] by taking the mean across batch
                if norm_weight.dim() > 1 and norm_weight.shape[0] == x.shape[0]:
                    norm_weight = norm_weight.mean(dim=0)  # [128, 64] -> [64]
                    norm_bias = norm_bias.mean(dim=0)      # [128, 64] -> [64]
                x_norm = F.group_norm(x_norm, original_module.blocks[0].block[1].num_groups,
                                      norm_weight, norm_bias)
            else:
                x_norm = original_module.blocks[0].block[1](x_norm)
            x_norm = original_module.blocks[0].block[2](x_norm)  # Apply Mish
        else:
            x_norm = original_module.blocks[0](x)

        # FiLM conditioning
        if f'{prefix}cond_encoder.1.weight' in params and f'{prefix}cond_encoder.1.bias' in params:
            linear_weight = params[f'{prefix}cond_encoder.1.weight']
            linear_bias = params[f'{prefix}cond_encoder.1.bias']
            embed = F.linear(cond, linear_weight, linear_bias)
            embed = original_module.cond_encoder[0](embed)  # Apply Mish
            embed = embed.reshape(embed.shape[0], 2, original_module.out_channels, 1)
        else:
            embed = original_module.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, original_module.out_channels, 1)

        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        x_norm = scale * x_norm + bias

        # Second Conv1dBlock
        if f'{prefix}blocks.1.block.0.weight' in params and f'{prefix}blocks.1.block.0.bias' in params:
            conv2_weight = params[f'{prefix}blocks.1.block.0.weight']
            conv2_bias = params[f'{prefix}blocks.1.block.0.bias']
            x_norm = vmap_conv(
                x_norm,
                conv2_weight,
                conv2_bias,
                original_module.blocks[1].block[0].padding,
                original_module.blocks[1].block[0].stride
            )
            if f'{prefix}blocks.1.block.1.weight' in params and f'{prefix}blocks.1.block.1.bias' in params:
                norm_weight = params[f'{prefix}blocks.1.block.1.weight']
                norm_bias = params[f'{prefix}blocks.1.block.1.bias']
                # Ensure norm_weight and norm_bias are [num_channels] by taking the mean across batch
                if norm_weight.dim() > 1 and norm_weight.shape[0] == x.shape[0]:
                    norm_weight = norm_weight.mean(dim=0)  # [128, 64] -> [64]
                    norm_bias = norm_bias.mean(dim=0)      # [128, 64] -> [64]
                x_norm = F.group_norm(x_norm, original_module.blocks[1].block[1].num_groups,
                                      norm_weight, norm_bias)
            else:
                x_norm = original_module.blocks[1].block[1](x_norm)
            x_norm = original_module.blocks[1].block[2](x_norm)  # Apply Mish
        else:
            x_norm = original_module.blocks[1](x_norm)

        # Residual connection
        if f'{prefix}residual_conv.weight' in params and f'{prefix}residual_conv.bias' in params:
            res_weight = params[f'{prefix}residual_conv.weight']
            res_bias = params[f'{prefix}residual_conv.bias']
            x_shortcut = vmap_conv(
                x,
                res_weight,
                res_bias,
                0,
                original_module.residual_conv.stride
            )
        else:
            x_shortcut = original_module.residual_conv(x)

        return x_shortcut + x_norm

    def get_in_dims(self):
        """
        Return the input dimensions of the up path
        """
        return [self.up_path.up_modules[0][0].blocks[0].block[0].in_channels]  # Input channels of first resnet block

    def get_out_dims(self):
        """
        Return the output dimensions of the up path
        """
        return [self.up_path.final_conv[-1].out_channels]  # Output channels of final conv

    def get_submodules(self):
        """
        Return the list of submodules (up_path in this case)
        """
        return [self.up_path]

    def get_submodule_names(self):
        """
        Return the names of submodules
        """
        return ['up_path']

    def merge_submodule_weights(self, weight_dicts):
        """
        Merge the weight dictionaries of submodules into a single weight dictionary
        """
        return weight_dicts[0]  # Since there's only one submodule (up_path)


# 4. Redesigned ConditionalUnet1D using the new components
class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8
                 ):
        """
        Modified UNet that uses separate down and up path networks

        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines number of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        # Diffusion step encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed + global_cond_dim
        all_dims = [input_dim] + list(down_dims)

        # Create down path and up path networks
        self.down_path = DownPathNetwork(
            input_dim, down_dims, cond_dim, kernel_size, n_groups)

        self.up_path = UpPathNetwork(
            all_dims, cond_dim, kernel_size, n_groups)

        # Create target networks
        self.down_path_target = DownPathTargetNet(self.down_path)
        self.up_path_target = UpPathTargetNet(self.up_path)

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None,
                down_path_params=None,
                up_path_params=None):
        """
        Forward pass with optional parameter replacement for down and up paths

        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        down_path_params: Optional parameters for down path
        up_path_params: Optional parameters for up path
        output: (B,T,input_dim)
        """
        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)

        # Process timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # Get conditioning
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        # Down path (with optional parameter replacement)
        if down_path_params is not None:
            # Use target network with custom parameters
            x, h = self.down_path_target(sample, global_feature, down_path_params)
        else:
            x, h = self.down_path(sample, global_feature)

        # Up path (with optional parameter replacement)
        """ Detach the gradient of residual h """
        if up_path_params is not None:
            # Use target network with custom parameters
            x = self.up_path_target(x, [tensor.detach() for tensor in h], global_feature, up_path_params)
        else:
            x = self.up_path(x, [tensor.detach() for tensor in h], global_feature)

        # (B,C,T) -> (B,T,C)
        x = x.moveaxis(-1, -2)

        return x


# 5. Policy class for managing hypernetwork-generated parameters
class UNetPolicy(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8):
        super().__init__()

        self.unet = ConditionalUnet1D(
            input_dim, global_cond_dim, diffusion_step_embed_dim,
            down_dims, kernel_size, n_groups
        )

        n_params = sum(p.numel() for p in self.unet.parameters())
        print(f"UNetPolicy parameters: {n_params / 1e6:.2f}M")

    def forward(self, sample, timestep, global_cond=None,
                down_path_params=None, up_path_params=None):
        """
        Forward pass with optional parameter replacement

        This allows a hypernetwork to generate parameters for the down_path
        and up_path components separately.
        """
        return self.unet(sample, timestep, global_cond,
                         down_path_params, up_path_params)

    def get_down_path_parameters(self):
        """Returns parameter state dict for down path"""
        return {k: v for k, v in self.unet.down_path.state_dict().items()}

    def get_up_path_parameters(self):
        """Returns parameter state dict for up path"""
        return {k: v for k, v in self.unet.up_path.state_dict().items()}
