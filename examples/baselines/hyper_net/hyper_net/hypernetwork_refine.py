import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import List, Dict


class MLP(nn.Module):
    """
    Memory-efficient MLP implementation that uses weight sharing and reduced parameters
    """

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        # Use a single hidden layer with projection for multiple layer effect
        if num_layers == 1:
            self.proj_in = nn.Linear(in_dim, out_dim)
            self.has_hidden = False
        else:
            self.proj_in = nn.Linear(in_dim, hidden_dim)
            self.proj_hidden = nn.Linear(hidden_dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, out_dim)
            self.has_hidden = True

    def forward(self, x):
        if not self.has_hidden:
            return self.proj_in(x)

        x = self.activation(self.proj_in(x))
        # Efficiently apply middle layers through repeated application of the same weights
        for _ in range(self.num_layers - 2):
            x = self.activation(self.proj_hidden(x))
        return self.proj_out(x)


class OptSubBlock(nn.Module):
    """
    Memory-efficient version of OptSubBlock with shared parameters
    """

    def __init__(self, ftask_dim, in_dim, out_dim, weight_dim, hidden_dim, num_layers, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim

        # Reduced number of parameters using simpler architecture
        self.forward_net = MLP(in_dim + weight_dim, out_dim, hidden_dim, num_layers)

        # Direct prediction of update instead of factorization for simplicity and reliability
        self.update_predictor = MLP(in_dim + weight_dim + out_dim, weight_dim, hidden_dim, num_layers)

    def pseudo_forward(self, z_in, weight_emb):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = weight_emb.expand(z_in.shape[0], -1)  # More memory efficient than repeat

        combined = torch.cat([z_in, weight_emb], dim=-1)
        out = self.forward_net(combined)
        return out

    def pseudo_backward(self, z_in, weight_emb, dl_dout):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = weight_emb.expand(z_in.shape[0], -1)

        # Direct prediction of the weight update
        combined = torch.cat([z_in, weight_emb, dl_dout], dim=-1)
        weight_update = self.update_predictor(combined)

        # Return placeholder for dl_din (not used)
        dl_din = torch.zeros_like(z_in)

        return dl_din, weight_update


class LightweightOptBlock(nn.Module):
    """
    Memory-efficient version of OptBlock with shared computation
    """

    def __init__(self, target_net, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dims = target_net.get_in_dims()
        self.out_dims = target_net.get_out_dims()
        self.weight_dim = weight_dim

        # Create sub-blocks with reduced parameters
        self.opt_sub_blocks = nn.ModuleList([
            OptSubBlock(
                ftask_dim, in_dim, out_dim, weight_dim,
                deriv_hidden_dim, driv_num_layers
            )
            for in_dim, out_dim in zip(self.in_dims, self.out_dims)
        ])

        # Shared computation for input generation and loss gradients
        self.forward_in = MLP(ftask_dim, self.in_dims[0], deriv_hidden_dim, driv_num_layers)
        self.dloss_dout = MLP(ftask_dim + self.out_dims[-1], self.out_dims[-1], deriv_hidden_dim,
                                             driv_num_layers)

    def forward(self, ftask, weight_embs):
        # Generate initial input from task features
        z_ins = [self.forward_in(ftask)]

        # Forward pass through each sub-block
        for idx, (weight_emb, opt_sub_block) in enumerate(zip(weight_embs, self.opt_sub_blocks)):
            next_z = opt_sub_block.pseudo_forward(z_ins[-1], weight_emb)
            z_ins.append(next_z)

        # Compute gradients for backward pass
        dl_douts = [self.dloss_dout(torch.cat([ftask, z_ins[-1]], dim=-1))]

        # Backward pass to compute weight updates
        weight_updates = []
        for opt_sub_block, z_in, weight_emb in reversed(list(zip(self.opt_sub_blocks, z_ins[:-1], weight_embs))):
            dl_din, weight_update = opt_sub_block.pseudo_backward(z_in, weight_emb, dl_douts[-1])
            dl_douts.append(dl_din)
            weight_updates.append(weight_update)

        # Reverse the list to match forward order
        return list(reversed(weight_updates))


class OptimizedHypernet(nn.Module):
    def __init__(self, target_net,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 codec_hidden_dim, codec_num_layers,
                 num_layers, lr=None, *args, **kwargs):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.num_layers = num_layers

        # Create parameter references for each submodule
        self.submodule_parameter_shapes = []
        self.submodule_parameter_sizes = []
        self.submodule_parameter_names = []

        for module in target_net.get_submodules():
            shapes = {}
            names = []
            sizes = []
            for name, param in module.named_parameters():
                shapes[name] = param.shape
                names.append(name)
                sizes.append(param.numel())
            self.submodule_parameter_shapes.append(shapes)
            self.submodule_parameter_names.append(names)
            self.submodule_parameter_sizes.append(sizes)

        # Simplified encoder/decoder architecture
        self.encoders = nn.ModuleList([
            MLP(
                sum(sizes),
                weight_dim,
                codec_hidden_dim // 2,  # Reduced hidden dimension
                max(2, codec_num_layers - 1)  # Reduced number of layers
            )
            for sizes in self.submodule_parameter_sizes
        ])

        self.decoders = nn.ModuleList([
            MLP(
                weight_dim,
                sum(sizes),
                codec_hidden_dim // 2,
                max(2, codec_num_layers - 1)
            )
            for sizes in self.submodule_parameter_sizes
        ])

        # Memory-efficient optimization block
        self.opt_block = LightweightOptBlock(
            target_net, ftask_dim, weight_dim,
            deriv_hidden_dim // 2,  # Reduced hidden dimension
            max(2, driv_num_layers - 1),  # Reduced number of layers
            *args, **kwargs
        )

        # Lightweight task integration
        self.task_projection = nn.Linear(ftask_dim, weight_dim)

        # Simple modulation mechanism
        self.modulation = nn.Linear(ftask_dim, weight_dim)

        # Layer norm for stability (shared across submodules to save memory)
        self.layer_norm = nn.LayerNorm(weight_dim)

        # Learnable step sizes (scalar per iteration rather than per-dimension)
        if lr is not None:
            self.dynamic_lrs = nn.Parameter(torch.ones(num_layers) * lr)
        else:
            self.dynamic_lrs = nn.Parameter(torch.ones(num_layers) * -0.01)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MemoryEfficientHypernet parameters: {n_params / 1e6:.2f}M")

    def encode_parameters(self, module_idx, param_dict):
        """Encode parameters to weight embedding efficiently"""
        # Extract and flatten parameters
        param_names = self.submodule_parameter_names[module_idx]

        # Concatenate all parameters into a single vector
        flattened = []
        for name in param_names:
            flattened.append(param_dict[name].reshape(-1))

        param_vector = torch.cat(flattened)

        # Add batch dimension if needed
        if param_vector.dim() == 1:
            param_vector = param_vector.unsqueeze(0)

        # Encode to embedding
        return self.encoders[module_idx](param_vector)

    def decode_parameters(self, module_idx, weight_emb):
        """Decode weight embedding to parameters efficiently"""
        # Decode to flattened parameter vector
        param_vector = self.decoders[module_idx](weight_emb)

        # Reconstruct parameter dictionary
        param_dict = {}
        param_names = self.submodule_parameter_names[module_idx]
        param_shapes = self.submodule_parameter_shapes[module_idx]
        param_sizes = self.submodule_parameter_sizes[module_idx]

        # Split the vector based on parameter sizes
        start_idx = 0
        for i, name in enumerate(param_names):
            shape = param_shapes[name]
            size = param_sizes[i]

            # Extract and reshape parameter
            param = param_vector[:, start_idx:start_idx + size].reshape(-1, *shape)
            param_dict[name] = param

            start_idx += size

        return param_dict

    def forward_blocks(self, ftask):
        """Optimized forward pass through blocks with memory efficiency"""
        # Get initial parameters
        param_dicts = [dict(module.named_parameters()) for module in self.target_net.get_submodules()]

        # Encode parameters (done once to save memory)
        weight_embs = [self.encode_parameters(i, param_dict) for i, param_dict in enumerate(param_dicts)]

        # Ensure batch dimension matches ftask
        if ftask.shape[0] > 1 and weight_embs[0].shape[0] == 1:
            weight_embs = [w.expand(ftask.shape[0], -1) for w in weight_embs]

        # Task integration
        task_proj = self.task_projection(ftask)
        weight_embs = [w + task_proj for w in weight_embs]

        # Track weight dictionaries for each iteration
        final_weight_dicts = []

        # Iterative optimization
        for i in range(self.num_layers):
            # Task modulation (simpler version)
            mod = torch.sigmoid(self.modulation(ftask))
            modulated_weight_embs = [w * mod for w in weight_embs]

            # Compute weight updates
            weight_upd_embs = self.opt_block(ftask, modulated_weight_embs)

            # Apply updates with dynamic learning rates (normalize each update separately)
            for j in range(len(weight_embs)):
                # Make sure the update has the correct shape for LayerNorm
                normalized_update = self.layer_norm(weight_upd_embs[j])
                weight_embs[j] = weight_embs[j] + self.dynamic_lrs[i] * normalized_update

            # Decode parameters and merge
            param_dicts = [self.decode_parameters(j, w) for j, w in enumerate(weight_embs)]
            weight_dict = self.target_net.merge_submodule_weights(param_dicts)
            final_weight_dicts.append(weight_dict)

        return final_weight_dicts

    def forward(self, ftask, inputs, early_sup=False):
        """Memory-efficient forward pass"""
        # Get optimized weights
        final_weight_dicts = self.forward_blocks(ftask)

        # Apply weights to target network
        if early_sup:
            results = []
            for weight_dict in final_weight_dicts:
                result = torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(
                    self.target_net, weight_dict, inputs
                )
                results.append(result)
            return torch.stack(results)
        else:
            return torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(
                self.target_net, final_weight_dicts[-1], inputs
            )
