import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import clip
from torch.hub import load

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class DINOv2Encoder(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_dim=256,
            pool_feature_map=True,
            last_act=True,
            freeze_backbone=True,
            pretrained=True,
            model_name="dinov2_vitb14",
            device="cpu"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        if not pretrained:
            raise ValueError("DINOv2Encoder only supports pretrained models")
        self.backbone = load('facebookresearch/dinov2', model_name).to(device)

        # 处理输入通道数不是3的情况
        if in_channels != 3:
            original_proj = self.backbone.patch_embed.proj
            new_proj = nn.Conv2d(
                in_channels,
                original_proj.out_channels,
                kernel_size=original_proj.kernel_size,
                stride=original_proj.stride,
                padding=original_proj.padding,
                bias=original_proj.bias is not None
            )
            with torch.no_grad():
                new_proj.weight[:, :3, :, :] = original_proj.weight.clone()
                nn.init.kaiming_normal_(new_proj.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                if new_proj.bias is not None:
                    new_proj.bias.data = original_proj.bias.clone()
            self.backbone.patch_embed.proj = new_proj

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_dim = 768  # ViT-B/14的隐藏维度

        if pool_feature_map:
            self.fc = make_mlp(self.feature_dim, [out_dim], last_act=last_act)
        else:
            raise ValueError("DINOv2 ViT uses [CLS] token, please set pool_feature_map=True")

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(image)
        if features.dim() == 3:
            cls_features = features[:, 0, :]  # Extract CLS token if 3D
        elif features.dim() == 2:
            cls_features = features  # Use directly if 2D
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        output = self.fc(cls_features)
        return output

    def get_feature_extractor(self):
        return self.backbone

    def unfreeze_layers(self, start_layer=None):
        if start_layer is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        found_layer = False
        for name, module in self.backbone.named_modules():
            if start_layer in name:
                found_layer = True
            if found_layer:
                for param in module.parameters():
                    param.requires_grad = True