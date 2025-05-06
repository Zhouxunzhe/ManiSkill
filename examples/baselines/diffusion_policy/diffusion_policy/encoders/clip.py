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

class CLIPEncoder(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_dim=256,
            pool_feature_map=True,
            last_act=True,
            freeze_backbone=True,
            pretrained=True,
            model_name="ViT-B/32",
            device="cpu"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        if not pretrained:
            raise ValueError("CLIPEncoder only supports pretrained models")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.backbone = self.model.visual

        # 处理输入通道数不是3的情况
        if in_channels != 3:
            original_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                if new_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.clone()
            self.backbone.conv1 = new_conv

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_dim = 512  # ViT-B/32的隐藏维度

        if pool_feature_map:
            self.fc = make_mlp(self.feature_dim, [out_dim], last_act=last_act)
        else:
            raise ValueError("CLIP ViT uses [CLS] token, please set pool_feature_map=True")

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
