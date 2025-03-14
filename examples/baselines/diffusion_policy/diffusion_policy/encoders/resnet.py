import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class ResNetEncoder(nn.Module):
    def __init__(
            self,
            out_dim=256,
            model_type='resnet50',  # 'resnet18', 'resnet34', 'resnet50', 'resnet101'
            pretrained=True,
            pool_feature_map=False,
            last_act=True,
    ):
        super().__init__()

        # Load ResNet model based on type
        if model_type == 'resnet18':
            self.backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            resnet_dim = 512
        elif model_type == 'resnet34':
            self.backbone = resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            resnet_dim = 512
        elif model_type == 'resnet50':
            self.backbone = resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            resnet_dim = 2048
        elif model_type == 'resnet101':
            self.backbone = resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            resnet_dim = 2048
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Remove the final classification layer
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))

        # Freeze ResNet parameters if using pretrained model
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Define projection layer from ResNet dimension to desired output dimension
        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(resnet_dim, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(resnet_dim, [out_dim], last_act=last_act)

        self.out_dim = out_dim

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        # Ensure input is properly normalized for ResNet (224x224)
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = F.interpolate(image, size=(224, 224), mode='bicubic')

        # Get ResNet features
        x = self.backbone(image)

        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x