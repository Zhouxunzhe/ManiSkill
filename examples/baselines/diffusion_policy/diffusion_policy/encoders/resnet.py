import torch
import torch.nn as nn
import torchvision.models as models


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
            in_channels=3,
            out_dim=256,
            pool_feature_map=True,
            last_act=True,
            freeze_backbone=True,
            pretrained=True
    ):
        super().__init__()
        self.out_dim = out_dim

        # 加载预训练的ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # 处理输入通道数不是3的情况
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 移除原始ResNet的最后一个全连接层
        self.feature_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 添加自定义的投影层
        if pool_feature_map:
            # 特征图已经通过ResNet的全局平均池化被池化为(batch_size, feature_dim, 1, 1)
            self.fc = make_mlp(self.feature_dim, [out_dim], last_act=last_act)
        else:
            # 如果需要使用未池化的特征图，需要移除ResNet的平均池化层
            # 但这通常不是必要的，因为ResNet已经内置了平均池化
            raise ValueError("ResNet已经包含池化层，请设置pool_feature_map=True")

        self.reset_parameters()

    def reset_parameters(self):
        # 只重置我们添加的全连接层
        for name, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        # 通过ResNet主干网络
        features = self.backbone(image)
        # ResNet的输出已经是池化后的向量，形状为(batch_size, feature_dim, 1, 1)
        features = features.flatten(1)
        # 通过投影层
        output = self.fc(features)
        return output

    def get_feature_extractor(self):
        """返回特征提取器部分，用于迁移学习"""
        return self.backbone

    def unfreeze_layers(self, start_layer=None):
        """解冻指定层之后的所有层，用于微调"""
        if start_layer is None:
            # 解冻所有层
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        # 解冻指定层后的层
        found_layer = False
        for name, module in self.backbone.named_modules():
            if start_layer in name:
                found_layer = True
            if found_layer:
                for param in module.parameters():
                    param.requires_grad = True