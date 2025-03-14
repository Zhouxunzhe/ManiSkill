import torch
import torch.nn as nn
import clip
from torchvision.transforms import Resize, Normalize, Compose

class CLIPEncoder(nn.Module):
    def __init__(self, out_dim=256, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load(model_name, device=self.device)
        self.out_dim = out_dim
        self.preprocess = Compose([
            Resize((224, 224)),  # 确保图像尺寸正确
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.fc = nn.Identity() if self.clip_model.visual.output_dim == out_dim else nn.Linear(self.clip_model.visual.output_dim, out_dim)

    def reset_parameters(self):
        if isinstance(self.fc, nn.Linear):
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

    def forward(self, image):
        image = self.preprocess(image.to(self.device))
        with torch.no_grad():
            x = self.clip_model.encode_image(image)
        x = x.to(self.fc.weight.dtype)
        return self.fc(x)

# Example usage
if __name__ == "__main__":
    # Example input: batch of 3 RGB images with size 224x224
    example_input = torch.randn(3, 3, 224, 224)
    encoder = CLIPEncoder(out_dim=256)
    output = encoder(example_input)
    print(output.shape)  # Should be [3, 256]
