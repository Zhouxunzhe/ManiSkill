import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipProcessor


class SigLIP2Encoder(nn.Module):
    def __init__(self, out_dim=256, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.out_dim = out_dim

        self.visual_encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224") # google/siglip-so400m-patch14-384
        self.processor = SiglipProcessor.from_pretrained("google/siglip2-base-patch16-224")
        self.visual_feature_dim = self.visual_encoder.config.hidden_size  # 通常是 768 或 1152，取决于变体
        self.visual_projection = nn.Linear(self.visual_feature_dim, self.out_dim)
        self.to(device)

    def forward(self, image):
        image = image[:, :3, :, :]
        inputs = self.processor(images=image.to(self.device), return_tensors="pt").to(self.visual_encoder.device)
        with torch.no_grad():
            outputs = self.visual_encoder(**inputs)
        visual_feature = outputs.pooler_output  # (B*obs_horizon, hidden_size)
        visual_feature = self.visual_projection(visual_feature)

        return visual_feature


# Example usage
if __name__ == "__main__":
    # Example input: batch of 3 RGB images with size 224x224
    example_input = torch.randn(3, 3, 224, 224)

    # Initialize encoder with default small ViT model
    encoder = SigLIP2Encoder(out_dim=256)

    # Get embeddings
    output = encoder(example_input)
    print(output.shape)  # Should be [3, 256]