import torch
import torch.nn as nn
from torchvision.transforms import Resize, Normalize, Compose


class DINOv2Encoder(nn.Module):
    def __init__(self, out_dim=256, model_name="dinov2_vitb14", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device

        # Load DINOv2 model
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()  # Set to evaluation mode

        # Get the output dimension of the DINOv2 model
        self.dino_dim = self.dino_model.embed_dim

        # Output projection if needed
        self.out_dim = out_dim
        self.fc = nn.Identity() if self.dino_dim == out_dim else nn.Linear(self.dino_dim, out_dim)

        # DINOv2 expects images normalized this way
        self.preprocess = Compose([
            Resize((224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        if isinstance(self.fc, nn.Linear):
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

    def forward(self, image):
        image = self.preprocess(image.to(self.device))
        with torch.no_grad():
            # DINOv2 forward pass to get embeddings
            x = self.dino_model.forward_features(image)
            # Get CLS token output
            x = x['x_norm_clstoken']
            x = x.to(self.fc.weight.dtype if isinstance(self.fc, nn.Linear) else x.dtype)

        return self.fc(x)


# Example usage
if __name__ == "__main__":
    # Example input: batch of 3 RGB images with size 224x224
    example_input = torch.randn(3, 3, 224, 224)

    # Initialize encoder with default small ViT model
    encoder = DINOv2Encoder(out_dim=256)

    # Get embeddings
    output = encoder(example_input)
    print(output.shape)  # Should be [3, 256]