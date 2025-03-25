import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, vision_model, processor, out_dim=256, encoder_type="encoder_only",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        self.encoder_type = encoder_type

        # Use the pre-loaded vision model and processor
        self.model = vision_model  # Reference, not a new instance
        self.processor = processor  # Reference, not a new instance
        self.visual_feature_dim = self.model.config.hidden_size

        # Projection layer based on encoder_type
        if encoder_type in ["encoder_ffn", "tokenizer_ffn"]:
            self.projection = nn.Sequential(
                nn.Linear(self.visual_feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, out_dim)
            )
        else:
            self.projection = nn.Linear(self.visual_feature_dim, out_dim)
        self.projection.to(device)

    def forward(self, image):
        # Select only the first three channels
        image = image[:, :3, :, :]
        inputs = self.processor(images=image.to(self.device), return_tensors="pt").to(self.device)
        # Encode image using the shared vision tower
        with torch.no_grad():
            outputs = self.model(**inputs)
        visual_feature = outputs.pooler_output  # (B*obs_horizon, hidden_size)
        visual_feature = self.projection(visual_feature)
        return visual_feature