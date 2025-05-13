import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, SiglipTextModel

class LanguageEncoder(nn.Module):
    def __init__(self, vision_model=None, processor=None,
                 encoder_type="encoder_only", output_dim=256, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.encoder_type = encoder_type
        self.output_dim = output_dim

        # Load T5 (text encoder/decoder) and SigLIP (vision encoder)
        if encoder_type in ["encoder_only", "encoder_ffn", "tokenizer_only", "tokenizer_ffn"]:
            self.text_model = SiglipTextModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
            self.text_feature_dim = self.text_model.config.hidden_size  # e.g., 768 for t5-base
        else:  # For decoder-involved types
            self.text_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.text_feature_dim = self.text_model.config.d_model  # e.g., 768 for t5-base

        # Load SigLIP for vision encoding
        self.vision_model = vision_model
        self.processor = processor
        self.visual_feature_dim = self.vision_model.config.hidden_size  # e.g., 768 for SigLIP base

        # Projection layers based on encoder_type
        if encoder_type == "encoder_only":
            self.projection = nn.Linear(self.text_feature_dim, output_dim).to(device)
        elif encoder_type == "tokenizer_only":
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.text_feature_dim).to(device)
            self.projection = nn.Linear(self.text_feature_dim, output_dim).to(device)
        elif encoder_type in ["encoder_decoder", "tokenizer_decoder"]:
            self.projection = nn.Linear(self.text_feature_dim, output_dim).to(device)
            self.visual_projection = nn.Linear(self.visual_feature_dim, self.text_feature_dim).to(device)
        elif encoder_type == "encoder_ffn":
            input_dim = self.text_feature_dim
            self.visual_projection = nn.Linear(self.visual_feature_dim, self.text_feature_dim).to(device)
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            ).to(device)
        elif encoder_type == "tokenizer_ffn":
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.text_feature_dim).to(device)
            input_dim = self.text_feature_dim
            self.visual_projection = nn.Linear(self.visual_feature_dim, self.text_feature_dim).to(device)
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            ).to(device)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def forward(self, text_instructions, obs_horizon, image=None):
        # Tokenize text with T5 tokenizer
        inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                max_length=128).to(self.device)

        # Process image with SigLIP if provided
        visual_features = None
        if image is not None:
            # Select only the first three channels
            batch_size = image.shape[0]
            flatten_img = image.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
            flatten_img = flatten_img[:, :3, :, :]
            vision_inputs = self.processor(images=flatten_img.to(self.device), return_tensors="pt").to(self.device)
            # Encode image using the shared vision tower
            with torch.no_grad():
                vision_outputs = self.vision_model(**vision_inputs)
            visual_features = vision_outputs.pooler_output  # (B*obs_horizon, hidden_size)
            visual_features = visual_features.reshape(
                batch_size, obs_horizon, visual_features.shape[1]
            )  # (B, obs_horizon, D)

        # Forward pass based on encoder_type
        if self.encoder_type == "encoder_only":
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            text_features = outputs.last_hidden_state[:, 0]  # First token (CLS-like)
            return self.projection(text_features)
        elif self.encoder_type == "tokenizer_only":
            embeds = self.embedding(inputs.input_ids)
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1)
            return self.projection(pooled)
        elif self.encoder_type in ["encoder_decoder", "tokenizer_decoder"]:
            if visual_features is not None and hasattr(self, "visual_projection"):
                visual_features = self.visual_projection(visual_features)
                text_embeds = self.text_model.get_input_embeddings()(inputs.input_ids)
                B = len(text_instructions)
                obs_horizon = visual_features.size(1)
                seq_len = inputs.attention_mask.size(1)
                visual_mask = torch.ones(B, obs_horizon).to(inputs.attention_mask.device)
                combined_mask = torch.cat([visual_mask, inputs.attention_mask], dim=1)
                combined_embeds = torch.cat([visual_features, text_embeds], dim=1)
                pad_token_id = self.text_model.config.pad_token_id
                decoder_input_ids = torch.full((B, 1), pad_token_id).to(self.device)
                outputs = self.text_model(inputs_embeds=combined_embeds, attention_mask=combined_mask, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                return self.projection(outputs.decoder_hidden_states[-1]).mean(dim=1)
            else:
                outputs = self.text_model(**inputs, decoder_input_ids=inputs.input_ids, output_hidden_states=True)
                return self.projection(outputs.decoder_hidden_states[-1]).mean(dim=1)
        elif self.encoder_type == "encoder_ffn":
            outputs = self.text_model(**inputs)
            text_features = outputs.last_hidden_state  # [batch_size, seq_len, text_feature_dim]
            if visual_features is not None and hasattr(self, "visual_projection"):
                visual_features = self.visual_projection(visual_features)  # [batch_size, obs_horizon, text_feature_dim]
                combined_features = torch.cat([text_features, visual_features], dim=1)  # [batch_size, seq_len + obs_horizon, text_feature_dim]
            else:
                combined_features = text_features
            transformed = self.ffn(combined_features)  # [batch_size, seq_len + obs_horizon, output_dim] 或 [batch_size, seq_len, output_dim]
            pooled = transformed.mean(dim=1)  # [batch_size, output_dim]
            return pooled
        elif self.encoder_type == "tokenizer_ffn":
            embeds = self.embedding(inputs.input_ids)
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1)
            if visual_features is not None and hasattr(self, "visual_projection"):
                visual_features = self.visual_projection(visual_features)
                combined_features = torch.cat([pooled, visual_features], dim=1)
                return self.ffn(combined_features)
            return self.ffn(pooled)

# Example usage
if __name__ == "__main__":
    model = LanguageEncoder(t5_model_name="t5-base", siglip_model_name="google/siglip2-base-patch16-224",
                            encoder_type="encoder_ffn", output_dim=256)
    text = ["This is a test sentence."]
    # Assume `image` is a list of PIL image or tensor
    output = model(text_instructions=text, image=None)
    print(output.shape)  # Should be [batch_size, output_dim]