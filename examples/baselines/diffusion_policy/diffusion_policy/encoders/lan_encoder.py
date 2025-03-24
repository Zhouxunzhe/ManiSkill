import torch
import torch.nn as nn


class LanguageEncoder(nn.Module):
    def __init__(self, encoder_type, output_dim=256):
        super().__init__()
        self.encoder_type = encoder_type
        self.output_dim = output_dim

        if encoder_type == "encoder_only":
            # RoBERTa works well for robotic manipulation as it has good semantic understanding
            from transformers import RobertaModel, RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaModel.from_pretrained("roberta-base")
            self.projection = nn.Linear(self.model.config.hidden_size, output_dim)

        elif encoder_type == "tokenizer_only":
            # Simple tokenizer with learned embeddings
            from transformers import RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, 768)
            self.projection = nn.Linear(768, output_dim)

        elif encoder_type == "encoder_decoder":
            # T5 works well for instruction following tasks
            from transformers import T5Model, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.model = T5Model.from_pretrained("t5-base")
            self.projection = nn.Linear(self.model.config.d_model, output_dim)

        elif encoder_type == "tokenizer_decoder":
            # Use T5 tokenizer with decoder only
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.projection = nn.Linear(self.model.config.d_model, output_dim)

        elif encoder_type == "encoder_ffn":
            # CLIP text encoder works well for cross-modal alignment
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.ffn = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )

        elif encoder_type == "tokenizer_ffn":
            # Simple tokenizer with FFN
            from transformers import RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, 768)
            self.ffn = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )

    def forward(self, text_instructions):
        if self.encoder_type == "encoder_only":
            inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.model.device)
            outputs = self.model(**inputs)
            # Use [CLS] token representation
            return self.projection(outputs.last_hidden_state[:, 0])

        elif self.encoder_type == "tokenizer_only":
            inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.embedding.weight.device)
            # Average the embeddings
            embeds = self.embedding(inputs.input_ids)
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1)
            return self.projection(pooled)

        elif self.encoder_type == "encoder_decoder":
            inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.model.device)
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            # Use encoder's last hidden state pooled
            encoder_hidden = outputs.last_hidden_state.mean(dim=1)
            return self.projection(encoder_hidden)

        elif self.encoder_type == "tokenizer_decoder":
            inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.model.device)
            # Generate a single decoder step to get decoder features
            decoder_input_ids = torch.zeros((inputs.input_ids.shape[0], 1), dtype=torch.long, device=self.model.device)
            outputs = self.model(input_ids=inputs.input_ids,
                                 attention_mask=inputs.attention_mask,
                                 decoder_input_ids=decoder_input_ids)
            # Use decoder's representation
            return self.projection(outputs.last_hidden_state.mean(dim=1))

        elif self.encoder_type == "encoder_ffn":
            inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.model.device)
            outputs = self.model(**inputs)
            # Apply FFN to pooled output
            return self.ffn(outputs.pooler_output)

        elif self.encoder_type == "tokenizer_ffn":
            inputs = self.tokenizer(text_instructions, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.embedding.weight.device)
            # Average the embeddings
            embeds = self.embedding(inputs.input_ids)
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1)
            return self.ffn(pooled)