from typing import List

import torch
import torch.nn as nn
import clip 
from mode.models.networks.clip import build_model, load_clip, tokenize
# from transformers import (
#     AutoProcessor,
#     AutoModel,
#     SiglipProcessor,
#     SiglipModel
# )


class LangClip(nn.Module):
    def __init__(self, freeze_backbone: bool = True, model_name: str = "RN50"):
        super(LangClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model
        print(f"loading language CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        model, _ = load_clip(model_name, device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)

    def forward(self, x: List) -> torch.Tensor:
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            tokens = tokens.long()  # Ensure tokens are of type Long
            # print('token dtype:', tokens.dtype)
            # print('clip_rn50 dtype:', self.clip_rn50.dtype)
            emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)


class LangClip2(nn.Module):

    def __init__(self, freeze_backbone: bool = True, model_name: str = "ViT-B/32") -> None:
        super().__init__()
        clip_model, clip_preprocess = clip.load(model_name)
        # freeze clip
        if freeze_backbone:
            for _, param in clip_model.named_parameters():
                param.requires_grad = False

        self.text_tokenizer=clip.tokenize
        self.text_encoder=clip_model   

    def forward(self, x: List) -> torch.Tensor:
        inputs = self.text_tokenizer(x)
        device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder.encode_text(inputs.to(device))
        return encoder_hidden_states

