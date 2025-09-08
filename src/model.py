# src/model.py
from __future__ import annotations
import math, torch
import torch.nn as nn
from .image_encoder import ViTEncoder
from .text_encoder import TextTransformer

class CLIPModel(nn.Module):

    def __init__(self,
                 image_size=224, patch_size=32,
                 vision_width=768, vision_layers=12, vision_heads=12,
                 text_width=512, text_layers=12, text_heads=8,
                 context_length=77, vocab_size=49408,
                 embed_dim=512):
        super().__init__()
        self.visual = ViTEncoder(image_size, patch_size, vision_width, vision_layers, vision_heads, embed_dim)
        self.text   = TextTransformer(context_length, vocab_size, text_width, text_layers, text_heads, embed_dim)
    
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    @property
    def temperature(self):
        return self.logit_scale.exp()

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, text_tokens):
        return self.text(text_tokens)

    def forward(self, images, text_tokens):
        img = self.encode_image(images)        
        txt = self.encode_text(text_tokens)    
        logits = self.temperature * (img @ txt.t())  
        return {
            "image_embeds": img, "text_embeds": txt,
            "logits_per_image": logits, "logits_per_text": logits.t(),
        }
