# src/image_encoder.py
from __future__ import annotations
import math, torch
import torch.nn as nn
from .clip_layers import Transformer

class ViTEncoder(nn.Module):

    def __init__(self, image_size=224, patch_size=32, width=768, layers=12, heads=12, embed_dim=512):
        super().__init__()
        assert image_size % patch_size == 0
        grid = image_size // patch_size
        self.seq_len = grid * grid + 1  
        self.width = width

        self.conv = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed   = nn.Parameter(torch.empty(1, self.seq_len, width))

        self.transformer = Transformer(width=width, layers=layers, heads=heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, embed_dim))  

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)
        nn.init.normal_(self.proj, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
   
        x = self.conv(images)            
        x = x.flatten(2).transpose(1, 2)  
        cls = self.class_token.expand(x.size(0), -1, -1)   
        x = torch.cat([cls, x], dim=1)   
        x = x + self.pos_embed          
        x = self.transformer(x)     
        x = self.ln_post(x[:, 0, :])     
        x = x @ self.proj                 
        x = x / x.norm(dim=-1, keepdim=True)  
        return x
