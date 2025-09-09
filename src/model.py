from __future__ import annotations
import math, torch
import torch.nn as nn
from .image_encoder import ViTEncoder
from .resnet_encoder import ResNetEncoder
from .text_encoder import TextTransformer

class CLIPModel(nn.Module):
    def __init__(self,
                 image_encoder_type="vit",
                 image_size=224, patch_size=32,
                 vision_width=768, vision_layers=12, vision_heads=12,
                 text_width=512, text_layers=12, text_heads=8,
                 context_length=77, vocab_size=49408,
                 embed_dim=512):
        super().__init__()
        
        if image_encoder_type.lower() == "vit":
            self.visual = ViTEncoder(
                image_size=image_size, 
                patch_size=patch_size, 
                width=vision_width, 
                layers=vision_layers, 
                heads=vision_heads, 
                embed_dim=embed_dim
            )
        elif image_encoder_type.lower() == "resnet":
            resnet_mapping = {
                (vision_layers, vision_width): "RN50",
                (23, vision_width): "RN101", 
                (10, 80): "RN50x4",
                (18, 96): "RN50x16",
                (36, 128): "RN50x64",
            }
            
            model_name = resnet_mapping.get((vision_layers, vision_width), "RN50")
            
            self.visual = ResNetEncoder(
                model_name=model_name,
                embed_dim=embed_dim,
                input_resolution=image_size
            )
        else:
            raise ValueError(f"Unknown image encoder type: {image_encoder_type}")
            
        self.text = TextTransformer(
            context_length=context_length, 
            vocab_size=vocab_size, 
            width=text_width, 
            layers=text_layers, 
            heads=text_heads, 
            embed_dim=embed_dim
        )
    
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

def clip_vit_b32(embed_dim=512):
    return CLIPModel(
        image_encoder_type="vit",
        image_size=224, patch_size=32,
        vision_width=768, vision_layers=12, vision_heads=12,
        embed_dim=embed_dim
    )

def clip_vit_b16(embed_dim=512):
    return CLIPModel(
        image_encoder_type="vit",
        image_size=224, patch_size=16,
        vision_width=768, vision_layers=12, vision_heads=12,
        embed_dim=embed_dim
    )

def clip_vit_l14(embed_dim=768):
    return CLIPModel(
        image_encoder_type="vit",
        image_size=224, patch_size=14,
        vision_width=1024, vision_layers=24, vision_heads=16,
        embed_dim=embed_dim
    )

def clip_resnet50(embed_dim=1024):
    return CLIPModel(
        image_encoder_type="resnet",
        vision_layers=12, vision_width=64,
        embed_dim=embed_dim
    )

def clip_resnet101(embed_dim=512):
    return CLIPModel(
        image_encoder_type="resnet", 
        vision_layers=23, vision_width=64,
        embed_dim=embed_dim
    )

def clip_resnet50x4(embed_dim=640):
    return CLIPModel(
        image_encoder_type="resnet",
        vision_layers=10, vision_width=80,
        embed_dim=embed_dim
    )