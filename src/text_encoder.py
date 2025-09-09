from __future__ import annotations
import torch
import torch.nn as nn
from .clip_layers import Transformer, build_causal_mask
from .bpe_tokenizer import SimpleTokenizer

class TextTransformer(nn.Module):
    def __init__(self, context_length=77, vocab_size=49408, width=512, layers=12, heads=8, embed_dim=512):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Parameter(torch.empty(1, context_length, width))
        self.transformer = Transformer(width=width, layers=layers, heads=heads)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        self._init_params()

        self.tokenizer = SimpleTokenizer(context_length=context_length)
        self.eot_token_id = self.tokenizer.eot_token

    def _init_params(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)
        nn.init.normal_(self.text_projection, std=0.02)

    def tokenize(self, texts):
        return self.tokenizer.tokenize(texts, self.context_length)

    def forward(self, text_tokens: torch.LongTensor) -> torch.Tensor:
        device = text_tokens.device
        x = self.token_embedding(text_tokens)
        x = x + self.pos_embed

        attn_mask = build_causal_mask(self.context_length, device=device)
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)

        with torch.no_grad():
            eot_mask = (text_tokens == self.eot_token_id) 
            fallback_idx = torch.full((text_tokens.size(0),), self.context_length-1, device=device)
            eot_idx = torch.where(eot_mask.any(dim=1), eot_mask.float().argmax(dim=1), fallback_idx)

        feat = x[torch.arange(x.size(0), device=device), eot_idx]
        feat = feat @ self.text_projection
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat