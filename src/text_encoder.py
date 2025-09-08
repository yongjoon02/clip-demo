# src/text_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
from .clip_layers import Transformer, build_causal_mask

class SimpleTokenizer:
    """간단한 자체 토크나이저 (실제 CLIP BPE 대신 단순화된 버전)"""
    
    def __init__(self, vocab_size=49408, context_length=77):
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        # 특수 토큰들
        self.sot_token = 49406  # start of text
        self.eot_token = 49407  # end of text  
        self.pad_token = 0      # padding
        
        # 간단한 단어-토큰 매핑 (실제로는 BPE 사용)
        # 여기서는 ASCII 기반 간단 매핑
        self.char_to_token = {chr(i): i for i in range(32, 127)}  # 기본 ASCII
        self.char_to_token.update({
            ' ': 32, '.': 46, ',': 44, '!': 33, '?': 63,
            'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101,
            'f': 102, 'g': 103, 'h': 104, 'i': 105, 'j': 106,
            'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111,
            'p': 112, 'q': 113, 'r': 114, 's': 115, 't': 116,
            'u': 117, 'v': 118, 'w': 119, 'x': 120, 'y': 121, 'z': 122
        })
        
    def tokenize(self, texts):
        """텍스트 리스트를 토큰 텐서로 변환"""
        if isinstance(texts, str):
            texts = [texts]
            
        batch_tokens = []
        for text in texts:
            # 텍스트를 소문자로 변환
            text = text.lower().strip()
            
            # 토큰 시퀀스 생성
            tokens = [self.sot_token]  # 시작 토큰
            
            for char in text[:self.context_length-2]:  # SOT, EOT 공간 확보
                token = self.char_to_token.get(char, 1)  # unknown token = 1
                tokens.append(token)
                
            tokens.append(self.eot_token)  # 끝 토큰
            
            # 패딩
            while len(tokens) < self.context_length:
                tokens.append(self.pad_token)
                
            # 길이 맞추기
            tokens = tokens[:self.context_length]
            batch_tokens.append(tokens)
            
        return torch.tensor(batch_tokens, dtype=torch.long)

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

        # 자체 토크나이저 사용
        self.tokenizer = SimpleTokenizer(vocab_size, context_length)
        self.eot_token_id = self.tokenizer.eot_token

    def _init_params(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)
        nn.init.normal_(self.text_projection, std=0.02)

    def tokenize(self, texts):
        """텍스트를 토큰으로 변환"""
        return self.tokenizer.tokenize(texts)

    def forward(self, text_tokens: torch.LongTensor) -> torch.Tensor:
        device = text_tokens.device
        x = self.token_embedding(text_tokens)  # (B, L, width)
        x = x + self.pos_embed                 # positional encoding

        # Causal attention mask
        attn_mask = build_causal_mask(self.context_length, device=device)
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)                  # final layer norm

        # EOT 토큰 위치 찾기 (마지막 유효 토큰)
        with torch.no_grad():
            eot_mask = (text_tokens == self.eot_token_id) 
            # EOT가 없으면 마지막 토큰 사용
            fallback_idx = torch.full((text_tokens.size(0),), self.context_length-1, device=device)
            eot_idx = torch.where(eot_mask.any(dim=1), eot_mask.float().argmax(dim=1), fallback_idx)

        # EOT 위치의 특성 추출
        feat = x[torch.arange(x.size(0), device=device), eot_idx] # (B, width)
        feat = feat @ self.text_projection                         # (B, embed_dim)
        feat = feat / feat.norm(dim=-1, keepdim=True)               # L2 normalize
        return feat
