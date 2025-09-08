# src/bpe_tokenizer.py
from __future__ import annotations
import gzip
import html
import os
from functools import lru_cache
from typing import List, Union
import ftfy
import regex as re
import torch

# BPE 토크나이저 (CLIP 공식 구현 기반)
@lru_cache()
def default_bpe():
    """기본 BPE vocab 파일 경로 반환 (간소화된 버전)"""
    # 실제로는 OpenAI의 BPE vocab을 사용하지만, 여기서는 간단한 매핑 생성
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    """
    바이트를 유니코드 문자로 매핑하는 딕셔너리 반환
    UTF-8 바이트를 유니코드 문자로 안전하게 변환
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """단어에서 연속된 문자 쌍을 반환"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def whitespace_clean(text):
    """공백 정규화"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = None, context_length: int = 77):
        self.context_length = context_length
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 간단한 BPE vocab 생성 (실제로는 파일에서 로드)
        self.encoder = self._build_simple_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # BPE 병합 규칙 (간소화)
        self.bpe_ranks = self._build_simple_bpe_ranks()
        
        # 정규식 패턴 (CLIP과 동일)
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        
        # 특수 토큰
        vocab_size = len(self.encoder)
        self.encoder["<|startoftext|>"] = vocab_size
        self.encoder["<|endoftext|>"] = vocab_size + 1
        self.decoder[vocab_size] = "<|startoftext|>"
        self.decoder[vocab_size + 1] = "<|endoftext|>"
        
        self.sot_token = vocab_size
        self.eot_token = vocab_size + 1

    def _build_simple_vocab(self):
        """간단한 vocab 생성 (실제로는 BPE vocab 파일 사용)"""
        vocab = {}
        
        # 기본 바이트 레벨 토큰
        for i, byte_char in enumerate(self.byte_encoder.values()):
            vocab[byte_char] = i
            
        # 일반적인 BPE 병합 추가 (간소화)
        common_merges = [
            'er', 'in', 'on', 'an', 'ed', 'nd', 'ha', 'en', 'he', 'to', 'or',
            'th', 'te', 're', 'at', 'es', 'ng', 'al', 'it', 'ar', 'ou', 'as',
            'le', 'is', 'et', 'ti', 've', 'll', 'nt', 'st', 'se', 'ly', 'me',
            'de', 'pe', 'ro', 'ne', 'ta', 'la', 'ec', 'ri', 'ac', 'be', 'ur',
            'ma', 'si', 'li', 'ra', 'co', 'ca', 'un', 'el', 'tr', 'ch', 'pr'
        ]
        
        idx = len(vocab)
        for merge in common_merges:
            if merge not in vocab:
                vocab[merge] = idx
                idx += 1
                
        return vocab

    def _build_simple_bpe_ranks(self):
        """간단한 BPE 병합 순위 생성"""
        ranks = {}
        # 간소화된 병합 규칙
        common_pairs = [
            ('t', 'h'), ('h', 'e'), ('i', 'n'), ('e', 'r'), ('a', 'n'),
            ('r', 'e'), ('e', 'd'), ('o', 'n'), ('e', 's'), ('n', 't'),
            ('e', 'n'), ('t', 'i'), ('o', 'r'), ('t', 'e'), ('a', 't'),
            ('s', 'e'), ('n', 'd'), ('o', 'u'), ('t', 'o'), ('h', 'a'),
            ('a', 'r'), ('o', 'u'), ('i', 't'), ('v', 'e'), ('w', 'a'),
            ('a', 'l'), ('l', 'l'), ('i', 's'), ('a', 's'), ('e', 't'),
        ]
        
        for i, pair in enumerate(common_pairs):
            ranks[pair] = i
            
        return ranks

    def bpe(self, token):
        """BPE 인코딩 수행"""
        if token in self.cache:
            return self.cache[token]
            
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """텍스트를 토큰 ID로 인코딩"""
        bpe_tokens = []
        # 텍스트 전처리
        text = whitespace_clean(ftfy.fix_text(text))
        text = html.unescape(text)
        
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """토큰 ID를 텍스트로 디코딩"""
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace")
        return text

    def tokenize(self, texts: Union[str, List[str]], context_length: int = None) -> torch.LongTensor:
        """
        텍스트를 토큰화하여 텐서로 반환
        CLIP 공식 구현과 동일한 방식
        """
        if isinstance(texts, str):
            texts = [texts]
            
        context_length = context_length or self.context_length
        
        # 캐시 초기화 (메모리 효율성)
        if not hasattr(self, 'cache'):
            self.cache = {}
            
        all_tokens = []
        for text in texts:
            tokens = [self.sot_token] + self.encode(text) + [self.eot_token]
            
            # 길이 조정
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token  # 마지막은 항상 EOT
            else:
                tokens = tokens + [0] * (context_length - len(tokens))  # 패딩
                
            all_tokens.append(tokens)
            
        return torch.tensor(all_tokens, dtype=torch.long)

# 전역 토크나이저 인스턴스
_tokenizer = SimpleTokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """편의를 위한 전역 토크나이저 함수"""
    return _tokenizer.tokenize(texts, context_length) 