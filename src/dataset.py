# src/dataset.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from .text_encoder import TextTransformer  # 토크나이저 공유용

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def build_image_transform(size=224):
    return T.Compose([
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

class ImageTextCsv(Dataset):
    """CSV: image_path, caption"""
    def __init__(self, csv_path, img_root, transform=None, tokenizer=None, context_length=77):
        self.df = pd.read_csv(csv_path)
        self.root = Path(img_root)
        self.t = transform or build_image_transform(size=224)
        # TextTokenizer: CLIP과 동일한 규칙
        self.tok_model = tokenizer or TextTransformer(context_length=context_length)
        self.context_length = context_length

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["image_path"]).convert("RGB")
        image = self.t(img)
        text = str(r["caption"])
        return {"image": image, "text": text}

def build_collate_clip(tokenizer, context_length=77):
    def _fn(batch):
        images = torch.stack([b["image"] for b in batch])
        texts  = [b["text"] for b in batch]
        tokens = tokenizer.tokenize(texts)  # (B,77)
        return {"images": images, "text_tokens": tokens}
    return _fn
