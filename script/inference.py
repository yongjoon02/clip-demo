# script/inference.py
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# CUDA warnings 무시
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.model import CLIPModel
from src.dataset import build_image_transform, IMG_EXTS

class ImageFolderDataset:
    """간단한 이미지 폴더 데이터셋"""
    def __init__(self, root: str, transform):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, i):
        from PIL import Image
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        return {"image": self.transform(img), "path": str(p)}

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="ViT-B/32", help="모델 크기 (표기용)")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--classes", required=True, help='comma-separated: "cat,dog,car"')
    ap.add_argument("--batch-size", type=int, default=64)
    # 모델 아키텍처 설정
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=512)
    return ap.parse_args()

def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 자체 구현 CLIP 모델 생성
    model = CLIPModel(
        image_size=args.image_size,
        patch_size=args.patch_size, 
        embed_dim=args.embed_dim
    ).to(device)
    
    # 추론 모드
    model.eval()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    templates = ["a photo of a {}.", "a blurry photo of a {}.", "a close-up of a {}."]

    # 텍스트 임베딩(템플릿 평균)
    prompts = [t.format(c) for c in classes for t in templates]
    with torch.no_grad():
        text_tokens = model.text.tokenize(prompts).to(device)
        text_emb = model.encode_text(text_tokens)                 # (#prompts, D)
        text_emb = text_emb.view(len(classes), -1, text_emb.size(-1)).mean(1)  # (C, D)

    # 이미지 임베딩
    transform = build_image_transform(args.image_size)
    ds = ImageFolderDataset(args.image_dir, transform)
    paths = [str(p) for p in ds.files]
    preds = []
    
    for i in range(0, len(ds), args.batch_size):
        batch = torch.stack([ds[j]["image"] for j in range(i, min(i+args.batch_size, len(ds)))])
        with torch.no_grad():
            img_emb = model.encode_image(batch.to(device))
            logits  = model.temperature * img_emb @ text_emb.t()
            y = logits.softmax(-1).argmax(-1).tolist()
        preds += y

    for p, yi in zip(paths, preds):
        print(p, "=>", classes[yi])

if __name__ == "__main__":
    main()
