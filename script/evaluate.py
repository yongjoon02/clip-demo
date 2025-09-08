# script/evaluate.py
import argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.model import CLIPModel
from src.dataset import build_image_transform

def recall_at_k(sim: torch.Tensor, k=1) -> float:
    topk = sim.topk(k, dim=1).indices
    gt = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    return (topk == gt).any(dim=1).float().mean().item()

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="ViT-B/32", help="모델 크기 (표기용)")
    ap.add_argument("--csv", required=True)          # image_path, caption
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--limit", type=int, default=512)
    # 모델 아키텍처 설정
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=512)
    return ap.parse_args()

def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 자체 구현 CLIP 모델 생성
    model = CLIPModel(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim
    ).to(dev)
    
    # 추론 모드
    model.eval()

    df = pd.read_csv(args.csv).head(args.limit)
    
    # 텍스트 임베딩
    with torch.no_grad():
        tok = model.text.tokenize(df["caption"].astype(str).tolist()).to(dev)
        txt = model.encode_text(tok)
        
    # 이미지 임베딩
    transform = build_image_transform(args.image_size)
    ims = [transform(Image.open(Path(args.img_root) / p).convert("RGB")) for p in df["image_path"]]
    ims = torch.stack(ims).to(dev)
    with torch.no_grad():
        im = model.encode_image(ims)

    # 유사도 계산 및 평가
    sim = model.temperature * im @ txt.t()
    forward = (recall_at_k(sim,1), recall_at_k(sim,5), recall_at_k(sim,10))
    backward= (recall_at_k(sim.t(),1), recall_at_k(sim.t(),5), recall_at_k(sim.t(),10))
    print(f"Image→Text R@1/5/10: {forward[0]:.3f}/{forward[1]:.3f}/{forward[2]:.3f}")
    print(f"Text→Image R@1/5/10: {backward[0]:.3f}/{backward[1]:.3f}/{backward[2]:.3f}")

if __name__ == "__main__":
    main()
