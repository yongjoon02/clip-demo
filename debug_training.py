import warnings
warnings.filterwarnings("ignore")

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.model import CLIPBackbone
from src.dataset import ImageTextCsv, build_collate_clip
from src.loss import clip_contrastive_loss
from torch.utils.data import DataLoader

def debug_training():
    # 모델 생성
    model = CLIPBackbone("ViT-B/32", finetune=True)
    model = model.cuda()
    
    # 데이터 로드
    ds = ImageTextCsv("data/train.csv", "data/images", model.preprocess)
    collate_fn = build_collate_clip(model.tokenize, model.context_length)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 배치 하나 가져오기
    batch = next(iter(loader))
    images = batch["images"].cuda()
    text_tokens = batch["text_tokens"].cuda()
    
    print(f"Images shape: {images.shape}")
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Text tokens: {text_tokens}")
    
    # Forward pass
    with torch.no_grad():
        out = model(images, text_tokens)
        print(f"Image embeds shape: {out['image_embeds'].shape}")
        print(f"Text embeds shape: {out['text_embeds'].shape}")
        print(f"Logits per image shape: {out['logits_per_image'].shape}")
        print(f"Temperature: {model.temperature}")
        
        # Loss 계산
        loss = clip_contrastive_loss(out["logits_per_image"], out["logits_per_text"])
        print(f"Loss: {loss.item()}")
        
        # 로짓 값 확인
        print(f"Logits per image:\n{out['logits_per_image']}")
        print(f"Max logit: {out['logits_per_image'].max().item()}")
        print(f"Min logit: {out['logits_per_image'].min().item()}")

if __name__ == "__main__":
    debug_training() 