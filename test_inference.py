#!/usr/bin/env python3
# test_inference.py - 자체 구현 CLIP 모델 추론 테스트

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import torch
from src.model import CLIPModel
from src.dataset import build_image_transform
from PIL import Image

def test_inference():
    print("=== 자체 구현 CLIP 추론 테스트 ===")
    
    try:
        # 1. 모델 생성
        print("1. 모델 생성 중...")
        model = CLIPModel().eval()
        print("   ✅ 모델 생성 완료")
        
        # 2. 클래스 및 템플릿 설정
        classes = ["cat", "dog", "sample"]
        templates = ["a photo of a {}.", "a blurry photo of a {}.", "a close-up of a {}."]
        
        # 3. 텍스트 임베딩 생성 (템플릿 평균)
        print("2. 텍스트 임베딩 생성 중...")
        prompts = [t.format(c) for c in classes for t in templates]
        print(f"   생성된 프롬프트: {prompts}")
        
        with torch.no_grad():
            text_tokens = model.text.tokenize(prompts)
            text_emb = model.encode_text(text_tokens)  # (#prompts, D)
            text_emb = text_emb.view(len(classes), -1, text_emb.size(-1)).mean(1)  # (C, D)
            print(f"   ✅ 텍스트 임베딩 크기: {text_emb.shape}")
        
        # 4. 이미지 로드 및 분류
        print("3. 이미지 분류 중...")
        img_path = Path("data/images/image.png")
        
        if img_path.exists():
            transform = build_image_transform(224)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                img_emb = model.encode_image(img_tensor)
                logits = model.temperature * img_emb @ text_emb.t()
                probs = logits.softmax(-1)
                pred_idx = logits.argmax(-1).item()
                
            print(f"   ✅ 이미지: {img_path}")
            print(f"   ✅ 예측 클래스: {classes[pred_idx]}")
            print(f"   ✅ 확률 분포: {dict(zip(classes, probs[0].tolist()))}")
            
        else:
            print(f"   ⚠️ 이미지 파일이 없음: {img_path}")
            print("   더미 이미지로 테스트...")
            
            dummy_img = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                img_emb = model.encode_image(dummy_img)
                logits = model.temperature * img_emb @ text_emb.t()
                probs = logits.softmax(-1)
                pred_idx = logits.argmax(-1).item()
                
            print(f"   ✅ 더미 이미지 예측 클래스: {classes[pred_idx]}")
            print(f"   ✅ 확률 분포: {dict(zip(classes, probs[0].tolist()))}")
        
        print("\n🎉 추론 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 추론 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1) 