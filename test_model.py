#!/usr/bin/env python3
# test_model.py - 자체 구현 CLIP 모델 테스트

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import torch
from src.model import CLIPModel
from src.text_encoder import TextTransformer

def test_clip_model():
    print("=== 자체 구현 CLIP 모델 테스트 ===")
    
    try:
        # 1. 모델 생성
        print("1. 모델 생성 중...")
        model = CLIPModel()
        print(f"   ✅ 모델 생성 완료: {type(model).__name__}")
        
        # 2. 텍스트 토크나이저 테스트
        print("2. 텍스트 토크나이저 테스트...")
        tokenizer = TextTransformer()
        texts = ['a photo of a cat', 'a photo of a dog']
        tokens = tokenizer.tokenize(texts)
        print(f"   ✅ 토큰 크기: {tokens.shape}")
        print(f"   ✅ 토큰 예시: {tokens[0][:10].tolist()}")
        
        # 3. 텍스트 임베딩 테스트
        print("3. 텍스트 임베딩 테스트...")
        with torch.no_grad():
            text_embeds = model.encode_text(tokens)
            print(f"   ✅ 텍스트 임베딩 크기: {text_embeds.shape}")
            print(f"   ✅ 임베딩 norm: {text_embeds.norm(dim=-1)}")
        
        # 4. 이미지 임베딩 테스트 (더미 데이터)
        print("4. 이미지 임베딩 테스트...")
        dummy_images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            image_embeds = model.encode_image(dummy_images)
            print(f"   ✅ 이미지 임베딩 크기: {image_embeds.shape}")
            print(f"   ✅ 임베딩 norm: {image_embeds.norm(dim=-1)}")
        
        # 5. 유사도 계산
        print("5. 유사도 계산...")
        similarity = (image_embeds @ text_embeds.t())
        print(f"   ✅ 유사도 매트릭스 크기: {similarity.shape}")
        print(f"   ✅ 유사도 값: {similarity}")
        print(f"   ✅ Temperature: {model.temperature.item():.4f}")
        
        # 6. 전체 forward 테스트
        print("6. 전체 forward 테스트...")
        with torch.no_grad():
            output = model(dummy_images, tokens)
            print(f"   ✅ Forward output keys: {list(output.keys())}")
            print(f"   ✅ logits_per_image 크기: {output['logits_per_image'].shape}")
            
        print("\n🎉 모든 테스트 통과! 자체 구현 CLIP 모델이 정상 작동합니다.")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clip_model()
    sys.exit(0 if success else 1) 