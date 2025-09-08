# 🎨 CLIP-Demo: 자체 구현 CLIP 모델

PyTorch Lightning을 사용한 완전한 **자체 구현 CLIP** (Contrastive Language-Image Pre-training) 모델입니다.

## ✨ 주요 특징

- 🔧 **완전 자체 구현**: OpenAI CLIP을 사용하지 않고 처음부터 구현
- 🏗️ **모듈화된 아키텍처**: Vision Transformer + Text Transformer
- ⚡ **PyTorch Lightning**: 깔끔한 학습 파이프라인
- 🎯 **Zero-Shot 분류**: 학습 시 보지 못한 클래스도 분류 가능
- 📊 **완전한 평가**: Recall@K 메트릭 지원

## 🏛️ 모델 아키텍처

```
┌─────────────────┐    ┌──────────────────┐
│   이미지 입력    │    │    텍스트 입력    │
│   (224×224)     │    │   ("a cat")     │
└─────────┬───────┘    └─────────┬────────┘
          │                      │
    ┌─────▼──────┐        ┌──────▼───────┐
    │ViT Encoder │        │Text Encoder  │
    │  (12 layers)│        │ (12 layers)  │
    └─────┬──────┘        └──────┬───────┘
          │                      │
    ┌─────▼──────┐        ┌──────▼───────┐
    │Image Embed │        │ Text Embed   │
    │   (512D)   │        │   (512D)     │
    └─────┬──────┘        └──────┬───────┘
          │                      │
          └──────┬─────────┬─────┘
                 │         │
            ┌────▼─────────▼────┐
            │  Cosine Similarity │
            │  × Temperature     │
            └────────────────────┘
```

## 📁 프로젝트 구조

```
clip-demo/
├── src/                    # 핵심 모델 코드
│   ├── model.py           # CLIP 메인 모델
│   ├── image_encoder.py   # Vision Transformer
│   ├── text_encoder.py    # Text Transformer + 토크나이저
│   ├── clip_layers.py     # Transformer 블록들
│   ├── trainer.py         # PyTorch Lightning 모듈
│   ├── dataset.py         # 데이터 로더
│   ├── loss.py           # Contrastive Loss
│   └── utils.py          # 유틸리티 함수
├── script/                # 실행 스크립트
│   ├── train.py          # 학습 스크립트
│   ├── inference.py      # 추론 스크립트
│   └── evaluate.py       # 평가 스크립트
├── data/                  # 샘플 데이터
│   ├── train.csv         # 학습 데이터 목록
│   └── images/           # 이미지 파일들
└── test_model.py         # 모델 테스트 스크립트
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 리포지토리 클론
git clone https://github.com/yongjoon02/clip-demo.git
cd clip-demo

# 의존성 설치
pip install torch torchvision pytorch-lightning pandas pillow transformers
```

### 2. 모델 테스트

```bash
# 기본 모델 동작 확인
python test_model.py
```

### 3. 학습 실행

```bash
# 기본 설정으로 학습
python script/train.py

# 커스텀 설정
python script/train.py --epochs 10 --batch-size 64 --lr 1e-4
```

### 4. 추론 실행

```bash
# Zero-shot 이미지 분류
python script/inference.py --image-dir data/images --classes "cat,dog,car"
```

### 5. 모델 평가

```bash
# Recall@K 평가
python script/evaluate.py --csv data/train.csv --img-root data/images
```

## 🔧 모델 구성 요소

### Vision Transformer (ViT)
- **패치 크기**: 32×32 (224×224 → 7×7 패치)
- **임베딩 차원**: 768
- **레이어 수**: 12
- **어텐션 헤드**: 12

### Text Transformer
- **컨텍스트 길이**: 77 토큰
- **임베딩 차원**: 512  
- **레이어 수**: 12
- **어텐션 헤드**: 8
- **토크나이저**: 자체 구현 (간단한 문자 기반)

### 공통 임베딩 공간
- **차원**: 512
- **정규화**: L2 정규화
- **온도 스케일링**: 학습 가능한 파라미터

## 📊 학습 과정

### Contrastive Learning
```python
# 배치 내에서 이미지-텍스트 쌍 매칭
loss = 0.5 * (
    CrossEntropy(image→text_logits, diagonal_targets) +
    CrossEntropy(text→image_logits, diagonal_targets)
)
```

### 학습 설정
- **옵티마이저**: AdamW (β₁=0.9, β₂=0.98)
- **학습률 스케줄러**: CosineAnnealingLR
- **그래디언트 클리핑**: 1.0
- **정밀도**: FP32

## 🎯 Zero-Shot 분류

### 다중 템플릿 사용
```python
templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up of a {}."
]
```

### 분류 과정
1. 각 클래스별로 다중 템플릿 적용
2. 텍스트 임베딩 생성 후 평균화
3. 이미지와 텍스트 간 코사인 유사도 계산
4. 가장 높은 유사도의 클래스 선택

## 📈 성능 평가

### Recall@K 메트릭
- **Image→Text**: 이미지로 텍스트 검색
- **Text→Image**: 텍스트로 이미지 검색
- **K=1,5,10**: 상위 K개 결과 중 정답 포함률

## 🔍 예제 결과

```bash
=== 자체 구현 CLIP 모델 테스트 ===
✅ 모델 생성 완료: CLIPModel
✅ 텍스트 임베딩 크기: torch.Size([2, 512])
✅ 이미지 임베딩 크기: torch.Size([2, 512])
✅ Temperature: 14.2857
🎉 모든 테스트 통과!
```

## 🛠️ 커스터마이징

### 모델 아키텍처 변경
```python
# script/train.py에서 하이퍼파라미터 조정
--vision-width 768        # ViT 임베딩 차원
--vision-layers 12        # ViT 레이어 수
--text-width 512          # 텍스트 임베딩 차원
--embed-dim 512           # 공통 임베딩 차원
```

### 데이터셋 형식
```csv
image_path,caption
image1.jpg,"a photo of a cat"
image2.jpg,"a dog playing in the park"
```

## 🤝 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 참고 문헌

- [Learning Transferable Visual Representations with Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

---

**Made with ❤️ by [yongjoon02](https://github.com/yongjoon02)**
