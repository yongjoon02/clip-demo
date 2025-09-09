import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pathlib import Path
import random

def create_synthetic_images_and_captions(num_samples=100, output_dir="data/synthetic"):
    """합성 이미지-텍스트 쌍 생성"""
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    # 색상과 도형 정의
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255), 
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0)
    }
    
    shapes = ["circle", "rectangle", "triangle"]
    
    data = []
    
    for i in range(num_samples):
        # 이미지 생성
        img = Image.new('RGB', (224, 224), (255, 255, 255))  # 흰 배경
        draw = ImageDraw.Draw(img)
        
        # 랜덤 색상과 도형 선택
        color_name = random.choice(list(colors.keys()))
        color_rgb = colors[color_name]
        shape = random.choice(shapes)
        
        # 도형 그리기
        if shape == "circle":
            x, y = random.randint(50, 174), random.randint(50, 174)
            radius = random.randint(20, 40)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color_rgb)
            
        elif shape == "rectangle":
            x1, y1 = random.randint(20, 100), random.randint(20, 100)
            x2, y2 = x1 + random.randint(40, 80), y1 + random.randint(40, 80)
            draw.rectangle([x1, y1, x2, y2], fill=color_rgb)
            
        elif shape == "triangle":
            x, y = random.randint(50, 174), random.randint(50, 174)
            size = random.randint(30, 50)
            points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
            draw.polygon(points, fill=color_rgb)
        
        # 이미지 저장
        img_filename = f"synthetic_{i:04d}.png"
        img_path = output_dir / "images" / img_filename
        img.save(img_path)
        
        # 캡션 생성 (여러 변형)
        captions = [
            f"a {color_name} {shape}",
            f"an image of a {color_name} {shape}",
            f"a {color_name} colored {shape}",
            f"a picture showing a {color_name} {shape}",
            f"{color_name} {shape} on white background"
        ]
        
        # 랜덤하게 1-3개 캡션 선택
        num_captions = random.randint(1, 3)
        selected_captions = random.sample(captions, num_captions)
        
        for caption in selected_captions:
            data.append({
                "image_path": f"images/{img_filename}",
                "caption": caption
            })
    
    # CSV 파일 저장
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "train.csv", index=False)
    
    print(f"Generated {num_samples} synthetic images")
    print(f"Created {len(data)} image-text pairs")
    print(f"Saved to: {output_dir}")
    
    return df

if __name__ == "__main__":
    df = create_synthetic_images_and_captions(num_samples=200)
    print("\nFirst few samples:")
    print(df.head(10)) 