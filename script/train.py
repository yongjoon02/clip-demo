import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.trainer import CLIPLightning, CLIPDataModule
from src.utils import set_seed

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="ViT-B/32")
    ap.add_argument("--train-csv", default="data/train.csv")
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--img-root",  default="data/images")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--image-encoder", default="vit", choices=["vit", "resnet"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--vision-width", type=int, default=768)
    ap.add_argument("--vision-layers", type=int, default=12)
    ap.add_argument("--vision-heads", type=int, default=12)
    
    ap.add_argument("--text-width", type=int, default=512)
    ap.add_argument("--text-layers", type=int, default=12)
    ap.add_argument("--text-heads", type=int, default=8)
    ap.add_argument("--context-length", type=int, default=77)
    ap.add_argument("--vocab-size", type=int, default=49408)
    
    ap.add_argument("--embed-dim", type=int, default=512)
    
    return ap.parse_args()

def main():
    args = parse()
    set_seed(args.seed)

    dm = CLIPDataModule(
        args.train_csv, args.img_root, args.val_csv,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, context_length=args.context_length
    )
    
    model = CLIPLightning(
        model_name=args.model_name,
        image_encoder_type=args.image_encoder,
        image_size=args.image_size, patch_size=args.patch_size,
        vision_width=args.vision_width, vision_layers=args.vision_layers, vision_heads=args.vision_heads,
        text_width=args.text_width, text_layers=args.text_layers, text_heads=args.text_heads,
        context_length=args.context_length, vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        lr=args.lr, weight_decay=args.wd, finetune=args.finetune
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="32",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()