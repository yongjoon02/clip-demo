# src/trainer.py
import math, torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
from .model import CLIPModel
from .dataset import ImageTextCsv, build_image_transform, build_collate_clip
import torch.nn.functional as F

def clip_contrastive_loss(logits_per_image, logits_per_text):
    b = logits_per_image.size(0)
    tgt = torch.arange(b, device=logits_per_image.device)
    return 0.5 * (F.cross_entropy(logits_per_image, tgt) + F.cross_entropy(logits_per_text, tgt))

class CLIPLightning(pl.LightningModule):
    def __init__(self,
                 model_name="ViT-B/32",   # 표기용
                 # 이미지 인코더 설정
                 image_encoder_type="vit",  # "vit" 또는 "resnet"
                 image_size=224, patch_size=32,
                 vision_width=768, vision_layers=12, vision_heads=12,
                 # 텍스트 인코더 설정
                 text_width=512, text_layers=12, text_heads=8,
                 context_length=77, vocab_size=49408,
                 # 공통 설정
                 embed_dim=512,
                 lr=1e-4, weight_decay=0.01, finetune=False):
        super().__init__()
        self.save_hyperparameters()
        
        # 다중 인코더 지원 CLIP 모델 생성
        self.model = CLIPModel(
            image_encoder_type=image_encoder_type,
            image_size=image_size, patch_size=patch_size,
            vision_width=vision_width, vision_layers=vision_layers, vision_heads=vision_heads,
            text_width=text_width, text_layers=text_layers, text_heads=text_heads, 
            context_length=context_length, vocab_size=vocab_size, embed_dim=embed_dim
        )

        # finetune=False면 보통 logit_scale만 학습하거나, 전체 학습 (여기선 전체 학습 허용)
        if not finetune:
            for n,p in self.model.named_parameters():
                # logit_scale만 학습하려면 아래 주석해제
                # p.requires_grad = (n == "logit_scale")
                p.requires_grad = True

    @property
    def temperature(self): return self.model.temperature

    def training_step(self, batch, _):
        out = self.model(batch["images"], batch["text_tokens"].to(self.device))
        loss = clip_contrastive_loss(out["logits_per_image"], out["logits_per_text"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/logit_scale", self.temperature, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        out = self.model(batch["images"], batch["text_tokens"].to(self.device))
        loss = clip_contrastive_loss(out["logits_per_image"], out["logits_per_text"])
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            self.model.logit_scale.clamp_(max=math.log(100.0))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9,0.98))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_steps or 1000)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, img_root, val_csv=None, batch_size=128, num_workers=4,
                 image_size=224, context_length=77):
        super().__init__()
        self.train_csv, self.val_csv = train_csv, val_csv
        self.img_root = img_root
        self.batch_size, self.num_workers = batch_size, num_workers
        self.image_size, self.context_length = image_size, context_length
        
        # BPE 토크나이저와 이미지 변환 준비
        from .text_encoder import TextTransformer
        self.tokenizer = TextTransformer(context_length=context_length)
        self.transform = build_image_transform(image_size)

    def setup(self, stage=None):
        self.ds_train = ImageTextCsv(self.train_csv, self.img_root, self.transform, self.tokenizer, self.context_length)
        self.ds_val   = ImageTextCsv(self.val_csv,   self.img_root, self.transform, self.tokenizer, self.context_length) if self.val_csv else None

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers>0,
                          prefetch_factor=2 if self.num_workers>0 else None,
                          drop_last=shuffle,
                          collate_fn=build_collate_clip(self.tokenizer, self.context_length))

    def train_dataloader(self): return self._dl(self.ds_train, True)
    def val_dataloader(self):   return None if self.ds_val is None else self._dl(self.ds_val, False)
