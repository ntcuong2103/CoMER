
import torch
import torch.nn as nn
import timm
from einops import rearrange
import pytorch_lightning as pl
from comer.model.pos_enc import ImgPosEnc


class SwinEncoder(pl.LightningModule):
    def __init__(self, d_model=512, swin_name="swinv2_base_window8_256"):
        super().__init__()
        self.input_size = (256, 256)
        self.backbone = timm.create_model(swin_name, pretrained=True, features_only=True)
        self.out_channels = self.backbone.feature_info[-1]["num_chs"]
        self.proj = nn.Conv2d(self.out_channels, d_model, kernel_size=1)
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: torch.FloatTensor, img_mask: torch.LongTensor):
        if img.shape[1] == 1:
            img = torch.cat([img] * 3, dim=1)
        img = torch.nn.functional.interpolate(img, size=self.input_size, mode="bilinear", align_corners=False)

        feats = self.backbone(img)[-1]  # (B, H, W, C)
        feats = rearrange(feats, "b h w c -> b c h w")        
        feats = self.proj(feats)        # (B, d_model, H, W)

        B, C, H, W = feats.shape
        feats = rearrange(feats, "b c h w -> b h w c")
        mask = torch.nn.functional.interpolate(img_mask.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).bool()
        feats = self.pos_enc_2d(feats, mask)
        feats = self.norm(feats)
        return feats, mask
