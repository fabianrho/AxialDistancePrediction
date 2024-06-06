

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import sys
sys.path.append("..")
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

# from models.swin_transformer import SwinTransformer
from models.swin_unet_v2 import SwinTransformerV2

from SSL.simmim_dataloader import build_loader_simmim


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    
    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)
    
    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)
    
    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5
    
    return targets_


# class SwinTransformerForSimMIM(SwinTransformer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         assert self.num_classes == 0

#         self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         trunc_normal_(self.mask_token, mean=0., std=.02)

#     def forward(self, x, mask):
#         x = self.patch_embed(x)

#         assert mask is not None
#         B, L, _ = x.shape

#         mask_tokens = self.mask_token.expand(B, L, -1)
#         w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
#         x = x * (1. - w) + mask_tokens * w

#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)

#         for layer in self.layers:
#             x = layer(x)
#         x = self.norm(x)

#         x = x.transpose(1, 2)
#         B, C, L = x.shape
#         H = W = int(L ** 0.5)
#         x = x.reshape(B, C, H, W)
#         return x

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return super().no_weight_decay() | {'mask_token'}


class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.load_imagenet_weights()

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        # self.config = config
        self.backbone = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, x, mask):
        z = self.backbone(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        
        # norm target as prompted
        # if self.config.NORM_TARGET.ENABLE:
        x = norm_targets(x, 47)
        
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.backbone, 'no_weight_decay'):
            return {'encoder.' + i for i in self.backbone.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.backbone, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.backbone.no_weight_decay_keywords()}
        return {}


def build_simmim():
    # model_type = config.MODEL.TYPE
    # if model_type == 'swin':
    #     # encoder = SwinTransformerForSimMIM(
    #     #     img_size=config.DATA.IMG_SIZE,
    #     #     patch_size=config.MODEL.SWIN.PATCH_SIZE,
    #     #     in_chans=config.MODEL.SWIN.IN_CHANS,
    #     #     num_classes=0,
    #     #     embed_dim=config.MODEL.SWIN.EMBED_DIM,
    #     #     depths=config.MODEL.SWIN.DEPTHS,
    #     #     num_heads=config.MODEL.SWIN.NUM_HEADS,
    #     #     window_size=config.MODEL.SWIN.WINDOW_SIZE,
    #     #     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
    #     #     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
    #     #     qk_scale=config.MODEL.SWIN.QK_SCALE,
    #     #     drop_rate=config.MODEL.DROP_RATE,
    #     #     drop_path_rate=config.MODEL.DROP_PATH_RATE,
    #     #     ape=config.MODEL.SWIN.APE,
    #     #     patch_norm=config.MODEL.SWIN.PATCH_NORM,
    #     #     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    #     # encoder_stride = 32
    #     # in_chans = config.MODEL.SWIN.IN_CHANS
    #     # patch_size = config.MODEL.SWIN.PATCH_SIZE
    #     pass
    # elif model_type == 'swinv2':
    #     encoder = SwinTransformerV2ForSimMIM(
    #         img_size=config.DATA.IMG_SIZE,
    #         patch_size=config.MODEL.SWINV2.PATCH_SIZE,
    #         in_chans=config.MODEL.SWINV2.IN_CHANS,
    #         num_classes=0,
    #         embed_dim=config.MODEL.SWINV2.EMBED_DIM,
    #         depths=config.MODEL.SWINV2.DEPTHS,
    #         num_heads=config.MODEL.SWINV2.NUM_HEADS,
    #         window_size=config.MODEL.SWINV2.WINDOW_SIZE,
    #         mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
    #         qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
    #         drop_rate=config.MODEL.DROP_RATE,
    #         drop_path_rate=config.MODEL.DROP_PATH_RATE,
    #         ape=config.MODEL.SWINV2.APE,
    #         patch_norm=config.MODEL.SWINV2.PATCH_NORM,
    #         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    #     encoder_stride = 32
    #     in_chans = config.MODEL.SWINV2.IN_CHANS
    #     patch_size = config.MODEL.SWINV2.PATCH_SIZE
    # else:
    #     raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    encoder = SwinTransformerV2ForSimMIM(img_size = 256, num_classes=0, drop_rate=0.2, window_size=8, depths = [2,2,2,2])

    model = SimMIM(encoder=encoder, encoder_stride=32, in_chans=3, patch_size=4)

    return model


if __name__ == "__main__":
    model = build_simmim()
    dataloader = build_loader_simmim(path="data/Pelvis/axial/Preprocessed_data/train/Image", batch_size=4, num_workers=4)
    device = torch.device("cuda:1")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    model.train()

    epochs = 10

    for epoch in range(0,epochs):
        print(f"Epoch: {epoch}")
        for img, mask,_ in dataloader:
            optimizer.zero_grad()

            img = img.to(device)
            mask = mask.to(device)
            loss = model(img, mask)
            loss.backward()
            optimizer.step()

        print(loss)




        
        # img, mask = batch
        # print(img.shape, mask.shape)
    # test_image = torch.randn(1, 3, 256, 256)
    # print(model)






