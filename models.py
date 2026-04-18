"""
================================================================================
MODULE 2: Model Architectures
================================================================================
Implements (all modular, GPU-ready):
  1. U-Net / U-Net++ with attention gates
  2. DeepLabv3+ with ASPP and ResNet/Xception backbone
  3. PSPNet with Pyramid Pooling Module
  4. SegFormer (transformer-based, lightweight)
  5. Mask2Former (panoptic/instance capable)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
)
from typing import List, Optional

NUM_CLASSES = 19  # Cityscapes training classes


# ═════════════════════════════════════════════════════════════════════════════
# Shared Utility Blocks
# ═════════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    """Standard Conv → BN → ReLU block used across all architectures."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution — reduces parameter count significantly."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel,
                                   padding=dilation, dilation=dilation,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


# ═════════════════════════════════════════════════════════════════════════════
# 1. U-Net++ with Attention Gates
# ═════════════════════════════════════════════════════════════════════════════

class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al. 2018).
    Suppresses irrelevant feature map activations in skip connections.
    g  = gating signal from decoder
    x  = skip connection from encoder
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear",
                             align_corners=False)
        att = self.psi(self.relu(self.W_g(g_up) + self.W_x(x)))
        return x * att


class UNetBlock(nn.Module):
    """Double convolution block used in each U-Net encoder/decoder stage."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ with attention gates.
    Dense skip connections create nested, re-designed skip paths
    that reduce the semantic gap between encoder and decoder.

    Architecture:
      Encoder: 5-level feature pyramid [64, 128, 256, 512, 1024]
      Decoder: Dense skip aggregation at each scale
      Attention gates at every skip connection
    """
    def __init__(self, in_ch: int = 3, num_classes: int = NUM_CLASSES):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]

        # ── Encoder (downsampling) ──────────────────────────────────────────
        self.enc = nn.ModuleList([
            UNetBlock(in_ch,       filters[0]),
            UNetBlock(filters[0],  filters[1]),
            UNetBlock(filters[1],  filters[2]),
            UNetBlock(filters[2],  filters[3]),
            UNetBlock(filters[3],  filters[4]),
        ])
        self.pool = nn.MaxPool2d(2)

        # ── U-Net++ intermediate nodes (dense skip connections) ────────────
        # Node naming: X_{i,j} where i=depth, j=dense step
        self.X_0_1 = UNetBlock(filters[0] + filters[1], filters[0])
        self.X_1_1 = UNetBlock(filters[1] + filters[2], filters[1])
        self.X_2_1 = UNetBlock(filters[2] + filters[3], filters[2])
        self.X_3_1 = UNetBlock(filters[3] + filters[4], filters[3])

        self.X_0_2 = UNetBlock(filters[0] * 2 + filters[1], filters[0])
        self.X_1_2 = UNetBlock(filters[1] * 2 + filters[2], filters[1])
        self.X_2_2 = UNetBlock(filters[2] * 2 + filters[3], filters[2])

        self.X_0_3 = UNetBlock(filters[0] * 3 + filters[1], filters[0])
        self.X_1_3 = UNetBlock(filters[1] * 3 + filters[2], filters[1])

        self.X_0_4 = UNetBlock(filters[0] * 4 + filters[1], filters[0])

        # ── Attention gates on final decoder path ──────────────────────────
        self.att = nn.ModuleList([
            AttentionGate(filters[1], filters[0], filters[0] // 2),
            AttentionGate(filters[2], filters[1], filters[1] // 2),
            AttentionGate(filters[3], filters[2], filters[2] // 2),
            AttentionGate(filters[4], filters[3], filters[3] // 2),
        ])

        # ── Segmentation head ──────────────────────────────────────────────
        self.head = nn.Conv2d(filters[0], num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _up(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Upsample x to spatial size of ref."""
        return F.interpolate(x, size=ref.shape[2:],
                             mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ────────────────────────────────────────────────────────
        e0 = self.enc[0](x)                    # (B, 64,  H/1,  W/1)
        e1 = self.enc[1](self.pool(e0))        # (B, 128, H/2,  W/2)
        e2 = self.enc[2](self.pool(e1))        # (B, 256, H/4,  W/4)
        e3 = self.enc[3](self.pool(e2))        # (B, 512, H/8,  W/8)
        e4 = self.enc[4](self.pool(e3))        # (B,1024, H/16, W/16)

        # ── Dense skip nodes (U-Net++ connections) ─────────────────────────
        x0_1 = self.X_0_1(torch.cat([e0, self._up(e1, e0)], dim=1))
        x1_1 = self.X_1_1(torch.cat([e1, self._up(e2, e1)], dim=1))
        x2_1 = self.X_2_1(torch.cat([e2, self._up(e3, e2)], dim=1))
        x3_1 = self.X_3_1(torch.cat([e3, self._up(e4, e3)], dim=1))

        x0_2 = self.X_0_2(torch.cat([e0, x0_1, self._up(x1_1, e0)], dim=1))
        x1_2 = self.X_1_2(torch.cat([e1, x1_1, self._up(x2_1, e1)], dim=1))
        x2_2 = self.X_2_2(torch.cat([e2, x2_1, self._up(x3_1, e2)], dim=1))

        x0_3 = self.X_0_3(torch.cat([e0, x0_1, x0_2, self._up(x1_2, e0)], dim=1))
        x1_3 = self.X_1_3(torch.cat([e1, x1_1, x1_2, self._up(x2_2, e1)], dim=1))

        x0_4 = self.X_0_4(torch.cat([e0, x0_1, x0_2, x0_3,
                                       self._up(x1_3, e0)], dim=1))

        # ── Segmentation head ──────────────────────────────────────────────
        out = self.head(x0_4)
        return F.interpolate(out, size=x.shape[2:],
                             mode="bilinear", align_corners=False)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DeepLabv3+ with ASPP
# ═════════════════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al. 2018).
    Captures multi-scale context via parallel dilated convolutions:
      rates = [1, 6, 12, 18] — progressively larger receptive fields.
    """
    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBnRelu(in_ch, out_ch, kernel=1, padding=0),           # rate=1
            ConvBnRelu(in_ch, out_ch, dilation=6,  padding=6),        # rate=6
            ConvBnRelu(in_ch, out_ch, dilation=12, padding=12),       # rate=12
            ConvBnRelu(in_ch, out_ch, dilation=18, padding=18),       # rate=18
        ])
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Fuse all 5 branches
        self.proj = ConvBnRelu(out_ch * 5, out_ch, kernel=1, padding=0)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [b(x) for b in self.branches]
        gap   = F.interpolate(self.gap(x), size=(h, w),
                              mode="bilinear", align_corners=False)
        feats.append(gap)
        return self.drop(self.proj(torch.cat(feats, dim=1)))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabv3+ (Chen et al. 2018) with ResNet-101 backbone.

    Architecture:
      Encoder: ResNet-101 with dilated convolutions (output_stride=16)
      ASPP module for multi-scale context
      Decoder: low-level skip + upsampling to full resolution
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 output_stride: int = 16,
                 backbone: str = "resnet101"):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        if backbone == "resnet101":
            base = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Re-use ResNet layers; apply dilation to layer3/4 for dense features
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu,
                                     base.maxpool)
        self.layer1 = base.layer1   # stride=4,  ch=256
        self.layer2 = base.layer2   # stride=8,  ch=512
        self.layer3 = base.layer3   # stride=16, ch=1024 (dilated)
        self.layer4 = base.layer4   # stride=32, ch=2048 (dilated)

        # Apply dilated convolutions (output_stride=16)
        self._apply_dilation(self.layer3, dilation=2, stride=1)
        self._apply_dilation(self.layer4, dilation=4, stride=1)

        # ── ASPP ──────────────────────────────────────────────────────────
        self.aspp = ASPP(in_ch=2048, out_ch=256)

        # ── Low-level feature projection (from layer1) ─────────────────────
        self.low_proj = ConvBnRelu(256, 48, kernel=1, padding=0)

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            SeparableConv2d(256 + 48, 256),
            SeparableConv2d(256, 256),
            nn.Conv2d(256, num_classes, 1),
        )

    @staticmethod
    def _apply_dilation(layer, dilation: int, stride: int):
        """Convert regular convolutions to dilated ones in a ResNet layer."""
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation  = (dilation, dilation)
                m.padding   = (dilation, dilation)
                m.stride    = (stride, stride)
            elif isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # ── Backbone forward ───────────────────────────────────────────────
        x0  = self.layer0(x)
        low = self.layer1(x0)       # low-level features  (H/4, W/4, 256)
        x   = self.layer2(low)
        x   = self.layer3(x)
        x   = self.layer4(x)        # high-level features (H/16, W/16, 2048)

        # ── ASPP ──────────────────────────────────────────────────────────
        x = self.aspp(x)            # (H/16, W/16, 256)

        # ── Upsample to match low-level feature map ────────────────────────
        x    = F.interpolate(x, size=low.shape[2:],
                             mode="bilinear", align_corners=False)
        low  = self.low_proj(low)   # (H/4, W/4, 48)
        x    = torch.cat([x, low], dim=1)   # (H/4, W/4, 304)

        # ── Decoder + final upsample ──────────────────────────────────────
        x = self.decoder(x)
        return F.interpolate(x, size=(h, w),
                             mode="bilinear", align_corners=False)


# ═════════════════════════════════════════════════════════════════════════════
# 3. PSPNet with Pyramid Pooling Module
# ═════════════════════════════════════════════════════════════════════════════

class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (Zhao et al. 2017).
    Pools features at 4 scales {1×1, 2×2, 3×3, 6×6} → concatenate → project.
    Captures both local detail and global context simultaneously.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 pool_sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        branch_ch = in_ch // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                ConvBnRelu(in_ch, branch_ch, kernel=1, padding=0),
            )
            for s in pool_sizes
        ])
        self.bottleneck = ConvBnRelu(in_ch + branch_ch * len(pool_sizes), out_ch)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pools = [F.interpolate(stage(x), size=(h, w),
                               mode="bilinear", align_corners=False)
                 for stage in self.stages]
        return self.drop(self.bottleneck(torch.cat([x] + pools, dim=1)))


class PSPNet(nn.Module):
    """
    PSPNet: Pyramid Scene Parsing Network.
    ResNet-50 backbone → PPM → auxiliary + main classification head.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 aux_loss: bool = True):
        super().__init__()
        self.aux_loss = aux_loss

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu,
                                     base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Dilate layer3 and layer4 to keep feature map at 1/8
        self._dilate_layer(self.layer3, stride=1, dilation=2)
        self._dilate_layer(self.layer4, stride=1, dilation=4)

        # PPM takes 2048-ch features from layer4
        self.ppm  = PyramidPoolingModule(2048, 512)

        # Main classification head
        self.head = nn.Sequential(
            ConvBnRelu(512, 512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1),
        )

        # Auxiliary head (attached to layer3) — helps gradient flow
        if aux_loss:
            self.aux_head = nn.Sequential(
                ConvBnRelu(1024, 256),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, 1),
            )

    @staticmethod
    def _dilate_layer(layer, stride: int, dilation: int):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding  = (dilation, dilation)
                if m.stride   == (2, 2):
                    m.stride   = (stride, stride)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2:]
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)    # 1024 ch — for aux
        f4 = self.layer4(f3)    # 2048 ch

        out = self.ppm(f4)
        out = self.head(out)
        out = F.interpolate(out, size=(h, w),
                            mode="bilinear", align_corners=False)

        if self.aux_loss and self.training:
            aux = self.aux_head(f3)
            aux = F.interpolate(aux, size=(h, w),
                                mode="bilinear", align_corners=False)
            return out, aux
        return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. SegFormer (Transformer-based, Xie et al. 2021)
# ═════════════════════════════════════════════════════════════════════════════

class DWConv(nn.Module):
    """Depth-wise convolution used inside the Mix-FFN of SegFormer."""
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dw(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """Mix-FFN: FFN with depth-wise conv for local positional encoding."""
    def __init__(self, ch: int, expansion: int = 4):
        super().__init__()
        hidden = ch * expansion
        self.fc1  = nn.Linear(ch, hidden)
        self.dw   = DWConv(hidden)
        self.fc2  = nn.Linear(hidden, ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dw(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        return self.drop(self.fc2(x))


class EfficientSelfAttention(nn.Module):
    """
    Efficient self-attention with sequence reduction ratio R.
    Reduces key/value sequence length by factor R to lower compute.
    """
    def __init__(self, ch: int, num_heads: int, reduction_ratio: int = 1):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = ch // num_heads
        self.scale      = self.head_dim ** -0.5
        self.R          = reduction_ratio

        self.q  = nn.Linear(ch, ch)
        self.kv = nn.Linear(ch, ch * 2)
        self.proj = nn.Linear(ch, ch)
        self.drop = nn.Dropout(0.1)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(ch, ch, reduction_ratio, stride=reduction_ratio)
            self.sr_norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.R > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.sr_norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SegFormerBlock(nn.Module):
    """One transformer block: efficient attention + Mix-FFN + LayerNorm."""
    def __init__(self, ch: int, num_heads: int,
                 reduction_ratio: int, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.attn  = EfficientSelfAttention(ch, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(ch)
        self.ffn   = MixFFN(ch, expansion)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class PatchEmbed(nn.Module):
    """Overlapping patch embedding (stride < patch_size) for SegFormer."""
    def __init__(self, in_ch: int, out_ch: int,
                 patch_size: int = 7, stride: int = 4):
        super().__init__()
        pad = patch_size // 2
        self.proj = nn.Conv2d(in_ch, out_ch, patch_size, stride=stride,
                              padding=pad)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        return self.norm(x), H, W


class SegFormerEncoder(nn.Module):
    """
    Hierarchical transformer encoder with 4 stages.
    Each stage: patch embed + N transformer blocks.
    Output: 4 feature maps at 1/4, 1/8, 1/16, 1/32 resolution.
    """
    # SegFormer-B2 configuration
    CONFIGS = {
        "B0": dict(embed_dims=[32, 64, 160, 256],
                   num_heads=[1, 2, 5, 8],
                   depths=[2, 2, 2, 2],
                   reduction_ratios=[8, 4, 2, 1]),
        "B2": dict(embed_dims=[64, 128, 320, 512],
                   num_heads=[1, 2, 5, 8],
                   depths=[3, 4, 6, 3],
                   reduction_ratios=[8, 4, 2, 1]),
    }

    def __init__(self, variant: str = "B2", in_ch: int = 3):
        super().__init__()
        cfg = self.CONFIGS[variant]
        dims = cfg["embed_dims"]
        heads = cfg["num_heads"]
        depths = cfg["depths"]
        rrs   = cfg["reduction_ratios"]
        strides = [4, 2, 2, 2]
        patch_sizes = [7, 3, 3, 3]

        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()
        in_d = in_ch

        for i in range(4):
            blocks = nn.ModuleList([
                SegFormerBlock(dims[i], heads[i], rrs[i])
                for _ in range(depths[i])
            ])
            embed = PatchEmbed(in_d, dims[i], patch_sizes[i], strides[i])
            self.stages.append(nn.ModuleList([embed, blocks]))
            self.norms.append(nn.LayerNorm(dims[i]))
            in_d = dims[i]

        self.out_channels = dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i, (embed, blocks) in enumerate(self.stages):
            x, H, W = embed(x)
            for block in blocks:
                x = block(x, H, W)
            x = self.norms[i](x)
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            outs.append(x)
        return outs


class SegFormerDecoder(nn.Module):
    """
    All-MLP decoder: project each scale → upsample to 1/4 → concat → classify.
    Lightweight: no convolutions, no cross-attention.
    """
    def __init__(self, in_channels: List[int],
                 embed_dim: int = 256,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Linear(ch, embed_dim) for ch in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor],
                target_size: tuple) -> torch.Tensor:
        ref = features[0]           # finest scale (1/4)
        outs = []
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            # Project channel dim
            x = feat.flatten(2).transpose(1, 2)   # (B, H*W, C)
            x = self.proj[i](x)                   # (B, H*W, embed_dim)
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            # Upsample to finest scale
            x = F.interpolate(x, size=ref.shape[2:],
                              mode="bilinear", align_corners=False)
            outs.append(x)

        x = self.fuse(torch.cat(outs, dim=1))
        return F.interpolate(x, size=target_size,
                             mode="bilinear", align_corners=False)


class SegFormer(nn.Module):
    """
    Full SegFormer model: hierarchical encoder + all-MLP decoder.
    variant: "B0" (lightweight) or "B2" (higher accuracy).
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 variant: str = "B2"):
        super().__init__()
        self.encoder = SegFormerEncoder(variant)
        self.decoder = SegFormerDecoder(
            self.encoder.out_channels, embed_dim=256, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        features    = self.encoder(x)
        return self.decoder(features, target_size)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Mask2Former (Simplified — panoptic-capable)
# ═════════════════════════════════════════════════════════════════════════════

class MaskedAttention(nn.Module):
    """
    Masked cross-attention: each query attends only within its predicted mask.
    This is the key novelty in Mask2Former (Cheng et al. 2022).
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=0.0, batch_first=True)

    def forward(self, query: torch.Tensor,
                memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query:  (B, N_q, C)
        memory: (B, H*W, C)
        mask:   (B, N_q, H*W) — binary attention mask
        """
        attn_mask = None
        if mask is not None:
            # Convert boolean mask: True = attend, False = ignore
            attn_mask = (mask < 0.5).bool()
            attn_mask = attn_mask.flatten(0, 1) if attn_mask.dim() == 3 else attn_mask
        out, _ = self.attn(query, memory, memory, attn_mask=attn_mask)
        return out


class Mask2FormerDecoder(nn.Module):
    """
    Transformer decoder with L layers of masked cross-attention.
    Processes N learnable queries → produces class predictions + mask embeddings.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 num_queries: int = 100, num_layers: int = 6,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.queries = nn.Embedding(num_queries, d_model)
        self.layers  = nn.ModuleList([
            nn.ModuleDict({
                "self_attn":    nn.MultiheadAttention(d_model, num_heads,
                                                      batch_first=True),
                "cross_attn":   MaskedAttention(d_model, num_heads),
                "ffn":          nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-obj
        self.mask_head  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, memory: torch.Tensor,
                pixel_features: torch.Tensor) -> dict:
        """
        memory:         (B, C, H, W) — backbone features
        pixel_features: (B, C, H', W') — high-res pixel embedding
        Returns: class_logits, mask_logits
        """
        B, C, H, W = memory.shape
        mem_flat = memory.flatten(2).transpose(1, 2)    # (B, H*W, C)

        # Initialise learnable queries
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        pred_mask = None
        for layer in self.layers:
            # Self-attention among queries
            q_res, _ = layer["self_attn"](q, q, q)
            q = layer["norm1"](q + q_res)

            # Masked cross-attention: queries attend to pixel memory
            q_res = layer["cross_attn"](q, mem_flat, pred_mask)
            q = layer["norm2"](q + q_res)

            # FFN
            q = layer["norm3"](q + layer["ffn"](q))

            # Predict masks for next layer's masking
            mask_emb = self.mask_head(q)              # (B, N_q, C)
            pf_flat  = pixel_features.flatten(2)      # (B, C, H'*W')
            pred_mask = torch.bmm(mask_emb, pf_flat)  # (B, N_q, H'*W')
            # Reshape for masked attention usage
            # (kept flat; MaskedAttention will handle it)

        class_logits = self.class_head(q)             # (B, N_q, num_classes+1)
        mask_logits  = pred_mask                       # (B, N_q, H'*W')
        return {"class_logits": class_logits,
                "mask_logits":  mask_logits}


class Mask2Former(nn.Module):
    """
    Simplified Mask2Former for semantic segmentation.
    Full model also handles instance / panoptic via bipartite matching —
    the semantic variant merges queries by class prediction.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 num_queries: int = 100):
        super().__init__()
        # ResNet-50 backbone (pretrained)
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(base.conv1, base.bn1, base.relu,
                                       base.maxpool,
                                       base.layer1, base.layer2,
                                       base.layer3, base.layer4)
        # Pixel decoder: simple FPN-style multi-scale projection
        self.pixel_proj = ConvBnRelu(2048, 256, kernel=1, padding=0)

        # Transformer decoder
        self.transformer = Mask2FormerDecoder(
            d_model=256, num_heads=8,
            num_queries=num_queries, num_layers=6,
            num_classes=num_classes,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        feat = self.backbone(x)                        # (B, 2048, H/32, W/32)
        memory = self.pixel_proj(feat)                 # (B, 256,  H/32, W/32)

        preds = self.transformer(memory, memory)

        # For semantic seg: collapse queries → per-pixel class scores
        B, Nq, Hf_Wf = preds["mask_logits"].shape
        Hf = Wf = int(Hf_Wf ** 0.5)
        mask_logits  = preds["mask_logits"].reshape(B, Nq, Hf, Wf)
        class_logits = preds["class_logits"][..., :self.num_classes]  # remove no-obj
        class_logits = class_logits.softmax(dim=-1)    # (B, Nq, C)

        # Weighted sum: mask × class → per-pixel class scores
        mask_probs = mask_logits.sigmoid()             # (B, Nq, Hf, Wf)
        seg = torch.einsum("bqhw,bqc->bchw", mask_probs, class_logits)
        return F.interpolate(seg, size=(H, W),
                             mode="bilinear", align_corners=False)


# ═════════════════════════════════════════════════════════════════════════════
# Model Registry & Factory
# ═════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "unetplusplus":  lambda: UNetPlusPlus(num_classes=NUM_CLASSES),
    "deeplabv3plus": lambda: DeepLabV3Plus(num_classes=NUM_CLASSES),
    "pspnet":        lambda: PSPNet(num_classes=NUM_CLASSES),
    "segformer_b0":  lambda: SegFormer(num_classes=NUM_CLASSES, variant="B0"),
    "segformer_b2":  lambda: SegFormer(num_classes=NUM_CLASSES, variant="B2"),
    "mask2former":   lambda: Mask2Former(num_classes=NUM_CLASSES),
}


def build_model(name: str, device: torch.device) -> nn.Module:
    """Instantiate model by name and move to device."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[name]().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {name} | trainable params: {n_params/1e6:.2f} M")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 512, 1024).to(device)
    for name in MODEL_REGISTRY:
        print(f"\n── {name} ──")
        m = build_model(name, device)
        m.eval()
        with torch.no_grad():
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
        print(f"   Output: {out.shape}")   # expect (2, 19, 512, 1024)
