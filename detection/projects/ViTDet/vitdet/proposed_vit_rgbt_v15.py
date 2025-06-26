import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from collections import OrderedDict
from functools import partial

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.registry import MODELS
from typing import Optional

# 기존 유틸리티 함수들 유지
def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode='bicubic',
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False, 
                 rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_cfg=dict(type='GELU'), bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0,
                 init_values=None, norm_cfg=dict(type='LN', eps=1e-6), 
                 act_cfg=dict(type='GELU'), use_rel_pos=False, rel_pos_zero_init=True,
                 window_size=0, input_size=None):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_cfg=act_cfg)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), 
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, 
                              stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B dim H W -> B H W dim
        return x


class IRAdapter(nn.Module):
    """IR 전용 어댑터 모듈"""
    def __init__(self, embed_dim=768, adapter_dim=None, reduction_ratio=4):
        super().__init__()
        if adapter_dim is None:
            adapter_dim = embed_dim // reduction_ratio
            
        self.adapter_down = nn.Linear(embed_dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, embed_dim)
        self.adapter_act = nn.GELU()
        self.adapter_dropout = nn.Dropout(0.1)
        
        # Layer Scale for adapter
        self.adapter_scale = nn.Parameter(torch.ones(embed_dim) * 0.1)
        
        # 초기화
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        # x: [B, H, W, dim] 또는 [B, seq_len, dim]
        residual = x
        
        x = self.adapter_down(x)
        x = self.adapter_act(x)
        x = self.adapter_dropout(x)
        x = self.adapter_up(x)
        
        # Residual connection with scale
        x = residual + x * self.adapter_scale
        return x


class CrossModalFusion(nn.Module):
    """RGB와 IR feature를 융합하는 모듈"""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-attention for fusion
        self.rgb_proj = nn.Linear(embed_dim, embed_dim)
        self.ir_proj = nn.Linear(embed_dim, embed_dim)
        
        # Fusion weights
        self.fusion_weight = nn.Parameter(torch.ones(2) * 0.5)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_feat, ir_feat):
        # rgb_feat, ir_feat: [B, H, W, dim]
        B, H, W, dim = rgb_feat.shape
        
        # Project features
        rgb_proj = self.rgb_proj(rgb_feat)
        ir_proj = self.ir_proj(ir_feat)
        
        # Simple weighted fusion
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * rgb_proj + weights[1] * ir_proj
        
        # Alternative: concatenation + projection
        concat_feat = torch.cat([rgb_proj, ir_proj], dim=-1)
        fused_alt = self.out_proj(concat_feat)
        
        # Combine both approaches
        final_fused = (fused + fused_alt) * 0.5
        final_fused = self.norm(final_fused)
        
        return final_fused


@MODELS.register_module()
class ProposedViTRGBTv15(BaseModule):
    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_path_rate=0.0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 use_abs_pos=True,
                 use_rel_pos=True,
                 rel_pos_zero_init=True,
                 window_size=0,
                 window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 pretrain_img_size=224,
                 pretrain_use_cls_token=True,
                 adapter_dim=None,
                 init_cfg=None):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.init_cfg = init_cfg
        self.pretrain_size = _pair(pretrain_img_size)
        
        # Patch embedding (공통)
        self.patch_embed_rgb = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim)
        
        self.patch_embed_ir = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim)
        
        # Positional embedding
        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # RGB ViT blocks (freezed)
        self.rgb_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i], norm_cfg=norm_cfg,
                act_cfg=act_cfg, use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(depth)
        ])
        
        # IR ViT blocks (freezed)
        self.ir_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i], norm_cfg=norm_cfg,
                act_cfg=act_cfg, use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(depth)
        ])
        
        # Fusion ViT blocks (freezed)
        self.fusion_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i], norm_cfg=norm_cfg,
                act_cfg=act_cfg, use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(depth)
        ])
        
        # IR Adapter (trainable)
        self.ir_adapter = IRAdapter(embed_dim=embed_dim, adapter_dim=adapter_dim)
        
        # Cross-modal fusion module (trainable)
        self.fusion_module = CrossModalFusion(embed_dim=embed_dim)
        
        # Output projection layers for different feature dimensions
        self.rgb_proj = nn.Linear(embed_dim, 192)  # RGB → 192 dims
        self.ir_proj = nn.Linear(embed_dim, 384)   # Adapted IR → 384 dims  
        self.fusion_proj = nn.Linear(embed_dim, 768)  # Fused → 768 dims
        self.extra_proj = nn.Linear(embed_dim, 768)   # Extra → 768 dims
        
        self._init_weights()
        self._freeze_pretrained_blocks()

    def _init_weights(self):
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize projection layers
        for proj in [self.rgb_proj, self.ir_proj, self.fusion_proj, self.extra_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _freeze_pretrained_blocks(self):
        """Freeze pretrained ViT blocks"""
        for param in self.rgb_blocks.parameters():
            param.requires_grad = False
        for param in self.ir_blocks.parameters():
            param.requires_grad = False
        for param in self.fusion_blocks.parameters():
            param.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch')
            self._init_weights()
        else:
            assert 'checkpoint' in self.init_cfg
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            try:
                if 'model' in ckpt:
                    _state_dict = ckpt['model']
                else:
                    _state_dict = ckpt['state_dict']
            except KeyError:
                logger.error("Neither 'model' nor 'state_dict' found in the checkpoint.")
                raise
            self.load_state_dict(_state_dict, strict=False)

    def forward(self, vis, ir):
        B = vis.shape[0]
        
        # 1. Patch embedding
        rgb_x = self.patch_embed_rgb(vis)  # [B, H, W, dim]
        ir_x = self.patch_embed_ir(ir)     # [B, H, W, dim]
        
        H, W = rgb_x.shape[1], rgb_x.shape[2]
        
        # 2. Add positional embedding
        if self.pos_embed is not None:
            pos_embed = get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (H, W))
            rgb_x = rgb_x + pos_embed
            ir_x = ir_x + pos_embed
        
        # 3. RGB through freezed ViT blocks
        for block in self.rgb_blocks:
            rgb_x = block(rgb_x)
        
        # 4. IR through freezed ViT blocks + Adapter
        for block in self.ir_blocks:
            ir_x = block(ir_x)
        
        # Apply adapter to IR features
        adapted_ir_x = self.ir_adapter(ir_x)
        
        # 5. Fusion of RGB and Adapted IR
        fused_x = self.fusion_module(rgb_x, adapted_ir_x)
        
        # 6. Fused features through freezed ViT blocks
        for block in self.fusion_blocks:
            fused_x = block(fused_x)
        
        # 7. Convert to sequence format and project to different dimensions
        rgb_seq = rgb_x.reshape(B, H * W, -1)      # [B, seq_len, dim]
        ir_seq = adapted_ir_x.reshape(B, H * W, -1)  # [B, seq_len, dim]
        fused_seq = fused_x.reshape(B, H * W, -1)   # [B, seq_len, dim]
        
        # Project to target dimensions
        rgb_feat = self.rgb_proj(rgb_seq)       # [B, seq_len, 192]
        ir_feat = self.ir_proj(ir_seq)          # [B, seq_len, 384]
        fused_feat = self.fusion_proj(fused_seq)  # [B, seq_len, 768]
        extra_feat = self.extra_proj(fused_seq)   # [B, seq_len, 768] - 추가 feature
        
        # 8. Convert back to spatial format for neck
        rgb_out = rgb_feat.transpose(1, 2).view(B, 192, H, W)    # [B, 192, H, W]
        ir_out = ir_feat.transpose(1, 2).view(B, 384, H, W)      # [B, 384, H, W]
        fused_out = fused_feat.transpose(1, 2).view(B, 768, H, W)  # [B, 768, H, W]
        extra_out = extra_feat.transpose(1, 2).view(B, 768, H, W)  # [B, 768, H, W]
        
        # Return list for SimpleFPN: [192, 384, 768, 768] channels
        return [rgb_out, ir_out, fused_out, extra_out]