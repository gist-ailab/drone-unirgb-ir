# ProposedSimpleFPN: 멀티모달 SimpleFPN
# RGB(192ch) + IR(384ch) + Fused(768ch) + Extra(768ch) 입력을 처리
# 기존 SimpleFPN과 동일한 출력 구조 유지

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


class ModalityFusion(nn.Module):
    """멀티모달 feature 융합 모듈"""
    def __init__(self, 
                 in_channels_list: List[int], 
                 out_channels: int,
                 fusion_type: str = 'attention',
                 norm_cfg: OptConfigType = None):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.fusion_type = fusion_type
        
        # 각 모달리티를 공통 차원으로 변환
        self.modality_projections = nn.ModuleList()
        for in_ch in in_channels_list:
            self.modality_projections.append(
                ConvModule(
                    in_ch, out_channels, 1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='GELU'),
                    inplace=False
                )
            )
        
        if fusion_type == 'attention':
            # Attention-based fusion
            self.attention_weights = nn.Sequential(
                nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1),
                nn.GELU(),
                nn.Conv2d(out_channels, len(in_channels_list), 1),
                nn.Softmax(dim=1)
            )
        elif fusion_type == 'channel_attention':
            # Channel attention fusion
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels * len(in_channels_list), out_channels // 4, 1),
                nn.GELU(),
                nn.Conv2d(out_channels // 4, len(in_channels_list), 1),
                nn.Sigmoid()
            )
        
        # Final fusion layer
        self.fusion_conv = ConvModule(
            out_channels, out_channels, 3, padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'),
            inplace=False
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Args:
            features: List of tensors [RGB, IR, Fused, Extra]
        Returns:
            Fused feature tensor
        """
        # Project all modalities to same dimension
        projected_features = []
        for feat, proj in zip(features, self.modality_projections):
            projected_features.append(proj(feat))
        
        if self.fusion_type == 'concat':
            # Simple concatenation + 1x1 conv
            fused = torch.cat(projected_features, dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == 'attention':
            # Attention-based weighted fusion
            concat_feat = torch.cat(projected_features, dim=1)
            attention_weights = self.attention_weights(concat_feat)  # [B, N, H, W]
            
            fused = torch.zeros_like(projected_features[0])
            for i, feat in enumerate(projected_features):
                weight = attention_weights[:, i:i+1, :, :]  # [B, 1, H, W]
                fused += weight * feat
                
        elif self.fusion_type == 'channel_attention':
            # Channel attention fusion
            concat_feat = torch.cat(projected_features, dim=1)
            channel_weights = self.channel_attention(concat_feat)  # [B, N, 1, 1]
            
            fused = torch.zeros_like(projected_features[0])
            for i, feat in enumerate(projected_features):
                weight = channel_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
                fused += weight * feat
                
        elif self.fusion_type == 'add':
            # Simple addition
            fused = sum(projected_features)
            
        else:  # weighted_add
            # Learnable weighted addition
            if not hasattr(self, 'fusion_weights'):
                self.fusion_weights = nn.Parameter(torch.ones(len(projected_features)) / len(projected_features))
            
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, projected_features))
        
        fused = self.fusion_conv(fused)
        return fused


@MODELS.register_module()
class ProposedSimpleFPN(BaseModule):
    """
    멀티모달 SimpleFPN for ViTDet
    
    입력: [RGB(192ch), IR(384ch), Fused(768ch), Extra(768ch)]
    출력: 기존 SimpleFPN과 동일한 구조
    """

    def __init__(self,
                 backbone_channel: int = 768,
                 in_channels: List[int] = [192, 384, 768, 768],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 fusion_type: str = 'attention',  # 'attention', 'channel_attention', 'add', 'weighted_add'
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        assert isinstance(in_channels, list)
        assert len(in_channels) == 4, "Expected 4 input modalities [RGB, IR, Fused, Extra]"
        
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.fusion_type = fusion_type

        # 멀티모달 융합 모듈
        self.modality_fusion = ModalityFusion(
            in_channels_list=in_channels,
            out_channels=backbone_channel,
            fusion_type=fusion_type,
            norm_cfg=norm_cfg
        )

        # ViTDet SimpleFPN과 동일한 구조
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(backbone_channel, backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(backbone_channel // 2, backbone_channel // 4, 2, 2)
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(backbone_channel, backbone_channel // 2, 2, 2)
        )
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # Lateral and FPN convolutions
        fpn_in_channels = [backbone_channel // 4, backbone_channel // 2, backbone_channel, backbone_channel]
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(4):  # 4 scales
            l_conv = ConvModule(
                fpn_in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, ...]:
        """
        Forward function.

        Args:
            inputs: List of 4 tensors [RGB, IR, Fused, Extra]
                   각각의 shape: [B, C_i, H, W] where C_i = [192, 384, 768, 768]
        Returns:
            tuple: Feature maps, each is a 4D-tensor with shape [B, out_channels, H_i, W_i]
        """
        assert len(inputs) == 4, f"Expected 4 inputs, got {len(inputs)}"
        
        # 1. 멀티모달 융합
        fused_feature = self.modality_fusion(inputs)  # [B, backbone_channel, H, W]
        
        # 2. ViTDet SimpleFPN과 동일한 처리
        # Build FPN pyramid from fused feature
        fpn_inputs = []
        fpn_inputs.append(self.fpn1(fused_feature))  # 4x upsampling
        fpn_inputs.append(self.fpn2(fused_feature))  # 2x upsampling  
        fpn_inputs.append(self.fpn3(fused_feature))  # same scale
        fpn_inputs.append(self.fpn4(fused_feature))  # 2x downsampling

        # 3. Build laterals
        laterals = [
            lateral_conv(fpn_inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 4. Build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(4)]

        # 5. Add extra levels if needed
        if self.num_outs > len(outs):
            for i in range(self.num_outs - 4):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
                
        return tuple(outs)


# Alternative: Scale-wise fusion version
@MODELS.register_module()
class ProposedSimpleFPNv2(BaseModule):
    """
    스케일별 융합 버전의 ProposedSimpleFPN
    각 스케일에서 멀티모달 융합을 수행
    """

    def __init__(self,
                 backbone_channel: int = 768,
                 in_channels: List[int] = [192, 384, 768, 768],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 fusion_type: str = 'attention',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        # 각 모달리티별 FPN branches
        self.rgb_fpns = self._build_fpn_branch(in_channels[0], norm_cfg)    # 192ch
        self.ir_fpns = self._build_fpn_branch(in_channels[1], norm_cfg)     # 384ch  
        self.fused_fpns = self._build_fpn_branch(in_channels[2], norm_cfg)  # 768ch
        self.extra_fpns = self._build_fpn_branch(in_channels[3], norm_cfg)  # 768ch

        # 스케일별 융합 모듈들
        fpn_channels = [backbone_channel // 4, backbone_channel // 2, backbone_channel, backbone_channel]
        self.scale_fusions = nn.ModuleList()
        
        for scale_ch in fpn_channels:
            fusion = ModalityFusion(
                in_channels_list=[scale_ch] * 4,  # 각 모달리티에서 같은 채널 수
                out_channels=scale_ch,
                fusion_type=fusion_type,
                norm_cfg=norm_cfg
            )
            self.scale_fusions.append(fusion)

        # Final processing
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for scale_ch in fpn_channels:
            l_conv = ConvModule(
                scale_ch, out_channels, 1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(
                out_channels, out_channels, 3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def _build_fpn_branch(self, in_ch: int, norm_cfg: OptConfigType) -> nn.ModuleList:
        """각 모달리티별 FPN branch 구성"""
        # 입력 채널을 backbone_channel로 변환
        input_proj = nn.Conv2d(in_ch, self.backbone_channel, 1)
        
        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2, self.backbone_channel // 4, 2, 2)
        )
        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2)
        )
        fpn3 = nn.Sequential(nn.Identity())
        fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.ModuleList([input_proj, fpn1, fpn2, fpn3, fpn4])

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, ...]:
        """
        Args:
            inputs: [RGB, IR, Fused, Extra] features
        Returns:
            FPN outputs
        """
        # 각 모달리티별로 FPN 처리
        all_scale_features = []  # [modality][scale]
        
        for modality_input, fpn_branch in zip(inputs, [self.rgb_fpns, self.ir_fpns, self.fused_fpns, self.extra_fpns]):
            # Project input to backbone_channel
            projected = fpn_branch[0](modality_input)  # input_proj
            
            # Apply FPN transforms
            scale_features = []
            scale_features.append(fpn_branch[1](projected))  # fpn1
            scale_features.append(fpn_branch[2](projected))  # fpn2
            scale_features.append(fpn_branch[3](projected))  # fpn3
            scale_features.append(fpn_branch[4](projected))  # fpn4
            
            all_scale_features.append(scale_features)

        # 스케일별로 멀티모달 융합
        fused_scales = []
        for scale_idx in range(4):
            scale_features = [modality_features[scale_idx] for modality_features in all_scale_features]
            fused_scale = self.scale_fusions[scale_idx](scale_features)
            fused_scales.append(fused_scale)

        # Final processing
        laterals = [self.lateral_convs[i](fused_scales[i]) for i in range(4)]
        outs = [self.fpn_convs[i](laterals[i]) for i in range(4)]

        # Add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - 4):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)