#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
import re
import torch
import logging
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

_logger = logging.getLogger(__name__)


def Norm(norm_type, num_feats, bn_momentum=0.1, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class Conv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 dimension=3):
        super(Conv, self).__init__()

        self.net = nn.Sequential(ME.MinkowskiConvolution(inplanes,
                                                         planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         dilation=dilation,
                                                         bias=bias,
                                                         dimension=dimension),)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v, return_attn=False):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
        x = self.proj(x)
        if return_attn:
            return x, attn
        
        return x

class AdaptivePooling(nn.Module):
    def __init__(self, feature_dim, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.query = nn.Parameter(torch.randn(output_channels, feature_dim))

    def forward(self, x, return_weights=False):
        """
        Args:
            x: Input tensor of shape (batch_size, input_channels, feature_dim)

        Returns:
            Output tensor of shape (batch_size, output_channels, feature_dim)
        """
        query = self.query.unsqueeze(0).repeat(x.shape[0],1,1)

        out = F.scaled_dot_product_attention(query=query,key=x,value=x)
        if return_weights:
            attn_scores = torch.einsum('ij,bkj->bki', self.query, x)
            attn_weights = F.softmax(attn_scores, dim=1)
            return out, attn_weights

        return out


class Encoder(ME.MinkowskiNetwork):
    """
    FCN encoder, used to extract features from the input point clouds.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels, norm_type, D=3):
        super(Encoder, self).__init__(D)

        self.in_channels = 3
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.conv_planes = [32, 64, 128, 256, 256, 256, 256, 512, 512]

        # in_channels, conv_planes, kernel_size, stride  dilation  bias
        self.conv1 = Conv(self.in_channels, self.conv_planes[0], 3, 1, 1, True)
        self.conv2 = Conv(self.conv_planes[0], self.conv_planes[1], 3, 2, bias=True)
        self.conv3 = Conv(self.conv_planes[1], self.conv_planes[2], 3, 2, bias=True)
        self.conv4 = Conv(self.conv_planes[2], self.conv_planes[3], 3, 2, bias=True)

        self.res1_conv1 = Conv(self.conv_planes[3], self.conv_planes[4], 3, 1, bias=True)
        # 1
        self.res1_conv2 = Conv(self.conv_planes[4], self.conv_planes[5], 1, 1, bias=True)
        self.res1_conv3 = Conv(self.conv_planes[5], self.conv_planes[6], 3, 1, bias=True)

        self.res2_conv1 = Conv(self.conv_planes[6], self.conv_planes[7], 3, 1, bias=True)
        # 2
        self.res2_conv2 = Conv(self.conv_planes[7], self.conv_planes[8], 1, 1, bias=True)
        self.res2_conv3 = Conv(self.conv_planes[8], self.out_channels, 3, 1, bias=True)

        self.res2_skip = Conv(self.conv_planes[6], self.out_channels, 1, 1, bias=True)

    def forward(self, x):

        x = MEF.relu(self.conv1(x))
        x = MEF.relu(self.conv2(x))
        x = MEF.relu(self.conv3(x))
        res = MEF.relu(self.conv4(x))

        x = MEF.relu(self.res1_conv1(res))
        x = MEF.relu(self.res1_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res1_conv3(x))

        res = res + x

        x = MEF.relu(self.res2_conv1(res))
        x = MEF.relu(self.res2_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


def one_hot(x, N):
    one_hot = torch.FloatTensor(x.size(0), N, x.size(1), x.size(2)).zero_().to(x.device)
    one_hot = one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


class CondLayer(nn.Module):
    """
    pixel-wise feature modulation.
    """
    def __init__(self, in_channels):
        super(CondLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x, gammas, betas):
        return F.relu(self.bn((gammas * x) + betas))

def linear(in_dim, out_dim, bias=True):
    return nn.Sequential(
        nn.Linear(
            in_dim, out_dim, bias),
        nn.ReLU(inplace=True),
    )

class FeatureMixerLayer(nn.Module):
    def __init__(self, num_token, token_dim, mlp_ratio):
        super().__init__()
        # Token Mixing (Point Mixing): (B, N, C) -> (B, C, N) -> MLP(N) -> (B, N, C)
        self.expanded_dim_t = int(num_token * mlp_ratio)
        self.mix_t = nn.Sequential(
            nn.LayerNorm(token_dim),
            Rearrange('b n c -> b c n'),
            nn.Linear(num_token, self.expanded_dim_t),
            nn.GELU(),
            nn.Linear(self.expanded_dim_t, num_token),
            Rearrange('b c n -> b n c'),
        )
        
        # Channel Mixing (Feature Mixing): (B, N, C) -> MLP(C) -> (B, N, C)
        self.expanded_dim_c = int(token_dim * mlp_ratio)
        self.mix_c = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.expanded_dim_c), # C차원에 대해 동작
            nn.GELU(),
            nn.Linear(self.expanded_dim_c , token_dim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x + self.mix_t(x)
        return x + self.mix_c(x)


class MLPDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MLPDecoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activataions = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            self.dropouts.append(nn.Dropout(0.08))
            self.activataions.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            act = self.activataions[i]
            # x  = F.relu(bn(fc(x)))  # [B, D]
            x = act(bn(fc(x)))
            x = self.dropouts[i](x)

        return x

class Cls_Head(nn.Module):
    def __init__(self, in_channels=512):
        super(Cls_Head, self).__init__()
        channels = in_channels
        num_mixer_layers = 1
        pose_regressor_layers = 6
        gdim_scale = 0.5 # Default value is 2 in FlashMix Paper

        # Adpative Pooling layer
        self.adaptive_pooling = AdaptivePooling(feature_dim=channels, output_channels=128)
        
        # Mixer layers
        self.mixer = nn.Sequential(
            *[  
                FeatureMixerLayer(128, channels, 1)
                for _ in range(num_mixer_layers)
            ],
            nn.LayerNorm(channels),
            linear(channels, channels)
        )

        # Pose prediction layers
        self.decoder = MLPDecoder(channels, [channels] * pose_regressor_layers)
        self.fct = nn.Linear(channels, 3)
        self.fcq = nn.Linear(channels, 3)

        # Projected embedding for contrastive learning
        self.emb_prejector = nn.Sequential(
            nn.Linear(channels, int(gdim_scale * channels)),
            nn.ReLU(),
            nn.Linear(int(gdim_scale * channels), int(gdim_scale * channels)),
        )
    
    def get_aggregated_features(self, local_features, batch_ids):
        num_batch = len(torch.unique(batch_ids))
        feat_dim = local_features.shape[1]
        local_features_BNC = torch.empty((num_batch, 128, feat_dim), device=local_features.device)
        
        for batch_id in torch.unique(batch_ids):
            # Get batch samples
            batch_indices = torch.where(batch_ids == batch_id)[0]
            local_features_batch = local_features[batch_indices]
            # Adaptive pooling
            local_features_batch_pooled = self.adaptive_pooling(local_features_batch.unsqueeze(0), return_weights=False)
            local_features_BNC[batch_id, :, :] = local_features_batch_pooled.squeeze(0)
        
        emb_local = self.mixer(local_features_BNC)
        # Global Average Pooling
        emb = emb_local.mean(dim=1)

        return emb

    def forward(self, local_features, batch_ids, return_emb=True):
        num_batch = len(torch.unique(batch_ids))
        feat_dim = local_features.shape[1]
        local_features_BNC = torch.empty((num_batch, 128, feat_dim), device=local_features.device)
        
        for batch_id in torch.unique(batch_ids):
            # Get batch samples
            batch_indices = torch.where(batch_ids == batch_id)[0]
            local_features_batch = local_features[batch_indices]
            # Adaptive pooling
            local_features_batch_pooled = self.adaptive_pooling(local_features_batch.unsqueeze(0), return_weights=False)
            local_features_BNC[batch_id, :, :] = local_features_batch_pooled.squeeze(0)
        
        emb_local = self.mixer(local_features_BNC)

        # Global Average Pooling
        emb = emb_local.mean(dim=1)

        # 6-DOF pose prediction
        y = self.decoder(emb) 
        t = self.fct(y)
        q = self.fcq(y)

        # Projected embedding for triplet loss
        projected_emb = self.emb_prejector(emb)

        if return_emb:
            return t, q, projected_emb
        return t, q

class Reg_Head(nn.Module):
    """
    nn.Linear版
    """
    def __init__(self, num_head_blocks, in_channels=512, mlp_ratio=1.0, feature_dim=512):
        super(Reg_Head, self).__init__()
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = in_channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels \
            else nn.Linear(self.in_channels, self.head_channels)

        block_channels = int(self.head_channels * mlp_ratio)
        self.res3_conv1 = nn.Linear(self.in_channels, self.head_channels)
        self.res3_conv2 = nn.Linear(self.head_channels, block_channels)
        self.res3_conv3 = nn.Linear(block_channels, self.head_channels)

        self.res_blocks = []
        self.norm_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Linear(self.head_channels, self.head_channels),
                nn.Linear(self.head_channels, block_channels),
                nn.Linear(block_channels, self.head_channels),
            ))

            super(Reg_Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Reg_Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Reg_Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Linear(self.head_channels, self.head_channels)
        self.fc2 = nn.Linear(self.head_channels, block_channels)
        self.fc3 = nn.Linear(block_channels, 3)
        
        # Self-Attention -> Instead of projection layer
        self.self_attn = Attention(512, num_heads=8, proj_drop=0.1)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout_attn = nn.Dropout(0.1)
        self.dropout_ffn = nn.Dropout(0.1)
        self.ffn = nn.Sequential( # Hardcoded
            nn.Linear(512, 512 * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512 * 4, 512)
        )
    
    def apply_self_attention(self, res_local, batch_ids):
        # Local feature self-attention for Student
        res_attn = torch.empty_like(res_local)
        for batch_id in torch.unique(batch_ids):
            # Get batch samples
            batch_indices = torch.where(batch_ids == batch_id)[0]
            local_features_batch = res_local[batch_indices]
            # Pre-norm
            x = local_features_batch.unsqueeze(0)
            q = k = v = self.norm1(x)
            # Self Attention (1, N, C)
            attn_out = self.self_attn(q, k, v)
            # Residual Connection
            x = x + self.dropout_attn(attn_out)
            # Pass the FFN
            x = x + self.dropout_ffn(self.ffn(self.norm2(x)))
            # Normalization
            attn_out = F.normalize(x.squeeze(0), p=2, dim=-1)
            # Residual Connection
            res_attn[batch_indices] = local_features_batch + 0.1 * attn_out
        
        return res_attn

    def forward(self, res):
        # Pass the teacher regressor head with cross-attentioned features
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))
        res = self.head_skip(res) + x

        for res_block in self.res_blocks:

            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        return sc

class Regressor(ME.MinkowskiNetwork):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, num_head_blocks, num_encoder_features, level_clusters=25,
                 mlp_ratio=1.0, sample_cls=False, D=3):
        super(Regressor, self).__init__(D)

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim, norm_type='BN')
        self.cls_heads = Cls_Head(in_channels=self.feature_dim)
        if not sample_cls:
            self.reg_heads = Reg_Head(num_head_blocks=num_head_blocks, in_channels=self.feature_dim + level_clusters,
                                      mlp_ratio=mlp_ratio, feature_dim=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, classifier_state_dict=None,
                            num_head_blocks=None, level_clusters=25, mlp_ratio=1.0, sample_cls=False):
        num_encoder_features = encoder_state_dict['res2_conv3.net.0.bias'].shape[1]
        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(num_head_blocks, num_encoder_features, level_clusters, mlp_ratio, sample_cls)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        if classifier_state_dict!=None:
            regressor.cls_heads.load_state_dict(classifier_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^reg_heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.net.0.bias'].shape[1]
        num_decoder_features = num_encoder_features
        head_channels = num_encoder_features
        
        if 'reg_heads.res3_conv1.weight' in state_dict:
            reg_in_channels = state_dict['reg_heads.res3_conv1.weight'].shape[1]
            level_clusters = reg_in_channels - num_encoder_features
        else:
            level_clusters = 25 # Default fallback
        
        reg = any(key.startswith("reg_heads") for key in state_dict)
        if reg:
            mlp_ratio = state_dict['reg_heads.res3_conv2.weight'].shape[0] / \
                        state_dict['reg_heads.res3_conv2.weight'].shape[1]
        else:
            mlp_ratio = 1

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tEncoder feature size: {num_encoder_features}"
                     f"\n\tDecoder feature size: {num_decoder_features}"
                     f"\n\tHead channels: {head_channels}"
                     f"\n\tMLP ratio: {mlp_ratio}")
        regressor = cls(num_head_blocks, num_encoder_features, mlp_ratio=mlp_ratio, level_clusters=level_clusters)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, cls_head_state_dict, reg_head_state_dict=None):
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in cls_head_state_dict.items():
            merged_state_dict[f"cls_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        if reg_head_state_dict != None:
            for k, v in reg_head_state_dict.items():
                merged_state_dict[f"reg_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        return cls.create_from_state_dict(merged_state_dict)


    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        out = self.reg_heads(features)
        return out
    def get_aggregated_features(self, features, batch_ids):
        return self.cls_heads.get_aggregated_features(features, batch_ids)

    def get_poses_and_projection_embedding(self, features, batch_ids):
        out = self.cls_heads(features, batch_ids)
        return out

    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.encoder(inputs)
        out = self.get_scene_coordinates(features.F)
        out = ME.SparseTensor(
            features=out,
            coordinates=features.C,
        )

        return {'pred': out}