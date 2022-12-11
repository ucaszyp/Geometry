import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from model.vit import forward_flex
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

from model.utils import _make_fusion_block, norm_normalize, sample_points

class DPT_Uncertain(BaseModel):
    def __init__(
        self,
        head,
        sampling_ratio=None,
        importance_ratio=None,
        features=256,
        out_features=4,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT_Uncertain, self).__init__()

        self.sampling_ratio = sampling_ratio
        self.importance_ratio = importance_ratio

        
        # produces 1/8 res output
        # 512 -> 128
        self.out_conv_res8 = nn.Conv2d(features, out_features, kernel_size=3, stride=1, padding=1)
        # F.relu = F.relu()


        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(features + out_features, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, out_features, kernel_size=1),
        )

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(features + out_features, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, out_features, kernel_size=1),
        )

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(features + out_features, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, out_features, kernel_size=1),
        )

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head
        self.out_features = out_features

    def forward(self, x, gt_norm_mask=None, mode='train'):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # 1/8 res
        out_res8_1 = self.out_conv_res8(path_3)     # out_res8: [2, 4, 60, 80]      1/8 res output
        if self.out_features >= 4:
            out_res8 = norm_normalize(out_res8_1)       # out_res8: [2, 4, 60, 80]      1/8 res output
        if self.out_features <= 3:
            pred_depth = F.relu(out_res8_1[:, 0:1, :])
            pred_sigma = out_res8_1[:, 1:, :]
            out_res8 = torch.cat([pred_depth, pred_sigma], dim=1)

        # 1/4 res
        if mode == 'train':
            # upsampling ... out_res8: [2, 4, 60, 80] -> out_res8_res4: [2, 4, 120, 160]
            out_res8_res4 = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res8_res4.shape

            # samples: [B, 1, N, 2]
            point_coords_res4, rows_int, cols_int = sample_points(out_res8_res4.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio, out_feat=self.out_features)

            # output (needed for evaluation / visualization)
            out_res4 = out_res8_res4

            # grid_sample feature-map
            feat_res4 = F.grid_sample(path_3, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 512, 1, N)
            init_pred = F.grid_sample(out_res8, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res4 = torch.cat([feat_res4, init_pred], dim=1)  # (B, 512+4, 1, N)

            # prediction (needed to compute loss)
            sample_pred_res4 = self.out_conv_res4(feat_res4[:, :, 0, :])  # (B, 4, N)
            if self.out_features >= 4:
                samples_pred_res4 = norm_normalize(sample_pred_res4)  # (B, 4, N) - normalized
            if self.out_features <= 3:
                pred_depth = F.relu(sample_pred_res4[:, 0:1, :])
                pred_sigma = sample_pred_res4[:, 1:, :]
                # print(pred_depth.shape, pred_sigma.shape)
                samples_pred_res4 = torch.cat([pred_depth, pred_sigma], dim=1)
                # print(samples_pred_res4.shape)

            for i in range(B):
                out_res4[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res4[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(path_3, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            # try all pixels
            out_res4 = self.out_conv_res4(feat_map.view(B, 256 + self.out_features, -1))  # (B, 4, N)
            if self.out_features >= 4:
                out_res4 = norm_normalize(out_res4)  # (B, 4, N) - normalized
            out_res4 = out_res4.view(B, self.out_features, H, W)
            samples_pred_res4 = point_coords_res4 = None
        
        # 1/2 res
        if mode == 'train':
    
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res4_res2 = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res4_res2.shape

            # samples: [B, 1, N, 2]
            point_coords_res2, rows_int, cols_int = sample_points(out_res4_res2.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio, out_feat=self.out_features)

            # output (needed for evaluation / visualization)
            out_res2 = out_res4_res2

            # grid_sample feature-map
            feat_res2 = F.grid_sample(path_2, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 256, 1, N)
            init_pred = F.grid_sample(out_res4, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res2 = torch.cat([feat_res2, init_pred], dim=1)  # (B, 256+4, 1, N)

            # prediction (needed to compute loss)
            sample_pred_res2 = self.out_conv_res2(feat_res2[:, :, 0, :])  # (B, 4, N)
            if self.out_features >= 4:
                samples_pred_res2 = norm_normalize(sample_pred_res2)  # (B, 4, N) - normalized
            if self.out_features <= 3:
                pred_depth = F.relu(sample_pred_res2[:, 0:1, :])
                pred_sigma = sample_pred_res2[:, 1:, :]
                samples_pred_res2 = torch.cat([pred_depth, pred_sigma], dim=1)   

            for i in range(B):
                out_res2[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res2[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(path_2, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            out_res2 = self.out_conv_res2(feat_map.view(B, 256 + self.out_features, -1))  # (B, 4, N)

            if self.out_features >= 4:
                out_res2 = norm_normalize(out_res2)  # (B, 4, N) - normalized

            out_res2 = out_res2.view(B, self.out_features, H, W)
            samples_pred_res2 = point_coords_res2 = None

        # 1 res
        if mode == 'train':
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res2_res1 = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res2_res1.shape

            # samples: [B, 1, N, 2]
            point_coords_res1, rows_int, cols_int = sample_points(out_res2_res1.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio, out_feat=self.out_features)

            # output (needed for evaluation / visualization)
            out_res1 = out_res2_res1

            # grid_sample feature-map
            feat_res1 = F.grid_sample(path_1, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 128, 1, N)
            init_pred = F.grid_sample(out_res2, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res1 = torch.cat([feat_res1, init_pred], dim=1)  # (B, 128+4, 1, N)

            # prediction (needed to compute loss)
            sample_pred_res1 = self.out_conv_res1(feat_res1[:, :, 0, :])  # (B, 4, N)
            if self.out_features >= 4:
                samples_pred_res1 = norm_normalize(sample_pred_res1)  # (B, 4, N) - normalized
            
            if self.out_features <= 3:
                pred_depth = F.relu(sample_pred_res1[:, 0:1, :])
                pred_sigma = sample_pred_res1[:, 1:, :]
                samples_pred_res1 = torch.cat([pred_depth, pred_sigma], dim=1)     

            for i in range(B):
                out_res1[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res1[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(path_1, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            out_res1 = self.out_conv_res1(feat_map.view(B, 256 + self.out_features, -1))  # (B, 4, N)
            if self.out_features >= 4:
                out_res1 = norm_normalize(out_res1)  # (B, 4, N) - normalized
            out_res1 = out_res1.view(B, self.out_features, H, W)
            samples_pred_res1 = point_coords_res1 = None

        return [out_res8, out_res4, out_res2, out_res1], \
               [out_res8, samples_pred_res4, samples_pred_res2, samples_pred_res1], \
               [None, point_coords_res4, point_coords_res2, point_coords_res1]

class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = path_1
        for i, it in enumerate(self.scratch.output_conv):
            out = it(out)
            if i == 4:
                feature = out.clone()
        return [out], [feature]

class Cerberus(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(Cerberus, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet10 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet11 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet12 = _make_fusion_block(features, use_bn)

class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False), # zyp
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        out = super().forward(x)
        print(out[0][0].max())
        return out[0][0].clamp(min=1e-3, max=10.0)

class DPTNormalUncertainModel(DPT_Uncertain):
    def __init__(self, path=None, sampling_ratio=None, importance_ratio=None,**kwargs):

        features = kwargs["features"] if "features" in kwargs else 128

        kwargs["use_bn"] = True

        head = None
        super().__init__(head, sampling_ratio, importance_ratio, out_features=4, **kwargs)

        if path is not None:
            self.load(path)

class DPTDepthUncertainModel(DPT_Uncertain):
    def __init__(self, path=None, sampling_ratio=None, importance_ratio=None,**kwargs):

        features = kwargs["features"] if "features" in kwargs else 128

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, sampling_ratio, importance_ratio, out_features=2, **kwargs)

        if path is not None:
            self.load(path)

class GeoUncertainModel(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(Cerberus, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)