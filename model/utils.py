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


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

def norm_normalize(norm_out):
    min_kappa = 0.01
    norm_x, norm_y, norm_z, kappa = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    kappa = F.elu(kappa) + 1.0 + min_kappa
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
    return final_out

@torch.no_grad()
def sample_points(init_normal, gt_norm_mask, sampling_ratio, beta, out_feat):
    device = init_normal.device
    B, _, H, W = init_normal.shape
    N = int(sampling_ratio * H * W)
    beta = beta

    # uncertainty map
    uncertainty_map = -1 * init_normal[:, out_feat - 1, :, :]  # B, H, W

    # gt_invalid_mask (B, H, W)
    if gt_norm_mask is not None:
        gt_invalid_mask = F.interpolate(gt_norm_mask.float(), size=[H, W], mode='nearest')
        gt_invalid_mask = gt_invalid_mask[:, 0, :, :] < 0.5
        uncertainty_map[gt_invalid_mask] = -1e4

    # (B, H*W)
    _, idx = uncertainty_map.view(B, -1).sort(1, descending=True)

    # importance sampling
    if int(beta * N) > 0:
        importance = idx[:, :int(beta * N)]    # B, beta*N

        # remaining
        remaining = idx[:, int(beta * N):]     # B, H*W - beta*N

        # coverage
        num_coverage = N - int(beta * N)

        if num_coverage <= 0:
            samples = importance
        else:
            coverage_list = []
            for i in range(B):
                idx_c = torch.randperm(remaining.size()[1])    # shuffles "H*W - beta*N"
                coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))     # 1, N-beta*N
            coverage = torch.cat(coverage_list, dim=0)                                      # B, N-beta*N
            samples = torch.cat((importance, coverage), dim=1)                              # B, N

    else:
        # remaining
        remaining = idx[:, :]  # B, H*W

        # coverage
        num_coverage = N

        coverage_list = []
        for i in range(B):
            idx_c = torch.randperm(remaining.size()[1])  # shuffles "H*W - beta*N"
            coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))  # 1, N-beta*N
        coverage = torch.cat(coverage_list, dim=0)  # B, N-beta*N
        samples = coverage

    # point coordinates
    rows_int = samples // W         # 0 for first row, H-1 for last row
    rows_float = rows_int / float(H-1)         # 0 to 1.0
    rows_float = (rows_float * 2.0) - 1.0       # -1.0 to 1.0

    cols_int = samples % W          # 0 for first column, W-1 for last column
    cols_float = cols_int / float(W-1)         # 0 to 1.0
    cols_float = (cols_float * 2.0) - 1.0       # -1.0 to 1.0

    point_coords = torch.zeros(B, 1, N, 2)
    point_coords[:, 0, :, 0] = cols_float             # x coord
    point_coords[:, 0, :, 1] = rows_float             # y coord
    point_coords = point_coords.to(device)
    return point_coords, rows_int, cols_int