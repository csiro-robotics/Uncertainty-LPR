import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import functools
import numpy as np
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from typing import List

import loupe as lp
from models.ppt_transformer.libs.pointops.functions import pointops
from util import pt_util

from torchpack.utils.config import configs 

__all__ = ['Network']


class Network(nn.Module):
    def __init__(self, param=None):
        super(Network, self).__init__()
        # backbone
        self.backbone = PointNet2(param=param)
        
        # global descriptor
        aggregation = configs.model.aggregator.NAME     # param["AGGREGATION"]

        if aggregation == 'spvlad':
            self.aggregation = lp.SpatialPyramidNetVLAD(
                feature_size = configs.model.aggregator.FEATURE_SIZE, #param["FEATURE_SIZE"],         # 256,256,256,256
                max_samples = configs.model.aggregator.MAX_SAMPLES,     # param["MAX_SAMPLES"],           # 64,256,1024,4096
                cluster_size=configs.model.aggregator.CLUSTER_SIZE,     # param["CLUSTER_SIZE"],         # 1,4,16,64
                output_dim=configs.model.aggregator.OUTPUT_DIM,     # param["OUTPUT_DIM"],             # 256,256,256,256
                gating=configs.model.aggregator.GATING,     # param['GATING'],                     # True
                add_batch_norm=True
            )
        else:
            print("No aggregation algorithm: ", aggregation)

    def forward(self, batch, return_feat=False):
        r"""
        x: B x 1 x N x 3
        """
        x = batch['cloud']
        # x = x.squeeze(1)
        f0, f1, f2, f3 = self.backbone(x)

        x = self.aggregation(f0, f1, f2, f3)   # B x C0x64x1, BxC1x256, BxC2x1024, BxC3x4096 -> Bx256
        
        if return_feat:
            return x, feature
        else:
            return x

class PointNet2(nn.Module):
    def __init__(self, param=None):
        super().__init__()
        c = 3
        k = 13
        use_xyz = True
        self.SA_modules = nn.ModuleList()
        sap = configs.model.pointnet2.SAMPLING #param['SAMPLING']
        knn = configs.model.pointnet2.KNN #param['KNN']
        fs = configs.model.pointnet2.FEATURE_SIZE #param['FEATURE_SIZE']
        gp = configs.model.pointnet2.GROUP #param['GROUP']
        self.SA_modules.append(PointNet2SAModule(npoint=sap[0], nsample=knn[0], gp=gp, mlp=[c, 32, 32, 64], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=sap[1], nsample=knn[1], gp=gp, mlp=[64, 64, 64, 128], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=sap[2], nsample=knn[2], gp=gp, mlp=[128, 128, 128, 256], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=sap[3], nsample=knn[3], gp=gp, mlp=[256, 256, 256, 512], use_xyz=use_xyz))
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[1] + c, 256, 256, fs[0]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[2] + 64, 256, fs[1]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[3] + 128, 256, fs[2]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, fs[3]]))

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        l_xyz, l_features = [pointcloud], [pointcloud.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        
        # l3: B x C x 64
        # l2: B x C x 256
        # l1: B x C x 1024
        # l0: B x C x 4096
        return l_features[3].unsqueeze(-1), l_features[2].unsqueeze(-1), l_features[1].unsqueeze(-1), l_features[0].unsqueeze(-1)

class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.sas = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()    # B x 3 x N
        center_idx = pointops.furthestsampling(xyz, self.npoint)
        new_xyz = pointops.gathering(
            xyz_trans,
            center_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None
        
        center_features = pointops.gathering(
            features,
            center_idx
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features, center_features)            # B x C x M x K
            new_features = self.mlps[i](new_features)   # B x C' x M x K
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])    # B x C' x M x 1
            new_features = new_features.squeeze(-1)     # B x C' x M
            g_features = self.sas[i](new_features)      # B x C' x M
            new_features_list.append(g_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], gp: int, bn: bool = True, use_xyz: bool = True):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.sas = nn.ModuleList()      # self-attention list
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup_Edge(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointops.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_util.SharedMLP(mlp_spec, bn=bn))
            self.sas.append(SA_Layer(mlp_spec[-1], gp))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, gp: int = None, bn: bool = True, use_xyz: bool = True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], gp=gp, bn=bn, use_xyz=use_xyz)

class SA_Layer(nn.Module):
    def __init__(self, channels, gp):
        super().__init__()
        mid_channels = channels
        self.gp = gp
        assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""
        x: B x C x N
        """
        bs, ch, nums = x.size()
        x_q = self.q_conv(x)                                # B x C x N
        x_q = x_q.reshape(bs, self.gp, ch//self.gp, nums)
        x_q = x_q.permute(0, 1, 3, 2)                       # B x gp x num x C'

        x_k = self.k_conv(x)                                # B x C x N        
        x_k = x_k.reshape(bs, self.gp, ch//self.gp, nums)   # B x gp x C' x nums

        x_v = self.v_conv(x)
        energy = torch.matmul(x_q, x_k)                     # B x gp x N x N 
        energy = torch.sum(energy, dim=1, keepdims=False)
                
        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        x_r = torch.matmul(x_v, attn)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class PointNet2FPModule(nn.Module):
    r"""Propigates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_util.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)
