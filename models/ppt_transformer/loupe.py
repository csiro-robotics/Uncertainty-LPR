import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLADBase(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLADBase, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        
        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))

        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        
        self.hidden1_weights = nn.Parameter(
            torch.randn(feature_size*cluster_size, output_dim)* 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))     # attention initialization 
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()                          # B x 1024 x N x 1 -> B x 1 x N x 1024
        x = x.view((-1, self.max_samples, self.feature_size))       # B x N x 1024

        activation = torch.matmul(x, self.cluster_weights)          # B x N x 1024 X 1024 x 64 -> B x N x 64
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)     # B x N x 64 -> BN x 64
            activation = self.bn1(activation)                       # BN x 64
            activation = activation.view(-1, self.max_samples, self.cluster_size)   # BN x 64 -> B x N x 64
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases           # B x N x 64 + 64 -> B x N x 64
        
        activation = self.softmax(activation)                       # B x N x 64 --(dim=-1)--> B x N x 64
        # activation = activation[:,:,:64]
        activation = activation.view((-1, self.max_samples, self.cluster_size))     # B x N x 64

        a_sum = activation.sum(-2, keepdim=True)    # B x N x K --(dim=-2)--> B x 1 x K
        a = a_sum * self.cluster_weights2           # B x 1 x K X 1 x C x K -> B x C x K
                                                    # element-wise multiply, broadcast mechanism


        activation = torch.transpose(activation, 2, 1)  # B x N x 64 -> B x 64 x N
        
        x = x.view((-1, self.max_samples, self.feature_size))   # B x N x C -> B x N x C
        vlad = torch.matmul(activation, x)                      # B x K x N X B x N x C -> B x K x C
        vlad = torch.transpose(vlad, 2, 1)                      # B x K x C -> B x C x K
        vlad = vlad - a                                         # B x C x K - B x C x K -> B x C x K

        vlad = F.normalize(vlad, dim=1, p=2).contiguous()               # B x C x K -> B x C x K
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))   # B x (C*K)
        return vlad

class SpatialPyramidNetVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(SpatialPyramidNetVLAD, self).__init__()
        # max_samples[0] = 64
        self.vlad0 = NetVLADBase(feature_size[0], max_samples[0], cluster_size[0], output_dim[0], gating, add_batch_norm)
        # max_samples[1] = 256
        self.vlad1 = NetVLADBase(feature_size[1], max_samples[1], cluster_size[1], output_dim[1], gating, add_batch_norm)
        # max_samples[2] = 1024
        self.vlad2 = NetVLADBase(feature_size[2], max_samples[2], cluster_size[2], output_dim[2], gating, add_batch_norm)
        # max_samples[3] = 4096
        self.vlad3 = NetVLADBase(feature_size[3], max_samples[3], cluster_size[3], output_dim[3], gating, add_batch_norm)
        
        sum_cluster_size = cluster_size[0] + cluster_size[1] + cluster_size[2] + cluster_size[3]
        self.hidden_weights = nn.Parameter(torch.randn(feature_size[0]*sum_cluster_size, output_dim[0])* 1 / math.sqrt(feature_size[0]))

        self.bn2 = nn.BatchNorm1d(output_dim[0])
        self.gating = gating
        if self.gating:
            self.context_gating = GatingContext(output_dim[0], add_batch_norm=add_batch_norm)

    def forward(self, f0, f1, f2, f3):
        v0 = self.vlad0(f0)
        v1 = self.vlad1(f1)
        v2 = self.vlad2(f2)
        v3 = self.vlad3(f3)
        vlad = torch.cat((v0, v1, v2, v3), dim=-1)
        vlad = torch.matmul(vlad, self.hidden_weights)      # B x (1024*64) X (1024*64) x 256 -> B x 256
        vlad = self.bn2(vlad)                               # B x 256 -> B x 256
        
        if self.gating:
            vlad = self.context_gating(vlad)                # B x 256 -> B x 256
        return vlad                                         # B x 256
        
class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)    # B x 256 X 256 x 256 -> B x 256

        if self.add_batch_norm:
            gates = self.bn1(gates)                     # B x 256 -> B x 256
        else:
            gates = gates + self.gating_biases          # B x 256 + 256 -> B x 256

        gates = self.sigmoid(gates)                     # B x 256 -> B x 256

        activation = x * gates                          # B x 256 * B x 256 -> B x 256

        return activation
