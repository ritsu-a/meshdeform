from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

"""
Template Discovery -> Patch deform
Tamplate learning -> Point translation 
"""


class PointNetfeat(nn.Module):
    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


class patchDeformationMLP(nn.Module):
    """ Deformation of a 2D patch into a 3D surface """

    def __init__(self, patchDim=2, patchDeformDim=3, tanh=True):

        super(patchDeformationMLP, self).__init__()
        layer_size = 128
        self.tanh = tanh
        self.conv1 = torch.nn.Conv1d(patchDim, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, patchDeformDim, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size)
        self.bn2 = torch.nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        print("bottleneck_size", bottleneck_size)
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2 * self.th(self.conv4(x))
        return x




class AE_AtlasNet_Humans(nn.Module):
    def __init__(self, num_points=6890, bottleneck_size=1024, point_translation=False, dim_template=3,
                 patch_deformation=False, dim_out_patch=3, start_from="TEMPLATE", dataset_train=None):
        super(AE_AtlasNet_Humans, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.point_translation = point_translation
        self.dim_template = dim_template
        self.patch_deformation = patch_deformation
        self.dim_out_patch = dim_out_patch
        self.dim_before_decoder = 3
        self.count = 0
        self.start_from = start_from
        self.dataset_train = dataset_train


        self.dim_before_decoder = dim_template


        

        self.encoder = PointNetfeat(num_points, bottleneck_size).double()
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size).double()])

    def morph_points(self, x, data_batch, idx=None):
        if not idx is None:
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)

        rand_grid = data_batch["template"] # 6890, 3
        if not idx is None:
            rand_grid = rand_grid[idx, :]  # batch x 2500, 3
            rand_grid = rand_grid.view(x.size(0), -1, self.dim_template).transpose(1,
                                                                                   2).contiguous()  # batch , 2500, 3 -> batch, 6980, 3
        else:
            rand_grid = rand_grid.transpose(1, 2).contiguous() # 3, 6980 -> 1,3,6980 -> batch, 3, 6980

        if self.patch_deformation:
            rand_grid = self.templateDiscovery[0](rand_grid)
        if self.point_translation:
            if idx is None:
                trans = self.template[0].vertex_trans.unsqueeze(0).transpose(1,2).contiguous().expand(x.size(0), self.dim_template, -1)
            else:
                trans = self.template[0].vertex_trans[idx, :].view(x.size(0), -1, self.dim_template).transpose(1,2).contiguous()
            rand_grid = rand_grid + trans

        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        return self.decoder[0](y).contiguous().transpose(2, 1).contiguous()  # batch, 1, 3, num_point


    def decode(self, x, data_batch, idx=None):
        return self.morph_points(x, data_batch, idx)

    def forward(self, data_dict, idx=None):
        x = self.encoder(data_dict["noisy_pcd"].transpose(1, 2))
        return self.decode(x, data_dict, idx)

    


if __name__ == '__main__':
    a = AE_AtlasNet_Humans()