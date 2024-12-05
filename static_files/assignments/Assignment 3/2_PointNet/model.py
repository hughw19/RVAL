from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet 
# ----------TODO------------

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, d=1024):
        super(PointNetfeat, self).__init__()

        self.d = d

    def forward(self, x):

        n_pts = x.size()[2]

        if self.global_feat:


        else:

            return 

class PointNetCls1024D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls1024D, self).__init__()


    def forward(self, x):

        return F.log_softmax(x, dim=1), vis_feature # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()


    def forward(self, x):


        return F.log_softmax(x, dim=1)





class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.k = k


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        return x

