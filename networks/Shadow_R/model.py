import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os

from .model_convnext import fusion_net

from .Restormer.restormer_arch import Restormer


class final_net(nn.Module):
    def __init__(self, only_intermediate=False):
        super(final_net, self).__init__()
        self.remove_model = fusion_net()
        if not only_intermediate:
            self.enhancement_model =  Restormer()
        self.only_intermediate = only_intermediate

    def forward(self, input, scale=0.05):
        x = self.remove_model(input)
        if self.only_intermediate:
            return x
        else:
            x_ = (self.enhancement_model(x) * scale + x ) / (1 + scale)
            return x_
