import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from DBLLNet.Layers import Convolutional_Layer, Fully_Connected_Layer
from DBLLNet.bilateral_slicing_op.bsliceapply import Slicing_Apply_Function


class Lr_Splat(nn.Module):  # Extract low-level features
    def __init__(self, cm, sb, lb, bn, lris):
        super(Lr_Splat, self).__init__()
        n_conv_layers = int(np.log2(lris / sb))  # Number of conv layers required to reduce the spatial size to sb
        self.splat_layers = nn.ModuleList()
        in_channels = 3
        for i in range(n_conv_layers):
            b_n = bn if i > 0 else False
            out_channels = cm * (2 ** i) * lb
            self.splat_layers.append(Convolutional_Layer(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=3, stride=2, batch_norm=b_n))
            in_channels = out_channels

    def forward(self, x):
        out = x
        for layer in self.splat_layers:
            out = layer(out)
        return out


class Lr_LocalFeatures(nn.Module):  # Local features in low-res stream
    def __init__(self, cm, sb, lb, bn, lris):
        super(Lr_LocalFeatures, self).__init__()
        n_lr_splat_channels = int(cm * (2 ** int(np.log2(lris / sb) - 1)) * lb)
        self.lf_layers = nn.ModuleList()
        b_n = bn if bn else False
        self.lf_layers.append(Convolutional_Layer(in_channels=n_lr_splat_channels,
                                                  out_channels=n_lr_splat_channels,
                                                  kernel_size=3, stride=1, batch_norm=b_n))
        self.lf_layers.append(Convolutional_Layer(in_channels=n_lr_splat_channels,
                                                  out_channels=n_lr_splat_channels,
                                                  kernel_size=3, stride=1, activation=None))

    def forward(self, x):
        out = x
        for layer in self.lf_layers:
            out = layer(out)
        return out


class Lr_GlobalFeatures(nn.Module):  # Global features in low-res stream
    def __init__(self, cm, sb, lb, bn, lris):
        super(Lr_GlobalFeatures, self).__init__()
        n_lr_splat_channels = int(cm * (2 ** int(np.log2(lris / sb) - 1)) * lb)
        n_splat_conv_layers = int(np.log2(lris / sb))
        n_lrgf_conv_layers = int(np.log2(sb / 4))
        self.gf_conv_layers = nn.ModuleList()
        self.gf_fc_layers = nn.ModuleList()
        b_n = bn if bn else False
        # Convolution Layers
        for i in range(n_lrgf_conv_layers):
            self.gf_conv_layers.append(Convolutional_Layer(in_channels=n_lr_splat_channels,
                                                           out_channels=n_lr_splat_channels,
                                                           kernel_size=3, stride=2, batch_norm=b_n))
        # Fully Connected Layers
        n_prev_layer_size = int((lris / 2 ** (n_splat_conv_layers + n_lrgf_conv_layers)) ** 2)
        self.gf_fc_layers.append(Fully_Connected_Layer(in_features=n_prev_layer_size * n_lr_splat_channels,
                                                       out_features=32 * cm * lb,
                                                       batch_norm=b_n))
        self.gf_fc_layers.append(Fully_Connected_Layer(in_features=32 * cm * lb,
                                                       out_features=16 * cm * lb,
                                                       batch_norm=b_n))
        self.gf_fc_layers.append(Fully_Connected_Layer(in_features=16 * cm * lb,
                                                       out_features=8 * cm * lb, activation=None))

    def forward(self, x):
        out = x
        for layer in self.gf_conv_layers:
            out = layer(out)
        out = out.view(list(out.size())[0], -1)  # keep batch size
        # print(out.shape)
        for layer in self.gf_fc_layers:
            out = layer(out)
        return out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.Relu = nn.ReLU()

    def forward(self, LrLocalFeatures, LrGlobalFeatures):
        Rs_LrGlobalFeatures = LrGlobalFeatures.view(list(LrGlobalFeatures.size())[0], list(LrGlobalFeatures.size())[1],
                                                    1, 1)  # Pytorch: [batch size, channel, size, size]
        # print(Rs_LrGlobalFeatures.shape)
        out = torch.add(LrLocalFeatures, Rs_LrGlobalFeatures)
        out = self.Relu(out)
        # print(out.shape)
        return out


class LinearPredict_BGrid(nn.Module):
    def __init__(self, cm, lb, nin=4, nout=3):
        super(LinearPredict_BGrid, self).__init__()
        self.lb = lb
        self.conv = Convolutional_Layer(in_channels=8 * cm * lb, out_channels=lb * nin * nout,
                                        kernel_size=1, stride=1, padding=0, activation=None)  # No batch norm

    def forward(self, x):
        batch_size = list(x.size())[0]
        out = x
        out = self.conv(out)  # [batch_size, 96, 16, 16]
        out = torch.stack(tensors=torch.split(tensor=out, split_size_or_sections=self.lb, dim=1), dim=1)  # unroll grid
        # print(out.shape)
        return out


class Guide_PointwiseNN(nn.Module):
    def __init__(self, bn, guide_complexity=16):
        super(Guide_PointwiseNN, self).__init__()
        b_n = bn if bn else False
        self.conv1 = Convolutional_Layer(in_channels=3, out_channels=guide_complexity,
                                         kernel_size=1, stride=1, padding=0, batch_norm=b_n)
        self.conv2 = Convolutional_Layer(in_channels=guide_complexity, out_channels=1,
                                         kernel_size=1, stride=1, padding=0, activation=nn.Sigmoid)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        return out.squeeze(1)


class SliceNApply(nn.Module):
    def __init__(self):
        super(SliceNApply, self).__init__()

    def forward(self, Bilterial_Grid, Guide, fr):
        return Slicing_Apply_Function.apply(Bilterial_Grid, Guide, fr)


class Net(nn.Module):
    def __init__(self, _cm, _sb, _lb, _bn, _lris):
        super(Net, self).__init__()
        self.splat = Lr_Splat(cm=_cm, sb=_sb, lb=_lb, bn=_bn, lris=_lris)
        self.localf = Lr_LocalFeatures(cm=_cm, sb=_sb, lb=_lb, bn=_bn, lris=_lris)
        self.globalf = Lr_GlobalFeatures(cm=_cm, sb=_sb, lb=_lb, bn=_bn, lris=_lris)
        self.fusion = Fusion()
        self.bgrid = LinearPredict_BGrid(cm=_cm, lb=_lb)
        self.guide = Guide_PointwiseNN(bn=_bn)
        self.slice_op = SliceNApply()

    def forward(self, lr, fr):
        out = self.splat(lr)
        local_out = self.localf(out)
        global_out = self.globalf(out)
        fus_out = self.fusion(local_out, global_out)
        bg_out = self.bgrid(fus_out)
        g_out = self.guide(fr)
        fin_out = self.slice_op(bg_out, g_out, fr)
        return g_out, fin_out