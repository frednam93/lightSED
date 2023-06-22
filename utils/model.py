import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from copy import deepcopy
#from conformer.encoder import ConformerBlock as ConformerBlock2



class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        # ores = x * sig
        return res


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


########################################################################################################################
#                                                        DYconv                                                        #
########################################################################################################################


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, reduction=4, pool_dim='freq'):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, kernel_size, stride, n_basis_kernels, temperature, reduction, pool_dim)

        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes,
                                               self.kernel_size, self.kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes), requires_grad=True)
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x):                                           # x size : [bs, in_chan, frames, freqs]
        attention = self.attention(x)                               # attention size : [bs, n_ker, 1, 1, freqs]

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size)
                                                                    # weight size : [n_ker * out_chan, in_chan, ks, ks]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
                                                                    # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(x.size(0), self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
                                                                   # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq']:
            assert attention.shape[-2] == output.shape[-2]
        elif self.pool_dim in ['time']:
            assert attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * attention, dim=1)               # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, n_basis_kernels, temperature, reduction, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature


        hidden_planes = in_planes // reduction
        if hidden_planes < 4:
            hidden_planes = 4

        padding = int((kernel_size- 1) / 2)
        if pool_dim == 'both':
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)
        else:
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)

            # initialize
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):                                            # x size : [bs, chan, frames, freqs]
        ### Pool dimensions and apply pre-processings
        if self.pool_dim == 'freq':                               #TDY
            x = torch.mean(x, dim=3)                                 # x size : [bs, chan, frames]
        elif self.pool_dim in ['time']: #FDY
            x = torch.mean(x, dim=2)                             # x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':                           #DY
            # x = torch.mean(torch.mean(x, dim=2), dim=1)          #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        ### extract attention weights
        if  self.pool_dim == 'both':
            x = self.relu(self.fc1(x))                                             #x size : [bs, sqzd_chan]
            att = self.fc2(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)            #att size : [bs, n_ker, 1, 1, 1]
        elif self.pool_dim == 'freq':
            x = self.relu(self.bn(self.conv1d1(x)))                                #x size : [bs, sqzd_chan, frames]
            att = self.conv1d2(x).unsqueeze(2).unsqueeze(4)                        #x size : [bs, n_ker, 1, frames, 1]
        else:  #self.pool_dim == 'time', FDY
            x = self.relu(self.bn(self.conv1d1(x)))                                #x size : [bs, sqzd_chan, freqs]
            att = self.conv1d2(x).unsqueeze(2).unsqueeze(3)                        #att size : [bs, n_ker, 1, 1, freqs]

        return F.softmax(att / self.temperature, 1)



########################################################################################################################
#                                                Squeeze and Excitation                                                #
########################################################################################################################
class SELayer(nn.Module):
    def __init__(self, dim, reduction=16, attend_dim="chan"):
        super(SELayer, self).__init__()
        self.attend_dim = attend_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hid_dim = dim // reduction

        if hid_dim < 4:
            hid_dim = 4

        if attend_dim == "chan-freq":
            self.fc = nn.Sequential(nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 1, 3, stride=1, padding=1, bias=False),
                                    nn.Sigmoid())
            self.att_fc1 = nn.Linear(dim, dim)
            self.att_fc2 = nn.Linear(dim, dim)

        else:
            self.fc = nn.Sequential(nn.Linear(dim, hid_dim, bias=False),
                                    # nn.BatchNorm1d(hid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hid_dim, dim, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):   #x size : [bs, chan, frames, freqs]
        b, c, t, f = x.size()
        if self.attend_dim == "chan":
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)

        elif self.attend_dim == "chan_timewise":
            y = torch.mean(x, dim=3).transpose(1, 2)  #x size : [bs, frames, chan]
            y = self.fc(y).transpose(1, 2).view(b, c, t, 1)

        elif self.attend_dim == "freq":
            y = torch.mean(x, dim=(1, 2))
            y = self.fc(y).view(b, 1, 1, f)

        elif self.attend_dim == "freq_timewise":
            y = torch.mean(x, dim=1)                  #x size : [bs, frames, freqs]
            y = self.fc(y).view(b, 1, t, f)

        elif self.attend_dim == "chan-freq":
            y = x.transpose(1, 3)                           # y size :       [bs, freqs, frames, chan]
            y_att = torch.sigmoid(self.att_fc1(y))          # y_att size :   [bs, freqs, frames, chan]
            attention = F.softmax(self.att_fc2(y), dim=2)  # attention size : [bs, freqs, frames, chan]
            y = (y_att * attention).sum(2)                 # y size :       [bs, freqs, chan]
            y = y.view(b, 1, c, f)
            y = self.fc(y).view(b, c, 1, f)

        return x * y.expand_as(x)


class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 DY_layers=[0, 0, 0, 0, 0, 0, 0],
                 n_basis_kernels=4,
                 temperature=31,
                 dy_reduction=4,
                 pool_dim='freq',
                 SE_layers=[0, 0, 0, 0, 0, 0, 0],
                 se_reduction=16,
                 attend_dim='chan',
                 SE2_layers=[0, 0, 0, 0, 0, 0, 0],
                 se2_reduction=16,
                 attend_dim2='chan'):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()
        freq_dims = [128, 64, 32, 16, 8, 4, 2]
        time_dims = [626, 313, 156, 156, 156, 156, 156]

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            # convolution
            if DY_layers[i] == 1:
                if isinstance(n_basis_kernels, int):
                    n_bk = n_basis_kernels
                elif isinstance(n_basis_kernels, list):
                    n_bk = n_basis_kernels[i]
                cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
                                                                   n_basis_kernels=n_bk, temperature=temperature,
                                                                   pool_dim=pool_dim, reduction=dy_reduction,))
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            # normalization
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))
            # non-linearity
            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))
            # squeeze-excitation
            if SE_layers[i] == 1:
                if attend_dim in ["freq", "freq_timewise"]:
                    se_dim = freq_dims[i]
                else:
                    se_dim = out_dim
                cnn.add_module("SElayer{0}".format(i), SELayer(se_dim, reduction=se_reduction, attend_dim=attend_dim))
            if SE2_layers[i] == 1:
                if attend_dim2 in ["freq", "freq_timewise"]:
                    se_dim2 = freq_dims[i]
                else:
                    se_dim2 = out_dim
                cnn.add_module("SElayer2{0}".format(i), SELayer(se_dim2, reduction=se2_reduction,
                                                                attend_dim=attend_dim2))
            # dropout
            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, dropout=conv_dropout, **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(n_RNN_cell * 2, n_class)

        if self.attention:
            self.linear_att = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)  # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)                            # softmax on class dimension


    def forward(self, x):                                                # input size: [bs, freqs, frames]
        #cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1)                           # x size: [bs, chan, frames, freqs]
        x = self.cnn(x)
        x = x.squeeze(-1)                                                # x size: [bs, chan, frames]
        x = x.permute(0, 2, 1)                                           # x size: [bs, frames, chan]

        #rnn
        x = self.rnn(x)                                                  # x size: [bs, frames, 2 * chan]
        x = self.dropout(x)
        strong = self.linear(x)                                          # strong size: [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            attention = self.linear_att(x)                               # attention size: [bs, frames, n_class]
            attention = self.softmax(attention)                          # attention size: [bs, frames, n_class]
            attention = torch.clamp(attention, min=1e-7, max=1)
            weak = (strong * attention).sum(1) / attention.sum(1)        # weak size: [bs, n_class]
        else:
            weak = strong.mean(1)
        return strong.transpose(1, 2), weak