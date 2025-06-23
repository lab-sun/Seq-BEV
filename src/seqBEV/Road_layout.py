# 通过SENet获得路平面的注意力，再decode包含路平面注意力的特征，用road layout gt经行监督
#from collections import OrderedDict

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

verbose = False

class SE(nn.Module):
    """
    The SENet to calculate the attention
    """
    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(in_chnls//ratio)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x * torch.sigmoid(out)

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class layout_Decoder(nn.Module):
    """
    Decode the latent feature into road layout 
    """
    def __init__(self, num_class=4):   # layout label单类别时，num_class=2; layout label是multi-classes时，num_class=4
        super(layout_Decoder, self).__init__()
        self.num_output_channels = num_class
        #self.num_in_ch = num_chnls
        self.num_ch_dec = np.array([16, 64, 128, 256, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = 512 if i == 4 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["road_layout"] = Conv3x3(
            self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, x):
        """
        Parameters:
        -----------
        x: torch.FloatTensor
            Batch of fused feature tensors
            | Shape: (batch_size, 512, 15, 25)
        Returns
        -----------
        x: torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, 225, 400)
        """

        for i in range(4, -1, -1):
            if verbose: print("in decoder this is {} times to run upsampling".format(i))
            if verbose: print("in decoder, input x: ", x.shape)
            x = self.convs[("upconv", i, 0)](x)
            if verbose: print("in decoder, x after upconv: ", x.shape)
            x = self.convs[("norm", i, 0)](x)
            if verbose: print("in decoder, x after norm: ", x.shape)
            x = self.convs[("relu", i, 0)](x)
            if verbose: print("in decoder, x after relu: ", x.shape)
            x = upsample(x)
            if verbose: print("in decoder, x after upsample: ", x.shape)
            x = self.convs[("upconv", i, 1)](x)
            if verbose: print("in decoder, x after upconv: ", x.shape)
            x = self.convs[("norm", i, 1)](x)
            if verbose: print("in decoder, x after norm: ", x.shape)

        x = self.convs["road_layout"](x)
        # if is_training:
        #     x = self.convs["topview"](x)  # torch.Size([6, 2, 64, 128])
        #     #print("in decoder, x after convs[topview]: ", x.shape)
        # else:
        #     softmax = nn.Softmax2d()
        #     x = softmax(self.convs["topview"](x))
        return x

class road_layout(nn.Module):
    """
    combine both SE and layout_decoder
    """
    def __init__(self, in_chnls, ratio=16):
        super(road_layout, self).__init__()
        self.in_chnls = in_chnls
        self.ratio = ratio
        self.se_attention = SE(self.in_chnls, self.ratio)
        self.layout_decoder = layout_Decoder()

    def forward(self, x):
        x_attention = self.se_attention(x)
        output = self.layout_decoder(x_attention)
        return x_attention, output  # x_attention是包含road plane attention的特征，output是经过扩维的road layout 特征

if __name__ == "__main__":
    test_input = torch.randn([8, 512, 8, 16])

    # 分开检验
    # models = {}
    # print("input shape: ", test_input.shape)
    # models['road_attention'] = SE(in_chnls=512, ratio=16)
    # models['road_layout_decoder'] = layout_Decoder()
    # feature_with_attention = models['road_attention'](test_input)
    # print('feature shape after layout attention: ', feature_with_attention.shape)
    # output = models['road_layout_decoder'](feature_with_attention)
    # print('feature shape after decoder: ', output.shape)

    # 检验road_layout
    model = road_layout(in_chnls=512, ratio=16)
    x_attention, output = model(test_input)
    print('x_attention shape after decoder: ', x_attention.shape)  #([8, 512, 8, 16])
    print('output shape after decoder: ', output.shape)  # ([8, 2, 256, 512])
