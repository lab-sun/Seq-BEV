from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

verbose = False

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

def ratio_convert(in_channels, out_channels):
    """change the ratio between the img_width and img_height"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,4), stride=(1,2), padding=(0,1))

class Decoder(nn.Module):
    def __init__(self, in_chnls, num_class):
        super(Decoder, self).__init__()
        self.num_output_channels = num_class
        #self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = in_chnls if i==3 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, road_layout_x=None, objects_x=None):
        if road_layout_x == None and road_layout_x == None:   # 便于删减模块训练
            print("in decoder, both road_layout_input and objects_input are None")
            raise AssertionError
        elif road_layout_x != None and objects_x != None:
            x = torch.cat((road_layout_x, objects_x), 1)
        else:
            x = road_layout_x if road_layout_x != None else objects_x

        x = nn.functional.interpolate(x, (25,25), mode='bilinear', align_corners=True)

        for i in range(3, -1, -1):
            if verbose: print("in decoder this is {} times to run upsampling".format(i))
            if verbose: print("in decoder, inptu x: ", x.shape)
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

        x = self.convs["topview"](x)
        x = nn.functional.interpolate(x, (150,150), mode='bilinear', align_corners=True)
        if verbose: print("in decoder, x after convs[topview]: ", x.shape)

        return x

if __name__ == "__main__":
    test_layout = torch.randn(8, 512, 8, 16)
    test_object = torch.randn(8, 50, 8, 16)
    in_chnls = 512+50
    #in_chnls = 50
    model = Decoder(in_chnls=in_chnls, num_class=6)
    output = model(test_layout, test_object)
    print("output shape is ", output.shape)

