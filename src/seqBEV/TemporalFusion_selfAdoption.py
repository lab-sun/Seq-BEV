import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalFusion(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalFusion, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x, cur_epoch):
        x = self.shift(x, self.n_segment, fold_div_set=self.fold_div, inplace=self.inplace, cur_epoch=cur_epoch)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div_set=3, inplace=False, cur_epoch=1):
        nt, c, h, w = x.size()
        #print('in TemporalFusion.py number of channel is ', c)
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        # fold = c // fold_div  # 非自适应
        # ------------------------------------- #
        # 自适应调整分组
        total_epoch = 50
        #div = c//fold_div_set
        if c//fold_div_set == 0:
            fold_div_set = c
        fold_div = fold_div_set*(min(c//fold_div_set, max(math.floor(total_epoch/cur_epoch * 0.5), 1)))
        #print('in TemporalFusion_selfAdoption, c and fold_div_set: ', c, fold_div_set)
        #print('in TemporalFusion_selfAdoption, c//fold_div_set: ', c//fold_div_set)
        #print('in TemporalFusion_selfAdoption, fold_div: ', fold_div)
        fold = c // fold_div
        # ------------------------------------- #
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left +1
            out[:, -1, :fold] = x[:, 0, :fold]
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right +2
            out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]
            # out[:, :-1, 2 * fold: 3 * fold] = x[:, 1:, 2 * fold: 3 * fold]  # shift left -2
            # out[:, -1, 2 * fold: 3 * fold] = x[:, 0, 2 * fold: 3 * fold]
            # out[:, 1:, 3 * fold: 4 * fold] = x[:, :-1, 3 * fold: 4 * fold]  # shift right -1
            # out[:, 0, 3 * fold: 4 * fold] = x[:, -1, 3 * fold: 4 * fold]
            # out[:, :, 4 * fold:] = x[:, :, 4 * fold:]  # not shift
            out[:, :, 2 * fold:(fold_div-2) * fold] = x[:, :, 2 * fold:(fold_div-2) * fold]  # not shift
            out[:, :-1, (fold_div-2) * fold: (fold_div-1) * fold] = x[:, 1:, (fold_div-2) * fold: (fold_div-1) * fold]  # shift left -2
            out[:, -1, (fold_div-2) * fold: (fold_div-1) * fold] = x[:, 0, (fold_div-2) * fold: (fold_div-1) * fold]
            out[:, 1:, (fold_div-1) * fold: ] = x[:, :-1, (fold_div-1) * fold: ]  # shift right -1
            out[:, 0, (fold_div-1) * fold: ] = x[:, -1, (fold_div-1) * fold: ]

        return out.view(nt, c, h, w)