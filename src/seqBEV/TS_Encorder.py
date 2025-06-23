import torch
from torch import nn
import torchvision
from torch.nn.init import normal_, constant_

#from ops.basic_ops import ConsensusModule
#from ops.transforms import *

import sys
sys.path.append('/workspace/')
from src.seqBEV.temporal_shift_old import make_temporal_shift
from src.seqBEV.non_local import make_non_local

verbose = False

class TS_Encoder(nn.Module):
    """
    Encode the Seq_img in temporal branch (num of seq is setted in opt.py)
    Encode the current img in spatial branch

    Attributes
    ----------
    if_seq: bool
        flag for seq_img input
    base_model: str
        basic model used in the encoder
    num_segments: int
        the number of images in the sequence
    batch_size: int
        the frame number in a mini batch
    shift_div: int
        the number of the division in temporal dimension 
    shift_place: str
        to choose where to place the TSM, option: block & blockres
    pretrain: str
        the pretrain module
    partial_bn: bool
        partial batch normalization
    temporal_pool: bool
        wether to pool the temporal and spatial branch output
    non_local: bool
        whether or not to use the non-local conv

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """
    def __init__ (self, if_seq=True, base_model='resnet101', num_segments=3, batch_size=8,
                  shift_div=8, shift_place='block', pretrain='imagenet', 
                  partial_bn=True, temporal_pool=False, non_local=False):
        super(TS_Encoder, self).__init__()
        self.if_seq = if_seq
        self.base_model_name = base_model
        self.batch_size = batch_size
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.pretrain = pretrain
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        self.create_base_model(base_model)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def create_base_model(self, base_model):
        print("=>>> base model: {}".format(base_model))

        if 'resnet' in base_model:
            base_resnet = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.if_seq:
                if verbose: print("adding the temporal shift module....")
                make_temporal_shift(base_resnet, self.num_segments, n_div=self.shift_div, 
                                    place=self.shift_place, temporal_pool=self.temporal_pool)
            
            if self.non_local:
                if verbose: print("Using non-local conv...")
                make_non_local(base_resnet, self.num_segments)

            self.base_model = torch.nn.Sequential( *( list(base_resnet.children())[:-2] ))
        
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TS_Encoder, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            if verbose: print("In TS_Encoder, train, Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
    
    def temporal_fusion(self, x):
        bs = int(x.shape[0] / self.num_segments)
        #print("in TS_encoder.py, bs: ", bs)
        temp_x = x.view((bs, -1) + x.size()[-3:])
        #if verbose: print("in TS_Encoder, temporal_fuxion, temp_x.shape: ", temp_x.shape)
        fused_temp = temp_x.sum(dim=1)  # fusion though add
        #if verbose: print("in TS_Encoder, temporal_fuxion, fused_temp.shape: ", fused_temp.shape)
        return fused_temp
        

    def forward(self, input):
        sample_len = 3  # 我理解是输入图像的通道数
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        # if verbose: print("in TS_Encoder, forward: after base_mosel, base_out.shape: ", base_out.shape)
        if self.if_seq == True:
            base_out = self.temporal_fusion(base_out)
        return base_out


if __name__ == '__main__':
    input_temp = torch.randn(1,9,450,800)
    input_spatial = torch.randn(1,3,450,800)
    temp_model_test = TS_Encoder(if_seq=True, base_model='resnet101', 
                            num_segments=3, batch_size=1, shift_div=8, shift_place='block', 
                            pretrain='imagenet', temporal_pool=False, non_local=False)
    spatial_model_test = TS_Encoder(if_seq=False, base_model='resnet101', 
                            num_segments=3, batch_size=1, shift_div=8, shift_place='block', 
                            pretrain='imagenet', temporal_pool=False, non_local=False)

    temp_model_test.train()
    spatial_model_test.train()
    params = temp_model_test.get_optim_policies()  # 按照不同网络层，返回学习率，现在用的模型没有'custom_ops'，'lr5_weight'，'lr10_bias'
    #print(params)
    output_temp = temp_model_test(input_temp)
    output_spatial = spatial_model_test(input_spatial)
    print("to test TS_Encoder, the temporal output is ", output_temp.shape)
    print("to test TS_Encoder, the spatial output is ", output_spatial.shape)
