import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.mobilenetv2 import mobilenetv2
from src.nets.xception_SFM import xception
from src.nets import resnet
from src.nets.utils import IntermediateLayerGetter
from src.seqBEV.TemporalFusion import TemporalFusion
from src.seqBEV.Road_layout import SE
from src.seqBEV.STM_gpu1 import STM
from configs.opt import get_args

# from TemporalFusion import TemporalFusion
# from Road_layout import SE
# from STM_gpu1 import STM
# import sys 
# sys.path.append("..")
# from nets.mobilenetv2 import mobilenetv2
# from nets.xception_SFM import xception
# from nets import resnet
# from nets.utils import IntermediateLayerGetter
# sys.path.append("../..")
# from configs.opt import get_args


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True, IF_shift = False):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        self.opt = get_args()
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        self.shift = IF_shift

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

        # ---------------------------------------------------- #
        # 使用seqence fusion 
        # mobilenet 的 Block 先是通过一个1*1 conv的Expansion layer，然后再接一个3*3 conv的Depthwise layer，最后再加一个1*1 conv的projection layer
        if self.shift:
            for i in self.down_idx:
                print("in deeplabv3_plus_STM_multi_inputs_decoder_TSM.py, adding temporal shifting...")
                #self.features[i].conv[0] = TemporalFusion(self.features[i].conv[0], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)  # 在Expansion layer前做时间融合
                self.features[i].conv[3] = TemporalFusion(self.features[i].conv[3], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)  # 在Depthwise layer前做时间融合  需要修改预训练参数载入
                #self.features[i].conv[6] = TemporalFusion(self.features[i].conv[6], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)   # 在projection layer前做时间融合  需要修改预训练参数载入
        # ---------------------------------------------------- #

        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


class ResNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True, IF_shift = False):
        super(ResNet, self).__init__()
        self.backbone = ['resnet50', 'resnet101', 'resnet152']
        self.backbone_name = 'resnet50'
        if downsample_factor==8:
            self.replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            self.replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]

        self.model = resnet.__dict__[self.backbone_name](
            pretrained=pretrained,
            replace_stride_with_dilation=self.replace_stride_with_dilation
        )

        self.return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        self.model = IntermediateLayerGetter(self.model, return_layers=self.return_layers)

        self.opt = get_args()
        self.shift = IF_shift
        self.fusion = ['layer1', 'layer2', 'layer3', 'layer4']

        # ---------------------------------------------------- #
        # 使用seqence fusion 
        # mobilenet 的 Block 先是通过一个1*1 conv的Expansion layer，然后再接一个3*3 conv的Depthwise layer，最后再加一个1*1 conv的projection layer
        if self.shift:
            self.model.layer1[0].conv1 = TemporalFusion(self.model.layer1[0].conv1, n_segment=self.opt.num_squence, n_div=self.opt.shift_div)
            self.model.layer2[0].conv1 = TemporalFusion(self.model.layer2[0].conv1, n_segment=self.opt.num_squence, n_div=self.opt.shift_div)
            self.model.layer3[0].conv1 = TemporalFusion(self.model.layer3[0].conv1, n_segment=self.opt.num_squence, n_div=self.opt.shift_div)
            self.model.layer4[0].conv1 = TemporalFusion(self.model.layer4[0].conv1, n_segment=self.opt.num_squence, n_div=self.opt.shift_div)
            
            # for i in self.fusion:
            #     print("in deeplabv3_plus_STM_multi_inputs_decoder_TSM.py, adding temporal shifting...")
            #     #self.features[i].conv[0] = TemporalFusion(self.features[i].conv[0], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)  # 在Expansion layer前做时间融合
            #     self.features[i].conv[3] = TemporalFusion(self.features[i].conv[3], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)  # 在Depthwise layer前做时间融合  需要修改预训练参数载入
            #     #self.features[i].conv[6] = TemporalFusion(self.features[i].conv[6], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)   # 在projection layer前做时间融合  需要修改预训练参数载入
        # ---------------------------------------------------- #


    def forward(self, x):
        out = self.model(x)
        low_level_features = out['low_level']
        x = out['out']
        return low_level_features, x


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, num_classes_layout=4, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [64,128,256]
            #   主干部分    [32,64,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained, IF_SHIFT=False)  # 空间特征提取
            self.backbone_temporal = xception(downsample_factor=downsample_factor, pretrained=pretrained, IF_SHIFT=True)  # 时间特征提取
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":  # low-level 4, high-level 17
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [64,128,24]
            #   主干部分    [32,64,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained, IF_shift=False)  # 空间特征提取
            self.backbone_temporal = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained, IF_shift=True)  # 时间特征提取
            in_channels = 320
            low_level_channels = 24
        elif backbone=="resnet":  # low-level 1, high-level 4
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [64,128,256]
            #   主干部分    [32,64,2048]
            #----------------------------------#
            self.backbone = ResNet(downsample_factor=downsample_factor, pretrained=pretrained)  # 空间特征提取
            self.backbone_temporal = ResNet(downsample_factor=downsample_factor, pretrained=pretrained, IF_shift=True)  # 时间特征提取
            in_channels = 2048
            low_level_channels = 256
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception, resnet.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        # ---------------------------------------------------------------------------- #
        # 添加SE-Net提取注意力
        self.num_squence = 3
        self.in_chnls_lower = 48
        self.in_chnls_higher = 256
        self.ratio = 16
        self.SE_attention_low = SE(self.in_chnls_lower, self.ratio)
        self.SE_attention_high = SE(self.in_chnls_higher, self.ratio)

        # roadlayout 上采样
        self.layout_upsampleing = nn.Sequential(
            nn.Conv2d(48, 24, 3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.layout_cls_conv = nn.Conv2d(24, num_classes_layout, 1, stride=1)
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        # 添加STM空间变换
        self.in_chnls_STM = 48
        self.STM = STM(in_chnls=self.in_chnls_STM)
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        # # 空间时间特征融合
        # self.fuse_low = nn.Sequential(
        #     nn.Conv2d(48*2, 48, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5)
        # )
        # self.fuse_high = nn.Sequential(
        #     nn.Conv2d(256*2, 256, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5)
        # )
        # ---------------------------------------------------------------------------- #

        
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=[1,2], padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, input):
        cur_frame = torch.split(input, self.num_squence, dim=1)[1]
        # ---------------------------------------- #
        # 将输入由(bs, 3*c, w, h)变成(bs*3, c, w, h)
        seq = input.view((-1, 3) + input.size()[-2:])
        #print('seq shape: ', seq.shape)
        # ---------------------------------------- #


        layout_H, layout_W = cur_frame.size(2), cur_frame.size(3)
        H,W = 150, 150
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features_S, x_S = self.backbone(cur_frame)  # 空间特征提取
        x_S = self.aspp(x_S)  # (bs, 256, 32, 64)
        low_level_features_S = self.shortcut_conv(low_level_features_S)  # (bs, 48, 64, 128)
        low_level_features_T, x_T = self.backbone_temporal(seq)  #时间特征提取
        x_T = self.aspp(x_T) # (bs*3, 256, 32, 64)
        low_level_features_T = self.shortcut_conv(low_level_features_T)  # (bs*3, 48, 64, 128)
        x_T = torch.unsqueeze(x_T, dim=1).view(-1, 3, x_T.size(1), x_T.size(2), x_T.size(3)).sum(dim=1)  # (bs, 256, 32, 64)
        low_level_features_T = torch.unsqueeze(low_level_features_T, dim=1).view(-1, 3, low_level_features_T.size(1), low_level_features_T.size(2), low_level_features_T.size(3)).sum(dim=1)  # # (bs, 48, 64, 128)

        # 时空融合
        # # 用卷积融合
        # low_level_features = torch.cat((low_level_features_S, low_level_features_T), dim=1)
        # low_level_features = self.fuse_low(low_level_features)  # ([bs, 48, 64, 128])
        # x = torch.cat((x_S, x_T), dim=1)
        # x = self.fuse_high(x)  # ([bs, 256, 32, 64])
        # 用相加融合
        low_level_features = torch.add(low_level_features_S, low_level_features_T)
        x = torch.add(x_S, x_T)
        #print('low_level_features after fuse: ', low_level_features.shape)
        #print('high_level_features after fuse: ', x.shape)

        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        low_level_features = self.SE_attention_low(low_level_features)  # 包含注意力的低层级特征 
        low_level_features_BEV = self.STM(low_level_features)  # 进行视角变换
        
        x = self.SE_attention_high(x)  # 包含注意力的高层级特征
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        low_level_features = torch.add(low_level_features, low_level_features_BEV)  # 将俯视特征和前视特征相加
        # print('in the deeplabv3_plus, feature before cat_conv: ', torch.cat((x, low_level_features), dim=1).shape)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))  #  ([6, 304, 64, 128]) -> ([6, 256, 64, 64])
        # print('in the deeplabv3_plus, feature after cat_conv: ', x.shape)
        x = self.cls_conv(x)  # ([6, 256, 64, 64]) -> ([6, num_class, 64, 64])
        # print('in the deeplabv3_plus, feature before interpolate: ', x.shape)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)  # ([6, num_class, 64, 64]) -> ([6, num_class, 150, 150])
        # print('in the deeplabv3_plus, feature after interpolate: ', x.shape)

        # ---------------------------------------------------------------------------- #
        # roadlayout decoder
        layout_x = self.layout_upsampleing(low_level_features)
        layout_x = self.layout_cls_conv(layout_x)
        layout_x = F.interpolate(layout_x, size=(layout_H, layout_W), mode='bilinear', align_corners=True)
        # ---------------------------------------------------------------------------- #

        return x, layout_x


if __name__ == '__main__':
    import numpy as np
    test_input = torch.rand(6, 3*3, 256, 512)
    backbone_net = "resnet"
    model = DeepLab(num_classes=7, backbone=backbone_net, downsample_factor=8, pretrained=True)  # mobilenet  xception  resnet

    
    output, layout_output = model(test_input)
    print("output.shape: ", output.shape)
    print("layout_output.shape: ", layout_output.shape)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print('the total_parameters for the model is ', total_params)

