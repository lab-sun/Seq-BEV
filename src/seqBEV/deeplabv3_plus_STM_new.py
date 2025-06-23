# STM输出结果与low feature 相加

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
# sys.path.append("..")
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2
# from Road_layout import SE
# from STM_new import STM

from src.nets.xception import xception
from src.nets.mobilenetv2 import mobilenetv2
from src.seqBEV.Road_layout import SE
from src.seqBEV.STM_new import STM

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

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
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

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

    def forward(self, x):
        # print("in deeplabv3_plus.py, x.shape: ", x.shape)
        layout_H, layout_W = x.size(2), x.size(3)
        H,W = 150, 150
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        # print('in the deeplabv3_plus, high_level_features before aspp: ', x.shape)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        low_level_features = self.SE_attention_low(low_level_features)  # 包含注意力的低层级特征
        low_level_features_BEV = self.STM(low_level_features)  # 进行视角变换
        x = self.SE_attention_high(x)  # 包含注意力的高层级特征
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        # print('in the deeplabv3_plus, low_level_features: ', low_level_features.shape)
        # print('in the deeplabv3_plus, high_level_features after aspp: ', x.shape)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        low_level_features = torch.add(low_level_features, low_level_features_BEV)  # 将俯视特征和前视特征相加
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        # print('in the deeplabv3_plus, feature after cat_conv: ', x.shape)
        x = self.cls_conv(x)
        # print('in the deeplabv3_plus, feature after cls_conv: ', x.shape)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # ---------------------------------------------------------------------------- #
        # roadlayout decoder
        layout_x = self.layout_upsampleing(low_level_features)
        layout_x = self.layout_cls_conv(layout_x)
        layout_x = F.interpolate(layout_x, size=(layout_H, layout_W), mode='bilinear', align_corners=True)
        # ---------------------------------------------------------------------------- #

        return x, layout_x


if __name__ == '__main__':
    test_input = torch.rand(6, 3, 256, 512)
    model = DeepLab(num_classes=7, backbone="mobilenet", downsample_factor=8, pretrained=False)
    output, layout_output = model(test_input)
    print("output.shape: ", output.shape)
    print("layout_output.shape: ", layout_output.shape)
    #print(model)

