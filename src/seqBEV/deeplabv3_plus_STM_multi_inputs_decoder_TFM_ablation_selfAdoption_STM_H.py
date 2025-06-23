import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
# sys.path.append("..")
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2
# from Road_layout import SE
# from STM_gpu3 import STM
# from TemporalFusion_selfAdoption import TemporalFusion
# sys.path.append("../..")
# from configs.opt import get_args

from src.nets.xception import xception
from src.nets.mobilenetv2 import mobilenetv2
from src.seqBEV.Road_layout import SE
from src.seqBEV.STM_gpu3 import STM
from src.seqBEV.TemporalFusion_selfAdoption import TemporalFusion
from configs.opt import get_args

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
        # 使用temporal——shift
        self.shift_position = 0
        if self.shift:
            for i in self.down_idx:
                print("in deeplabv3_plus_STM_multi_inputs_decoder_TSM.py, adding temporal shifting...")
                self.features[i].conv[self.shift_position] = TemporalFusion(self.features[i].conv[self.shift_position], n_segment=self.opt.num_squence, n_div=self.opt.shift_div)
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

    def forward(self, x, cur_epoch = 1, IF_Seq = False):
        # Sequence Fusion 发生在 [2, 4, 7, 14]
        if IF_Seq:  # seq输入
            # print('!!!!!!!!!!!!!!start sequence fusion encoder!!!!!!!!!!!!!!')
            x = self.features[:2](x)
            x = self.features[2].conv[:self.shift_position](x)
            x = self.features[2].conv[self.shift_position](x, cur_epoch)  # TemporalFusion net
            x = self.features[2].conv[(self.shift_position+1):](x)
            low_level_features = self.features[3:4](x)
            x = self.features[4].conv[:self.shift_position](low_level_features)
            x = self.features[4].conv[self.shift_position](x, cur_epoch)  # TemporalFusion net
            x = self.features[4].conv[(self.shift_position+1):](x)
            x = self.features[5:7](x)
            x = self.features[7].conv[:self.shift_position](x)
            x = self.features[7].conv[self.shift_position](x, cur_epoch)  # TemporalFusion net
            x = self.features[7].conv[(self.shift_position+1):](x)
            x = self.features[8:14](x)
            x = self.features[14].conv[:self.shift_position](x)
            x = self.features[14].conv[self.shift_position](x, cur_epoch)  # TemporalFusion net
            x = self.features[14].conv[(self.shift_position+1):](x)
            x = self.features[15:](x)
        
        else:   # cur_frame 输入
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
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained, IF_shift=False)  # 空间特征提取
            self.backbone_temporal = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained, IF_shift=True)  # 时间特征提取
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
        self.in_chnls_STM_l = 48  # for low level
        self.in_chnls_STM_h = 256  # for high level
        self.STM_l = STM(in_chnls=self.in_chnls_STM_l)
        self.STM_h = STM(in_chnls=self.in_chnls_STM_h)
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        # 空间时间特征融合
        self.fuse_low = nn.Sequential(
            nn.Conv2d(48*2, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fuse_high = nn.Sequential(
            nn.Conv2d(256*2, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
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
        

    def forward(self, input, cur_epoch):
        cur_frame = torch.split(input, self.num_squence, dim=1)[1]
        #print('cur_frame shape: ', cur_frame.shape)

        # ---------------------------------------- #
        # 将输入由(bs, 3*c, w, h)变成(bs*3, c, w, h)
        seq = input.view((-1, 3) + input.size()[-2:])
        #print('seq shape: ', seq.shape)
        # ---------------------------------------- #

        # print("in deeplabv3_plus.py, x.shape: ", x.shape)
        layout_H, layout_W = cur_frame.size(2), cur_frame.size(3)
        H,W = 150, 150
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features_S, x_S = self.backbone(cur_frame, cur_epoch, IF_Seq = False)  # 空间特征提取
        x_S = self.aspp(x_S)  # (bs, 256, 32, 64)
        low_level_features_S = self.shortcut_conv(low_level_features_S)  # (bs, 48, 64, 128)
        low_level_features_T, x_T = self.backbone_temporal(seq, cur_epoch, IF_Seq = True)  #时间特征提取
        x_T = self.aspp(x_T) # (bs*3, 256, 32, 64)
        low_level_features_T = self.shortcut_conv(low_level_features_T)  # (bs*3, 48, 64, 128)
        x_T = torch.unsqueeze(x_T, dim=1).view(-1, 3, x_T.size(1), x_T.size(2), x_T.size(3)).sum(dim=1)  # (bs, 256, 32, 64)
        low_level_features_T = torch.unsqueeze(low_level_features_T, dim=1).view(-1, 3, low_level_features_T.size(1), low_level_features_T.size(2), low_level_features_T.size(3)).sum(dim=1)  # # (bs, 48, 64, 128)
        #print('spatial feature shape: low_level_features_S.shape:{}, high_features_S.shape:{}'.format(low_level_features_S.shape, x_S.shape) )
        #print('temporal feature shape: low_level_features_T.shape:{}, high_features_T.shape:{}'.format(low_level_features_T.shape, x_T.shape) )
        
        #-----------------------------------------#
        #  时空融合
        #-----------------------------------------#
        # 用卷积融合  concat+conv
        low_level_features = torch.cat((low_level_features_S, low_level_features_T), dim=1)
        low_level_features = self.fuse_low(low_level_features)  # ([bs, 48, 64, 128])
        x = torch.cat((x_S, x_T), dim=1)
        x = self.fuse_high(x)  # ([bs, 256, 32, 64])
        # # 用相加融合  add
        # low_level_features = torch.add(low_level_features_S, low_level_features_T)
        # x = torch.add(x_S, x_T)
        #print('low_level_features after fuse: ', low_level_features.shape)
        #print('high_level_features after fuse: ', x.shape)


        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        low_level_features = self.SE_attention_low(low_level_features)  # 包含注意力的低层级特征 
        low_level_features_BEV = self.STM_l(low_level_features)  # 进行视角变换  STM在low level进行视角变换
        
        x = self.SE_attention_high(x)  # 包含注意力的高层级特征
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x_BEV = self.STM_h(x)  # 进行视角变换  STM在high level进行视角变换
        ###low_level_features = torch.add(low_level_features, low_level_features_BEV)  # 将俯视特征和前视特征相加  low level融合
        x = torch.add(x, x_BEV)  # 将俯视特征和前视特征相加  high level融合
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
    model = DeepLab(num_classes=7, backbone="mobilenet", downsample_factor=8, pretrained=False)
    print(model)
    # print(model.state_dict().keys())
    # for k,v in model.state_dict().items():
    #     if 'features.2.' in k:
    #         print("***", k, "***")
    deeplabv3Plus_model_path = '../../pretrained_weight/deeplab_mobilenetv2.pth'
    # 载入预训练权重
    print('Load weights {}'.format(deeplabv3Plus_model_path))
    # 根据预训练权重的Key和模型的Key进行加载
    model_dict = model.state_dict()
    pretrained_dict = torch.load(deeplabv3Plus_model_path, map_location='cpu')
    load_key, no_load_key, temp_dict = [], [], {}
    temporal_layer = [2, 4, 7, 14]
    for k, v in pretrained_dict.items():
        # print('k from pretrained_dict: ', k)
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            if 'backbone' in k:
                #print('###', k, '###')
                InvertedResidual_layer = k.split('.')[2]
                conv_layer = k.split('.')[4]
                # print('###', InvertedResidual_layer, '###', conv_layer, '###')
                if int(InvertedResidual_layer) in temporal_layer and int(conv_layer) == 3:
                    print("!!!!!!!find one temporal_layer!!!!!!!")
                    print("backbone for spatial: ", k)
                    k_temporal = k.split('.')[0] + '_temporal.' + '.'.join(k.split('.')[1:-1]) + '.net.weight'
                    print("backbone for temporal: ", k_temporal)
                else:
                    k_temporal = k.split('.')[0] + '_temporal.' + '.'.join(k.split('.')[1:])
                temp_dict[k] = v
                temp_dict[k_temporal] = v
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    # 显示没有匹配上的Key
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    # 输出结果
    output, layout_output = model(test_input, cur_epoch=1)
    print("output.shape: ", output.shape)
    print("layout_output.shape: ", layout_output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print('the total_parameters for the model is ', total_params)
    #print(model)

