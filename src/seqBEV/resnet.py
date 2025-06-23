import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet152(pretrained=True)  # 更换resnet层数 18， 34， 50， 101， 152
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #x = self.model.avgpool(x)

        return x

class base_backbone(nn.Module):
    def __init__(self):
        super(base_backbone, self).__init__()
        self.model = base_resnet()
        self.conv = nn.Conv2d(in_channels=2048, out_channels=512, 
                              kernel_size=3, stride=1, padding=1)  # 使用resnet18， 34时 in_channels为512， 使用resnet50， 101， 152 in_channels为2048
        self.bn = nn.BatchNorm2d(num_features=512)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        input_cur, input_pre, input_prepre = torch.split(x, 3, dim=1)
        output_cur = self.model(input_cur)
        output_pre = self.model(input_pre)
        output_prepre = self.model(input_prepre)
        output = torch.add(output_cur, output_pre)
        output = torch.add(output, output_prepre)  # 将seq的输出结果融合，替代完整结构中的TS fusion module
        output = self.conv(output)
        output = self.bn(output)
        output = self.relu(output)
        return output


if __name__ == '__main__':
    input_temp = torch.randn(1,9,256,512)
    # input_spatial = torch.randn(1,3,450,800)
    # input_cur, input_pre, input_prepre = torch.split(input_temp, 3, dim=1)
    # print("in resnet, input_cur: ", input_cur.shape)
    # model = base_resnet()
    # output_cur = model(input_cur)
    # print("in resnet, output_cur: ", output_cur.shape)

    model = base_backbone()
    output = model(input_temp)
    print("in resnet, output.shape: ", output.shape)