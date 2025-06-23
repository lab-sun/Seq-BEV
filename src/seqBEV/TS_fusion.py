import torch
from torch import nn
import torchvision

class TS_Fusion(nn.Module):
    """
    Fuse the temporal feature and spatial feature together
    spt_channels: int
        the number of channels of the spatial feature
    tem_channels: int
        the number of channels of the temporal feature
    spt_feature: tensor
        the spatial feature from encoder
    tem_feature: tensor
        the temporal feature from encoder
    """
    def __init__ (self, spt_channels, tem_channels):
        super(TS_Fusion, self).__init__()
        self.spt_channels = spt_channels
        self.tem_channels = tem_channels
        #self.spt_feature = spt_feature
        #self.tem_feature = tem_feature
        self.fusion_channels = self.spt_channels + self.tem_channels
        self.conv1 = nn.Conv2d(in_channels=self.fusion_channels, out_channels=1024, 
                              kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, 
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)

    def forward(self, spt_input, tem_input):
        x = torch.cat([spt_input, tem_input], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    spt_input = torch.randn(8,2048,15,25)
    tem_input = torch.randn(8,2048,15,25)
    model = TS_Fusion(spt_channels=2048, tem_channels=2048)
    output = model(spt_input, tem_input)
    print(output.shape)  # torch.Size([8, 512, 15, 25])
