import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print("in STM, num of GPUs: ", torch.cuda.device_count())

from torch import nn
import torchvision
import torch.nn.functional as F
import kornia.geometry as g



verbose = False

class STM(nn.Module):
    def __init__(self, in_chnls):
        super(STM, self).__init__()
        self.input_channels = in_chnls

        # Spatial trainformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.input_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            #nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 60, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(60),
            nn.ReLU(True)
        )

        # Regressor for the 3*3 Homography matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(60*16*32, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 3*3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], 
                                       dtype=torch.float))
    
    # Spatial transformer network forward function
    def stm(self, x):
        if verbose: print("in stm, input shape is ", x.shape)
        xs = self.localization(x)
        if verbose: print("in stm, after localization, x shape is ", xs.shape)
        self.num_feat = xs.shape[1] * xs.shape[2] * xs.shape[3]
        xs = xs.view(-1, self.num_feat)
        if verbose: print("in stm, x shape after view ", xs.shape)
        theta = self.fc_loc(xs)
        div_factor = theta[:,-1]  # 除以最后的元素 归一化
        div_factor = torch.unsqueeze(div_factor, 1)
        theta = torch.div(theta, div_factor)
        if verbose: print("in stm, theta shape after fc_loc ", theta.shape)
        theta = theta.view(-1, 3, 3)
        if verbose: print("in stm, theta shape after view ", theta.shape)

        # grid = F.affine_grid(theta, x.size())
        # print("in stm, grid shape is ", grid.shape)
        # x = F.grid_sample(x, grid)
        # print("in stm, output shape from the whole stm ", x.shape)

        x_size = (x.shape[2],x.shape[3])
        x = g.warp_perspective(x, theta, dsize=x_size)

        return x

    def forward(self, x):
        # transform the input
        x = self.stm(x)
        if verbose: print("in forward, input shape is ", x.shape)

        return x


if __name__ == "__main__":
    test_input = torch.randn([8, 48, 64, 128])
    model = STM(in_chnls=48)
    output = model(test_input)
    print("in STM, the output shape from STM: ", output.shape)
