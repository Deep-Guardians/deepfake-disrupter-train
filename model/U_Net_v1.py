import torch
import torchvision.transforms.functional
from torch import nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Second 3x3 convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, out_channels: int):

        super().__init__()
        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from 64.
        down_conv_sizes = [(3, 64), (64, 128), (128, 256), (256, 512)]
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in down_conv_sizes])
        
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(512, 1024)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        upsample_sizes = [(1024, 512), (512, 256), (256, 128), (128, 64)]
        
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in upsample_sizes])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        up_conv_sizes = [(1024, 512), (512, 256), (256, 128), (128, 64)]
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in up_conv_sizes])
        
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        
        # Final 1x1 convolution layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two 3x3 convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two 3x3 convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two 3x3 convolutional layers
            x = self.up_conv[i](x)

        # Final 1x1 convolution layer
        out = self.final_conv(x)

        return out


# # First, the necessary modules are imported from the torch and torchvision packages, including the nn module for building neural networks and the pre-trained models provided in torchvision.models.
# # The relu function is also imported from torch.nn.functional.
# import torch
# import torch.nn as nn
# from torchvision import models
# from torch.nn.functional import relu
#
#
# # Then, a custom class UNet is defined as a subclass of nn.Module.
# # The __init__ method initializes the architecture of the U-Net by defining the layers for both the encoder and decoder parts of the network.
# # The argument n_class specifies the number of classes for the segmentation task.
# class UNet(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()
#
#         # Encoder
#         # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
#         # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
#         # -------
#         # input: 572x572x3
#         self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
#         self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64
#
#         # input: 284x284x64
#         self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
#         self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128
#
#         # input: 140x140x128
#         self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
#         self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256
#
#         # input: 68x68x256
#         self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
#         self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512
#
#         # input: 32x32x512
#         self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
#         self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
#
#
#         # Decoder
#         # In the decoder, transpose convolutional layers with the ConvTranspose2d function are used to upsample the feature maps to the original size of the input image.
#         # Each block in the decoder consists of an upsampling layer, a concatenation with the corresponding encoder feature map, and two convolutional layers.
#         # -------
#         self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
#         self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#
#         self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#
#         self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#
#         self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#
#         # Output layer
#         self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
#
#     def forward(self, x):
#         # Encoder
#         xe11 = relu(self.e11(x))
#         xe12 = relu(self.e12(xe11))
#         xp1 = self.pool1(xe12)
#
#         xe21 = relu(self.e21(xp1))
#         xe22 = relu(self.e22(xe21))
#         xp2 = self.pool2(xe22)
#
#         xe31 = relu(self.e31(xp2))
#         xe32 = relu(self.e32(xe31))
#         xp3 = self.pool3(xe32)
#
#         xe41 = relu(self.e41(xp3))
#         xe42 = relu(self.e42(xe41))
#         xp4 = self.pool4(xe42)
#
#         xe51 = relu(self.e51(xp4))
#         xe52 = relu(self.e52(xe51))
#
#         # Decoder
#         xu1 = self.upconv1(xe52)
#         xu11 = torch.cat([xu1, xe42], dim=1)
#         xd11 = relu(self.d11(xu11))
#         xd12 = relu(self.d12(xd11))
#
#         xu2 = self.upconv2(xd12)
#         xu22 = torch.cat([xu2, xe32], dim=1)
#         xd21 = relu(self.d21(xu22))
#         xd22 = relu(self.d22(xd21))
#
#         xu3 = self.upconv3(xd22)
#         xu33 = torch.cat([xu3, xe22], dim=1)
#         xd31 = relu(self.d31(xu33))
#         xd32 = relu(self.d32(xd31))
#
#         xu4 = self.upconv4(xd32)
#         xu44 = torch.cat([xu4, xe12], dim=1)
#         xd41 = relu(self.d41(xu44))
#         xd42 = relu(self.d42(xd41))
#
#         # Output layer
#         out = self.outconv(xd42)
#
#         return out
#

# In[107]:
