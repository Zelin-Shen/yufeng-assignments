from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    UNet++ (Nested U-Net) implementation
    主要改进：
    1. 添加了密集的跳跃连接（nested skip pathways）
    2. 支持深度监督（deep supervision）
    3. 重新设计了decoder结构，形成嵌套的U型网络
    """

    def __init__(
        self, in_channels=3, out_channels=1, init_features=32, deep_supervision=False
    ):
        super(UNet, self).__init__()

        self.deep_supervision = deep_supervision
        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv0_1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder0_1 = UNet._block(features * 2, features, name="dec0_1")

        self.upconv0_2 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder0_2 = UNet._block(features * 3, features, name="dec0_2")

        self.upconv0_3 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder0_3 = UNet._block(features * 4, features, name="dec0_3")

        self.upconv0_4 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder0_4 = UNet._block(features * 5, features, name="dec0_4")

        self.upconv1_1 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder1_1 = UNet._block(features * 4, features * 2, name="dec1_1")

        self.upconv1_2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder1_2 = UNet._block(features * 6, features * 2, name="dec1_2")

        self.upconv1_3 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder1_3 = UNet._block(features * 8, features * 2, name="dec1_3")

        self.upconv2_1 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder2_1 = UNet._block(features * 8, features * 4, name="dec2_1")

        self.upconv2_2 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder2_2 = UNet._block(features * 12, features * 4, name="dec2_2")

        self.upconv3_1 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder3_1 = UNet._block(features * 16, features * 8, name="dec3_1")

        if self.deep_supervision:
            self.output1 = nn.Conv2d(features, out_channels, kernel_size=1)
            self.output2 = nn.Conv2d(features, out_channels, kernel_size=1)
            self.output3 = nn.Conv2d(features, out_channels, kernel_size=1)
            self.output4 = nn.Conv2d(features, out_channels, kernel_size=1)
        else:
            self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.encoder1(x)
        x1_0 = self.encoder2(self.pool1(x0_0))
        x2_0 = self.encoder3(self.pool2(x1_0))
        x3_0 = self.encoder4(self.pool3(x2_0))
        x4_0 = self.bottleneck(self.pool4(x3_0))

        x0_1 = self.decoder0_1(torch.cat([x0_0, self.upconv0_1(x1_0)], dim=1))
        x1_1 = self.decoder1_1(torch.cat([x1_0, self.upconv1_1(x2_0)], dim=1))
        x2_1 = self.decoder2_1(torch.cat([x2_0, self.upconv2_1(x3_0)], dim=1))
        x3_1 = self.decoder3_1(torch.cat([x3_0, self.upconv3_1(x4_0)], dim=1))

        x0_2 = self.decoder0_2(torch.cat([x0_0, x0_1, self.upconv0_2(x1_1)], dim=1))
        x1_2 = self.decoder1_2(torch.cat([x1_0, x1_1, self.upconv1_2(x2_1)], dim=1))
        x2_2 = self.decoder2_2(torch.cat([x2_0, x2_1, self.upconv2_2(x3_1)], dim=1))

        x0_3 = self.decoder0_3(
            torch.cat([x0_0, x0_1, x0_2, self.upconv0_3(x1_2)], dim=1)
        )
        x1_3 = self.decoder1_3(
            torch.cat([x1_0, x1_1, x1_2, self.upconv1_3(x2_2)], dim=1)
        )

        x0_4 = self.decoder0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.upconv0_4(x1_3)], dim=1)
        )

        if self.deep_supervision:
            output1 = torch.sigmoid(self.output1(x0_1))
            output2 = torch.sigmoid(self.output2(x0_2))
            output3 = torch.sigmoid(self.output3(x0_3))
            output4 = torch.sigmoid(self.output4(x0_4))
            return [output1, output2, output3, output4]
        else:
            return torch.sigmoid(self.output(x0_4))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )