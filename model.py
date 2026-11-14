import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.decoder4 = ConvBlock(512 + 256, 256)
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 96)
        self.decoder1 = ConvBlock(96 + 64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.decoder4(torch.cat([d4, x3], dim=1))

        d3 = F.interpolate(d4, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([d3, x2], dim=1))

        d2 = F.interpolate(d3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x1], dim=1))

        d1 = F.interpolate(d2, size=x0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x0], dim=1))

        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        return self.final_conv(out)


if __name__ == "__main__":
    model = ResNet34UNet(pretrained=True)
    x = torch.randn(1, 3, 320, 480)
    with torch.no_grad():
        y = model(x)
    print("模型创建成功")
    print("输入尺寸:", tuple(x.shape))
    print("输出尺寸:", tuple(y.shape))
    print("参数总量:", sum(p.numel() for p in model.parameters()))
