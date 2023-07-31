from torch import nn
import torch
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.GELU(),
    )

class UNet(nn.Module):

    def __init__(self, 
                 n_pts = 12, 
                 n_classes = 2):

        super().__init__()
        
        base_model = models.resnet18()
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3])
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer2 = base_layers[5]
        self.layer3 = base_layers[6]
        self.layer4 = base_layers[7]

        self.decoders = nn.ModuleDict()

        for i in range(n_classes):

            self.decoders[f"decoder_{i}"] = nn.ModuleDict(
                {
                    "layer0_1x1": convrelu(64, 64, 1, 0),
                    "layer1_1x1": convrelu(64, 64, 1, 0),
                    "layer2_1x1": convrelu(128, 128, 1, 0),
                    "layer3_1x1": convrelu(256, 256, 1, 0),
                    "layer4_1x1": convrelu(512, 512, 1, 0),

                    "conv_up3": convrelu(256 + 512, 512, 3, 1),
                    "conv_up2": convrelu(128 + 512, 256, 3, 1),
                    "conv_up1": convrelu(64 + 256, 256, 3, 1),
                    "conv_up0": convrelu(64 + 256, 128, 3, 1),

                    "conv_1": convrelu(128, 64, 1, 0),
                    "conv_2": convrelu(64, n_pts, 1, 0),
                }
            )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        print("UNet initialized")
        print("Total number of parameters ", sum(p.numel() for p in self.parameters()))

    def forward(self, input):

        e_layer0 = self.layer0(input)
        e_layer1 = self.layer1(e_layer0)
        e_layer2 = self.layer2(e_layer1)
        e_layer3 = self.layer3(e_layer2)
        e_layer4 = self.layer4(e_layer3)

        e_layers = [e_layer0, e_layer1, e_layer2, e_layer3, e_layer4]

        for decoder in self.decoders:

            decoder = self.decoders[decoder]

            x = decoder["layer4_1x1"](e_layer4)
            x = self.upsample(x)

            for layer_index in range(3, -1, -1):

                y = e_layers[layer_index]
                y = decoder[f"layer{layer_index}_1x1"](y)
                x = torch.cat([x, y], dim=1)
                x = decoder[f"conv_up{layer_index}"](x)
                x = self.upsample(x)
            
            x = decoder["conv_1"](x)
            x = decoder["conv_2"](x)

        return x


if __name__ == "__main__":

    unet = UNet(12, 2)

    x = torch.Tensor(1, 3, 256, 256)

    out = unet(x)
