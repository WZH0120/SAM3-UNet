import os
import sam3
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam3 import build_sam3_image_model
from timm.models.layers import trunc_normal_   


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)


class SAM3UNet(nn.Module):
    def __init__(self) -> None:
        super(SAM3UNet, self).__init__()
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.sam3 = build_sam3_image_model(bpe_path=bpe_path, load_from_HF=False, device="cuda"
                                    #    , checkpoint_path="YOUR SAM3.PT CHECKPOINT PATH"
                                    ).backbone.vision_backbone.trunk
        for param in self.sam3.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.sam3.blocks:
            blocks.append(
                Adapter(block)
            )
        self.sam3.blocks = nn.Sequential(
            *blocks
        )
        self.reduce1 = nn.Conv2d(1024, 128, 1)
        self.reduce2 = nn.Conv2d(1024, 128, 1)
        self.reduce3 = nn.Conv2d(1024, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 128, 1)
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.sam3(x)[-1]
        x1 = F.interpolate(self.reduce1(x), size=(H//4, W//4), mode='bilinear')
        x2 = F.interpolate(self.reduce2(x), size=(H//8, W//8), mode='bilinear')
        x3 = F.interpolate(self.reduce3(x), size=(H//16, W//16), mode='bilinear')
        x4 = F.interpolate(self.reduce4(x), size=(H//32, W//32), mode='bilinear')
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        out = F.interpolate(self.head(x), size=(H, W), mode='bilinear')
        return out

    
if __name__ == "__main__":
    model = SAM3UNet().cuda().eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 1008, 1008).cuda()
        out = model(x)

        print(out.shape)
