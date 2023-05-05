from torch import nn
from transformers import AutoModel
import torch
from timm.models.vision_transformer import Block, Attention, Mlp
from timm.models.mobilenetv3 import mobilenetv3_small_075

class Where2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_buffer = nn.Linear(768, 12800)
        self.act = nn.GELU()
        self.decoder = nn.Sequential(
            nn.LayerNorm(256),
            Attention(256, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0),
            nn.LayerNorm(256),
            Mlp(256, out_features=768),
            nn.GELU(),
            Block(768, 8),
        )

    def forward(self, input):
        x = self.encoder_buffer(input['pooler_output'])
        y = torch.where(x > 0, x, torch.zeros_like(x))
        y = torch.where(x < 0, y, torch.ones_like(x))
        z = self.act(y.view(-1, 50, 256))
        z = self.decoder(z)
        return z


class WhereIsCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('openai/clip-vit-base-patch32').vision_model
        self.buffer = nn.Linear(768, 8192)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8192, 1024, 2, stride=2),  # 2
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 4
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),  # 8
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 2, stride=2),  # 32
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 2, stride=2),  # 64
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 128
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 7),  # 122
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 7),  # 116
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 5),  # 112
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 224
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),  # 224
        )

    def forward(self, input):
        with torch.no_grad():
            x = self.encoder(input)  # batch, embed_dim
        x = self.buffer(x['pooler_output'])
        y = torch.where(x > 0, x, torch.zeros_like(x))
        y = torch.where(x < 0, y, torch.ones_like(x))
        z = self.decoder(y.unsqueeze(2).unsqueeze(2))
        return z


class SimPlerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenetv3_small_075(True)
        self.buffer = nn.Conv2d(432, 8192, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8192, 1024, 2, stride=2),  # 7
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 14
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),  # 56
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 112
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 2, stride=2),  # 224
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 3, 1),  # 224
        )

    def forward(self, inputs):
        with torch.no_grad():
            x = self.backbone.conv_stem(inputs)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            x = self.backbone.blocks(x)
        y = self.buffer(x)
        z = self.decoder(y)
        return z


if __name__ == '__main__':
    model = SimPlerModel()
    model(torch.zeros((2, 3, 224, 224)))
