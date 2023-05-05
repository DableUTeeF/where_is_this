from torch import nn
from transformers import AutoModel
import torch
from timm.models.vision_transformer import Block, Attention, Mlp
from timm.models.mobilenetv3 import mobilenetv3_small_075
from timm.models.resnet import Bottleneck

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
        self.backbone = mobilenetv3_small_075(pretrained=True, features_only=True)
        # self.buffer = nn.Conv2d(432, 8192, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(432, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.GELU(),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # with torch.no_grad():
        x = self.backbone(inputs)[4]
        # x = self.encoder(x)
        z = self.decoder(x)
        return z


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.buffer_e = nn.Conv2d(4, 1024, 1)
        self.buffer_d = nn.Conv2d(1024, 4, 1)

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x, grad=True):
        if grad:
            x = self.encode(x)
        else:
            with torch.no_grad():
                x = self.encode(x)
            x = self.sigmoid(self.buffer_e(x))
            y = torch.where(x > 0.5, x, torch.zeros_like(x))
            y = torch.where(x < 0.5, y, torch.ones_like(x))
            x = self.relu(self.buffer_d(y))

        x = self.relu(self.t_conv1(x))
        x = self.sigmoid(self.t_conv2(x))
        return x


if __name__ == '__main__':
    model = SimPlerModel()
    model(torch.zeros((2, 3, 224, 224)))
