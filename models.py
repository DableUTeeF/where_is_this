from torch import nn
from transformers import AutoModel
import torch


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


if __name__ == '__main__':
    model = WhereIsCLIP()
    model(torch.zeros((2, 3, 224, 224)))
