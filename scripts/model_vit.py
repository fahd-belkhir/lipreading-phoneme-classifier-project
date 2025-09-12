import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, B, D)
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)

class LipReadingViTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Frame-level CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )

        self.embedding_dim = 128
        self.pos_encoder = PositionalEncoding(self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)              # (B*T, 128, 1, 1)
        features = features.view(B, T, -1)  # (B, T, 128)
        features = features.permute(1, 0, 2)  # (T, B, 128)

        features = self.pos_encoder(features)
        encoded = self.transformer(features)  # (T, B, 128)
        out = self.classifier(encoded)       # (T, B, num_classes)
        return out
