import torch
import torch.nn as nn

class LipReadingModel(nn.Module):
    def __init__(self, num_classes):
        super(LipReadingModel, self).__init__()

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # output shape: (B, 128, 1, 1)
        )

        # GRU
        self.rnn_input_size = 128  # matches CNN output channels
        self.hidden_size = 128
        self.num_layers = 1

        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True
        )

        # Output layer: maps GRU output to phoneme classes
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)           # → (B*T, C, H, W)
        feats = self.cnn(x)                  # → (B*T, 128, 1, 1)
        feats = feats.view(B, T, -1)         # → (B, T, 128)
        feats = feats.permute(1, 0, 2)       # → (T, B, 128) for GRU

        rnn_out, _ = self.gru(feats)         # → (T, B, 256)
        out = self.fc(rnn_out)               # → (T, B, num_classes)
        return out
