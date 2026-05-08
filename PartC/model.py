"""CRNN: 7-block CNN backbone (Shi et al. 2015 layout) + 2x BiLSTM + linear head.

Input:  (B, 1, H=96, W) grayscale; H must be 96 for the chosen pooling schedule.
Output: (T, B, vocab_size) log-probs for CTC. T = W // 4.
"""
from __future__ import annotations

import torch
from torch import nn

INPUT_HEIGHT = 96
WIDTH_DOWNSAMPLE = 4  # T = W // 4


class CRNN(nn.Module):
    def __init__(self, vocab_size: int, lstm_hidden: int = 256, lstm_layers: int = 2,
                 lstm_dropout: float = 0.2) -> None:
        super().__init__()
        # Channels: 32, 64, 128, 128, 256, 256, 512.
        # Pooling chosen so that H: 96 -> 48 -> 24 -> 12 -> 6 -> 3 -> 1 -> 1
        # and W: W -> W/2 -> W/2 -> W/4 -> W/4 -> W/4 -> W/4 -> W/4.
        # We squeeze H to 1 by alternating (2x2) and asymmetric (2x1) pooling.
        def conv_block(in_c: int, out_c: int, bn: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.cnn = nn.Sequential(
            conv_block(1, 32, bn=False),                               # 96xW
            nn.MaxPool2d(2, 2),                                        # 48 x W/2
            conv_block(32, 64),                                        # 48 x W/2
            nn.MaxPool2d(2, 2),                                        # 24 x W/4
            conv_block(64, 128),                                       # 24 x W/4
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),           # 12 x W/4
            conv_block(128, 128),                                      # 12 x W/4
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),           # 6  x W/4
            conv_block(128, 256),                                      # 6  x W/4
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),           # 3  x W/4
            conv_block(256, 256),                                      # 3  x W/4
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),           # 1  x W/4
            conv_block(256, 512),                                      # 1  x W/4
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=False,
        )
        self.head = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 96, W)
        feat = self.cnn(x)              # (B, 512, 1, W/4)
        assert feat.shape[2] == 1, f"Expected H=1 after CNN, got {feat.shape}"
        feat = feat.squeeze(2)          # (B, 512, W/4)
        feat = feat.permute(2, 0, 1)    # (T, B, 512)
        out, _ = self.lstm(feat)        # (T, B, 2*hidden)
        logits = self.head(out)         # (T, B, vocab)
        return logits.log_softmax(dim=2)

    @staticmethod
    def output_lengths(input_widths: torch.Tensor) -> torch.Tensor:
        """Map original image widths -> CNN-output time-step counts."""
        return input_widths // WIDTH_DOWNSAMPLE


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
