"""CRNN models for Part C. Two backbones:

  - "vgg" : 7-block conv stack from Shi et al. 2015. ~5.5M params, fast.
  - "resnet18" : ImageNet-pretrained ResNet-18 stem + custom OCR head.
                 Uses pretrained low-level features; W is downsampled 4x.

Both produce (T, B, vocab_size) log-probs for CTC. T = W // 4.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

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
        return input_widths // WIDTH_DOWNSAMPLE


class CRNNResNet18(nn.Module):
    """CRNN with ImageNet-pretrained ResNet-18 stem + OCR-tuned head.

    Stem (pretrained, frozen-or-finetuned):
      conv1 + bn1 + relu (stride 2): H/2 x W/2
      maxpool (stride 2):            H/4 x W/4
      layer1 (no stride):            H/4 x W/4, 64 ch
      layer2 (stride 2):             H/8 x W/8, 128 ch  -> stride forced to (2,1) so W stays at W/4

    OCR head (custom):
      conv 128 -> 256, pool (3,1) ->                    H/24 x W/4, 256 ch
      conv 256 -> 512, pool (H_remaining,1) ->          1 x W/4, 512 ch

    With H_in=96 the pre-stem H trace is 96 -> 48 -> 24 -> 24 -> 12 -> 4 -> 1.
    """

    def __init__(self, vocab_size: int, lstm_hidden: int = 256, lstm_layers: int = 2,
                 lstm_dropout: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = tvm.resnet18(weights=weights)
        except Exception:
            backbone = tvm.resnet18(pretrained=pretrained)
        # Pretrained low-level features.
        self.stem = nn.Sequential(
            backbone.conv1,    # 64ch, stride 2: 48 x W/2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # stride 2:        24 x W/4
            backbone.layer1,   # no stride:       24 x W/4, 64ch
        )
        # ResNet's layer2 default is stride (2,2). Patch its first block to stride (2,1)
        # so we keep horizontal resolution at W/4.
        layer2 = backbone.layer2
        first_block = layer2[0]
        first_block.conv1.stride = (2, 1)
        if first_block.downsample is not None:
            first_block.downsample[0].stride = (2, 1)
        self.layer2 = layer2  # 12 x W/4, 128ch

        # Custom head: H 12 -> 4 -> 1; channels 128 -> 256 -> 512.
        def conv_bn_relu(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.head_cnn = nn.Sequential(
            conv_bn_relu(128, 256),                                 # 12 x W/4
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),        # 4  x W/4
            conv_bn_relu(256, 512),                                 # 4  x W/4
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),        # 1  x W/4
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
        # x: (B, 1, 96, W). Replicate to 3 channels to match pretrained conv1.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.stem(x)             # (B, 64, 24, W/4)
        feat = self.layer2(feat)        # (B, 128, 12, W/4)
        feat = self.head_cnn(feat)      # (B, 512, 1, W/4)
        assert feat.shape[2] == 1, f"Expected H=1 after head, got {feat.shape}"
        feat = feat.squeeze(2)          # (B, 512, W/4)
        feat = feat.permute(2, 0, 1)    # (T, B, 512)
        out, _ = self.lstm(feat)
        logits = self.head(out)
        return logits.log_softmax(dim=2)

    @staticmethod
    def output_lengths(input_widths: torch.Tensor) -> torch.Tensor:
        return input_widths // WIDTH_DOWNSAMPLE


def build_model(backbone: str, vocab_size: int) -> nn.Module:
    if backbone == "vgg":
        return CRNN(vocab_size=vocab_size)
    if backbone == "resnet18":
        return CRNNResNet18(vocab_size=vocab_size)
    raise ValueError(f"Unknown backbone: {backbone!r}. Use 'vgg' or 'resnet18'.")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
