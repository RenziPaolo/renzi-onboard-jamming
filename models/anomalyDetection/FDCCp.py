import torch
import torch.nn as nn

class InceptionResNet1DBlock(nn.Module):
    """A simplified 1D version of an Inception-ResNet block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
        )
        self.conv_proj = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.conv_proj(out)
        out = self.bn(out)
        out = self.relu(out + x)  # Residual connection
        return out

class AnomalyDetectionModel1D(nn.Module):
    def __init__(self, num_anomaly_types, input_channels=1):
        super().__init__()

        # Simplified 1D backbone (inspired by Inception-ResNet-v2)
        self.backbone = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            InceptionResNet1DBlock(128, 256),
            InceptionResNet1DBlock(256, 320),
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 1D convolutional head
        self.conv_head = nn.Sequential(
            # Block 1: Two 1D conv layers with 512 filters, batch norm, and ReLU
            nn.Conv1d(320, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Block 2: 1x1 conv layer with output channels equal to the number of anomaly types
            nn.Conv1d(512, num_anomaly_types, kernel_size=1),
        )

    def forward(self, x):
        features = self.backbone(x)
        anomaly_scores = self.conv_head(features)
        return anomaly_scores

    def pseudo_huber_loss(self, input, target, delta=1.0):
        error = torch.abs(input - target)
        quadratic = torch.clamp(error, max=delta)
        linear = error - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean()

    def compute_anomaly_scores(self, x):
        anomaly_scores = self.forward(x)
        anomaly_scores = torch.sqrt(anomaly_scores**2 + 1) - 1
        return anomaly_scores