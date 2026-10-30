import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class AnomalyImageDetectionModel(nn.Module):
    def __init__(self, num_anomaly_types):
        super(AnomalyImageDetectionModel, self).__init__()

        # Load pre-trained Inception-v3 and freeze all layers
        backbone = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        # Extract the first three stages (up to the 'Mixed_5b' layer)
        self.backbone = nn.Sequential(
            backbone.conv2d_1a,
            backbone.conv2d_2a,
            backbone.conv2d_2b,
            backbone.maxpool_3a,
            backbone.conv2d_3b,
            backbone.conv2d_4a,
            backbone.conv2d_4b,
            backbone.repeat_1,
            backbone.mixed_5b,
        )

        # Custom convolutional head
        self.conv_head = nn.Sequential(
            # Block 1: Two 3x3 conv layers with 512 filters, batch norm, and ReLU
            nn.Conv2d(320, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Block 2: 1x1 conv layer with output channels equal to the number of anomaly types
            nn.Conv2d(512, num_anomaly_types, kernel_size=1),
        )

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)
        # Forward pass through the convolutional head
        anomaly_scores = self.conv_head(features)
        return anomaly_scores

