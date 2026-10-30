import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class GroupWiseHybridAttention(nn.Module):
    def __init__(self, in_channels, groups=4):
        super(GroupWiseHybridAttention, self).__init__()
        self.groups = groups
        self.group_channels = in_channels // groups
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(self.group_channels) for _ in range(groups)
        ])
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention() for _ in range(groups)
        ])

    def forward(self, x):
        b, c, h, w = x.size()
        group_features = torch.split(x, self.group_channels, dim=1)
        attended_features = []
        for i in range(self.groups):
            ca = self.channel_attentions[i](group_features[i])
            sa = self.spatial_attentions[i](ca)
            attended_features.append(sa)
        return torch.cat(attended_features, dim=1)

class LCNN_GWHA(nn.Module):
    def __init__(self, num_classes=2):
        super(LCNN_GWHA, self).__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # Group-wise hybrid attention blocks
        self.gwha1 = GroupWiseHybridAttention(64)
        self.gwha2 = GroupWiseHybridAttention(128)
        self.gwha3 = GroupWiseHybridAttention(256)
        self.gwha4 = GroupWiseHybridAttention(512)
        # Downsampling
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gwha1(x)
        x = F.relu(self.down1(x))
        x = self.gwha2(x)
        x = F.relu(self.down2(x))
        x = self.gwha3(x)
        x = F.relu(self.down3(x))
        x = self.gwha4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path+".pt")
        return path

if __name__ == "__main__":
    model = LCNN_GWHA(num_classes=7)
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input: batch_size=1, channels=3, height=256, width=256
    output = model(input_tensor)
    print(output)
