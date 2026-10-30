import torch
import torch.nn as nn
import torch.nn.functional as F

class SAR_RF_Interference_Classifier(nn.Module):
    def __init__(self):
        super(SAR_RF_Interference_Classifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Output fixed size (8, 8)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 8 * 8)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path+".pt")
        return path


class SAR_RF_Interference_Classifier_simp(nn.Module):
    def __init__(self, input_channels=2, num_classes=1):
        super(SAR_RF_Interference_Classifier_simp, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Output fixed size (8, 8)

        # Fully connected layers
        self.fc2 = nn.Linear(32 * 8 * 8, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 32 * 8 * 8)

        # Fully connected layers with dropout
        x = self.dropout(self.fc2(x))

        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path+".pt")
        return path

if __name__ == "__main__":
    model = SAR_RF_Interference_Classifier()
    input_tensor = torch.randn(1, 1, 2000, 24000)  # Example input: batch_size=1, channels=1, height=2000, width=24000
    output = model(input_tensor)
    print(output)
