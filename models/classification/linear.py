import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        """
        Simple linear model for classification.
        Args:
            input_dim (int): Flattened input dimension (e.g., 28x28=784 for MNIST).
            output_dim (int): Number of output classes (e.g., 10 for MNIST).
        """
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Linear layer + log softmax for classification
        x = self.fc(x)
        return x

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = SimpleLinearModel(input_dim=784, output_dim=10)
    print(model)

    # Example input (batch_size=3, 1 channel, 28x28 image)
    x = torch.randn(3, 1, 28, 28)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [3, 10]