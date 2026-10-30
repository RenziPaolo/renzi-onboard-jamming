import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate

class SAR_RF_Interference_Classifier_SNN(nn.Module):
    def __init__(self, h, w, classes, channels, tau=2.0):
        super(SAR_RF_Interference_Classifier_SNN, self).__init__()
        self.T = 1
        self.squential = nn.Sequential(
            layer.Flatten(),
            layer.Linear(h * w * channels, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(10, classes, bias=False)
        )

    def forward(self, x):
        x = self.squential(x)

        return x

    def get_layers(self):
        return [module for module in self.squential.modules() if isinstance(module, neuron.LIFNode)]

class SAR_RF_Interference_Classifier_SNN_duble(nn.Module):
    def __init__(self, h, w, classes, channels, tau=2.0):
        super(SAR_RF_Interference_Classifier_SNN, self).__init__()
        self.T = 1
        self.squential = nn.Sequential(
            layer.Flatten(),
            layer.Linear(h * w * channels, (h * w * channels)/2, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                layer.Linear((h * w * channels)/2, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(10, classes, bias=False)
        )

    def forward(self, x):
        x = self.squential(x)

        return x

    def get_layers(self):
        return [module for module in self.squential.modules() if isinstance(module, neuron.LIFNode)]

if __name__ == "__main__":
    h, w = 224, 224  # Example input dimensions
    model = SAR_RF_Interference_Classifier_SNN(h, w)
    input_tensor = torch.randn(1, 1, h, w)  # Example input: batch_size=1, channels=1, height=2000, width=24000
    output = model(input_tensor)
    print(output)
