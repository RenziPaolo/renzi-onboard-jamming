# spikingjelly.activation_based.examples.conv_fashion_mnist

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class CSNN(nn.Module):
    def __init__(self, T: int, h: int, w: int, channels: int, classes=1, use_cupy=False):
        super().__init__()
        
        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')
        
        self.T = T
        
        self.conv_fc = nn.Sequential(
        layer.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding=1, bias=False, step_mode='m'),
        layer.BatchNorm2d(1, step_mode='m'),
        neuron.IFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
        layer.MaxPool2d(2, 2, step_mode='m'), 


        layer.Flatten(step_mode='m'),
        layer.Linear(1 * (h//2) * (w//2), classes, bias=False, step_mode='m'),
        neuron.IFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
        layer.Linear(classes, classes, bias=False)

        )
        
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def get_layers(self):
        return [module for module in self.squential.modules() if isinstance(module, neuron.LIFNode)]

    
if __name__ == "__main__":
    model = CSNN(T=10, h=28, w=28, channels=1, classes=10, use_cupy=True)
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input: batch_size=1, channels=1, height=224, width=224
    output = model(input_tensor)
    print(output)