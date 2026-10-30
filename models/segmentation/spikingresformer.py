import torch
import torch.nn as nn
from typing import Any, List, Mapping
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based import surrogate, neuron

from torch.nn.common_types import _size_2_t


class IF(neuron.IFNode):
    def __init__(self):
        super().__init__(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                         detach_reset=True, step_mode='m', backend='cupy', store_v_seq=False)


class LIF(neuron.LIFNode):
    def __init__(self, step_mode='m'):
        super().__init__(tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class PLIF(neuron.ParametricLIFNode):
    def __init__(self):
        super().__init__(init_tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class BN(nn.Module):
    def __init__(self, num_features, step_mode='m'):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == 'm':
            # Multi-step: Input is [T, N, C, H, W]
            if x.dim() != 5:
                raise ValueError(
                    f'Expected x with shape [T, N, C, H, W] for step_mode="m", but got {x.shape}!')
            return functional.seq_to_ann_forward(x, self.bn)

        elif self.step_mode == 's':
            # Single-step: Input is [N, C, H, W]
            if x.dim() != 4:
                raise ValueError(
                    f'Expected x with shape [N, C, H, W] for step_mode="s", but got {x.shape}!')
            return self.bn(x)

        else:
            raise ValueError(f'Invalid step_mode: {self.step_mode}')

class SpikingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, step_mode='m', activation=LIF, **kwargs):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size, stride, padding, step_mode=step_mode, bias=False)
        self.bn = BN(out_channels, step_mode=step_mode)
        self.activation = activation(step_mode=step_mode)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class Conv3x3(layer.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        step_mode='m',
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                         dilation=dilation, groups=groups, bias=bias, padding_mode='zeros',
                         step_mode=step_mode)


class Conv1x1(layer.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
        step_mode='m',
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         dilation=1, groups=1, bias=bias, padding_mode='zeros', step_mode=step_mode)


class Linear(layer.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        step_mode='m',
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, step_mode=step_mode)


class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=LIF, step_mode='m'):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(step_mode=step_mode),
            Conv1x1(in_channels, inner_channels, step_mode=step_mode),
            BN(inner_channels, step_mode=step_mode),
        )
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(step_mode=step_mode),
                    Conv3x3(inner_channels, inner_channels, groups=inner_channels // group_size, step_mode=step_mode),
                    BN(inner_channels, step_mode=step_mode),
                ))
        self.down = nn.Sequential(
            activation(step_mode=step_mode),
            Conv1x1(inner_channels, in_channels, step_mode=step_mode),
            BN(in_channels, step_mode=step_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, patch_size, activation=LIF, step_mode='m'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth
        self.step_mode = step_mode

        # Register firing rate buffers (adjust dimensions for step_mode)
        if step_mode == 'm':
            self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
            self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        else:  # step_mode='s'
            self.register_buffer('firing_rate_x', torch.zeros(1, num_heads, 1, 1))
            self.register_buffer('firing_rate_attn', torch.zeros(1, num_heads, 1, 1))

        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        # Initialize sub-layers with step_mode
        self.activation_in = activation(step_mode=step_mode)
        self.W = layer.Conv2d(dim, dim * 2, patch_size, patch_size, bias=False, step_mode=step_mode)
        self.norm = BN(dim * 2, step_mode=step_mode)
        self.matmul1 = SpikingMatmul('r')  # Ensure SpikingMatmul supports step_mode
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = activation(step_mode=step_mode)
        self.activation_out = activation(step_mode=step_mode)
        self.Wproj = Conv1x1(dim, dim, step_mode=step_mode)  # Ensure Conv1x1 supports step_mode
        self.norm_proj = BN(dim, step_mode=step_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == 'm':
            # Multi-step: Input [T, B, C, H, W]
            if x.dim() != 5:
                raise ValueError(f'Expected 5D input [T, B, C, H, W] for step_mode="m", but got {x.dim()}D input with shape {x.shape}')
            T, B, C, H, W = x.shape
            x_feat = x.clone()
            x = self.activation_in(x)

            y = self.W(x)
            y = self.norm(y)
            y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
            y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
            x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

            # Firing rate for x
            if self.training:
                firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
                if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                    self.firing_rate_x = firing_rate_x
                self.init_firing_rate_x = True
                self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (1 - self.momentum)
            scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))

            # Attention
            attn = self.matmul1(y1.transpose(-1, -2), x)
            attn = attn * scale1
            attn = self.activation_attn(attn)

            # Firing rate for attention
            if self.training:
                firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
                if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                    self.firing_rate_attn = firing_rate_attn
                self.init_firing_rate_attn = True
                self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (1 - self.momentum)
            scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)

            # Output
            out = self.matmul2(y2, attn)
            out = out * scale2
            out = out.reshape(T, B, C, H, W)
            out = self.activation_out(out)

            out = self.Wproj(out)
            out = self.norm_proj(out)
            out = out + x_feat
            return out

        else:  # step_mode='s'
            # Single-step: Input [B, C, H, W]
            if x.dim() != 4:
                raise ValueError(f'Expected 4D input [B, C, H, W] for step_mode="s", but got {x.dim()}D input with shape {x.shape}')
            B, C, H, W = x.shape
            x_feat = x.clone()
            x = self.activation_in(x)

            y = self.W(x)
            y = self.norm(y)
            y = y.reshape(B, self.num_heads, 2 * C // self.num_heads, -1)
            y1, y2 = y[:, :, :C // self.num_heads, :], y[:, :, C // self.num_heads:, :]
            x = x.reshape(B, self.num_heads, C // self.num_heads, -1)

            # Firing rate for x
            if self.training:
                firing_rate_x = x.detach().mean((0, 2, 3), keepdim=True)
                if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                    self.firing_rate_x = firing_rate_x
                self.init_firing_rate_x = True
                self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (1 - self.momentum)
            scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))

            # Attention
            attn = self.matmul1(y1.transpose(-1, -2), x)
            attn = attn * scale1
            attn = self.activation_attn(attn)

            # Firing rate for attention
            if self.training:
                firing_rate_attn = attn.detach().mean((0, 2, 3), keepdim=True)
                if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                    self.firing_rate_attn = firing_rate_attn
                self.init_firing_rate_attn = True
                self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (1 - self.momentum)
            scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)

            # Output
            out = self.matmul2(y2, attn)
            out = out * scale2
            out = out.reshape(B, C, H, W)
            out = self.activation_out(out)

            out = self.Wproj(out)
            out = self.norm_proj(out)
            out = out + x_feat
            return out

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF, step_mode='m') -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride, step_mode=step_mode)
        self.norm = BN(out_channels, step_mode=step_mode)
        self.activation = activation(step_mode=step_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

class SpikingSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=256, activation=neuron.LIFNode):
        super().__init__()
        self.conv1 = layer.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False, step_mode='m')
        self.bn1 = BN(hidden_dim)
        self.activation1 = activation()

        self.conv2 = layer.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=False, step_mode='m')
        self.bn2 = BN(num_classes)
        self.activation2 = activation()
        
        self.conv_transpose = layer.ConvTranspose2d(
            in_channels=10,
            out_channels=10,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
            step_mode='m'
        )
        self.activation3 = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        T = x.shape[0]
        B = x.shape[1]
        #print(f"before x: {x.shape}")
        x = self.activation1(x)
        #print(f"after activation1 x: {x.shape}")
        x = self.conv1(x)
        #print(f"after conv1 x: {x.shape}")
        x = self.bn1(x)

        x = self.activation2(x)
        #print(f"after activation2 x: {x.shape}")
        x = self.conv2(x)
        #print(f"after conv2 x: {x.shape}")
        x = self.bn2(x)
        x = x.view(T, B, -1, x.shape[3], x.shape[4])
        x = self.conv_transpose(x)
        x = self.activation3(x)
        x = x.view(T, B, 10, -1)
        return x

class SpikingDetectionHeadSSD(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_final_boxes: int = 10,
        step_mode='m',  # NEW: Add step_mode parameter
        **kwargs,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.num_final_boxes = num_final_boxes

        # Bounding box regression heads
        self.loc_heads = nn.ModuleList([
            SpikingConv2d(
                in_ch,
                num_anchors[i] * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                step_mode=step_mode,
                **kwargs
            )
            for i, in_ch in enumerate(in_channels)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        all_loc_preds = []
        for i, feature_map in enumerate(x):
            loc_pred = self.loc_heads[i](feature_map)  # [T, B, num_anchors[i]*4, H, W] or [B, num_anchors[i]*4, H, W]
            if self.step_mode == 'm':
                T, B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 1, 3, 4, 2).contiguous().view(T, B, -1, 4)
            else:  # step_mode='s'
                B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            all_loc_preds.append(loc_pred)

        # Concatenate all predictions
        if self.step_mode == 'm':
            all_loc_preds = torch.cat(all_loc_preds, dim=2)  # [T, B, total_anchors, 4]
        else:
            all_loc_preds = torch.cat(all_loc_preds, dim=1)  # [B, total_anchors, 4]

        # Select top 10 boxes
        if all_loc_preds.shape[-2] >= self.num_final_boxes:
            if self.step_mode == 'm':
                loc_preds = all_loc_preds[:, :, :self.num_final_boxes, :]  # [T, B, 10, 4]
            else:
                loc_preds = all_loc_preds[:, :self.num_final_boxes, :]  # [B, 10, 4]
        else:
            # Pad with zeros
            if self.step_mode == 'm':
                padding = torch.zeros(
                    (all_loc_preds.shape[0], all_loc_preds.shape[1], self.num_final_boxes - all_loc_preds.shape[2], 4),
                    device=all_loc_preds.device,
                    dtype=all_loc_preds.dtype
                )
                loc_preds = torch.cat([all_loc_preds, padding], dim=2)  # [T, B, 10, 4]
            else:
                padding = torch.zeros(
                    (all_loc_preds.shape[0], self.num_final_boxes - all_loc_preds.shape[1], 4),
                    device=all_loc_preds.device,
                    dtype=all_loc_preds.dtype
                )
                loc_preds = torch.cat([all_loc_preds, padding], dim=1)  # [B, 10, 4]
        return loc_preds
    
class SpikingFPNHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_anchors: List[int] = [9, 9, 9],
        num_final_boxes: int = 10,
        step_mode='m',  # NEW: Add step_mode parameter
        **kwargs,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.num_final_boxes = num_final_boxes

        # Lateral layers
        self.lateral_layers = nn.ModuleList([
            SpikingConv2d(
                in_ch, out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                step_mode=step_mode,
                **kwargs
            )
            for in_ch in in_channels
        ])

        # Top-down layers
        self.top_down_layers = nn.ModuleList([
            SpikingConv2d(
                out_channels, out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                step_mode=step_mode,
                **kwargs
            )
            for _ in range(len(in_channels) - 1)
        ])

        # Bounding box heads
        self.loc_heads = nn.ModuleList([
            SpikingConv2d(
                out_channels, num_anchors[i] * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                step_mode=step_mode,
                **kwargs
            )
            for i in range(len(in_channels))
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        pyramid = []
        pyramid.append(self.lateral_layers[-1](x[-1]))

        for i in range(len(x) - 2, -1, -1):
            # Flatten non-spatial dimensions for interpolation
            if self.step_mode == 'm':
                T, B, C, H, W = pyramid[-1].shape
                x_flat = pyramid[-1].reshape(T * B, C, H, W)
                upsampled_flat = nn.functional.interpolate(
                    x_flat,
                    size=x[i].shape[3:],
                    mode='nearest'
                )
                upsampled = upsampled_flat.reshape(T, B, C, *x[i].shape[3:])
            else:  # step_mode='s'
                B, C, H, W = pyramid[-1].shape
                upsampled = nn.functional.interpolate(
                    pyramid[-1],
                    size=x[i].shape[2:],
                    mode='nearest'
                )

            lateral = self.lateral_layers[i](x[i])
            pyramid.append(self.top_down_layers[len(x) - 2 - i](upsampled + lateral))

        pyramid.reverse()

        # Apply bounding box heads
        all_loc_preds = []
        for i, p in enumerate(pyramid):
            loc_pred = self.loc_heads[i](p)
            if self.step_mode == 'm':
                T, B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 1, 3, 4, 2).contiguous().view(T, B, -1, 4)
            else:
                B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            all_loc_preds.append(loc_pred)

        # Concatenate predictions
        if self.step_mode == 'm':
            all_loc_preds = torch.cat(all_loc_preds, dim=2)  # [T, B, total_anchors, 4]
        else:
            all_loc_preds = torch.cat(all_loc_preds, dim=1)  # [B, total_anchors, 4]

        # Select top 10 boxes
        if all_loc_preds.shape[-2] >= self.num_final_boxes:
            if self.step_mode == 'm':
                loc_preds = all_loc_preds[:, :, :self.num_final_boxes, :]  # [T, B, 10, 4]
            else:
                loc_preds = all_loc_preds[:, :self.num_final_boxes, :]  # [B, 10, 4]
        else:
            # Pad with zeros
            if self.step_mode == 'm':
                padding = torch.zeros(
                    (all_loc_preds.shape[0], all_loc_preds.shape[1], self.num_final_boxes - all_loc_preds.shape[2], 4),
                    device=all_loc_preds.device,
                    dtype=all_loc_preds.dtype
                )
                loc_preds = torch.cat([all_loc_preds, padding], dim=2)  # [T, B, 10, 4]
            else:
                padding = torch.zeros(
                    (all_loc_preds.shape[0], self.num_final_boxes - all_loc_preds.shape[1], 4),
                    device=all_loc_preds.device,
                    dtype=all_loc_preds.dtype
                )
                loc_preds = torch.cat([all_loc_preds, padding], dim=1)  # [B, 10, 4]
        return loc_preds

class SpikingResformer(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        num_classes=1000,
        prologue=None,
        group_size=64,
        activation=LIF,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                SpikingConv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode='m'),
                BN(planes[0]),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(planes[idx - 1], planes[idx], stride=2, activation=activation))
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(planes[idx], num_heads[idx], (img_size // patch_sizes[idx])**2,
                             patch_sizes[idx], activation=activation))
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(planes[idx], group_size=group_size, activation=activation))
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.classifier = Linear(planes[-1], num_classes)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def transfer(self, state_dict: Mapping[str, Any]):
        _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def no_weight_decay(self):
        ret = set()
        for name, module in self.named_modules():
            if isinstance(module, PLIF):
                ret.add(name + '.w')
        return ret
    
class SpikingResformerSeg(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        num_classes=1000,
        prologue=None,
        group_size=64,
        activation=LIF,
        hidden_dim=256,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                layer.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode='m'),
                BN(planes[0]),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(planes[idx - 1], planes[idx], stride=2, activation=activation))
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(planes[idx], num_heads[idx], (img_size // patch_sizes[idx])**2,
                             patch_sizes[idx], activation=activation))
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(planes[idx], group_size=group_size, activation=activation))
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.seg_head = SpikingSegHead(planes[-1], num_classes, hidden_dim)

    def transfer(self, state_dict: Mapping[str, Any]):
        _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)
        x = self.prologue(x)
        #print(f"after prologue x: {x.shape}")
        x = self.layers(x)
        #print(f"after layers x: {x.shape}")
        x = self.avgpool(x)
        #print(f"after avgpool x: {x.shape}")
        seg_logits = self.seg_head(x)
        return seg_logits

    def no_weight_decay(self):
        ret = set()
        for name, module in self.named_modules():
            if isinstance(module, PLIF):
                ret.add(name + '.w')
        return ret
    
    
class SpikingResformerSSD(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        num_anchors_per_level=[10, 10, 10],
        prologue=None,
        group_size=64,
        activation=LIF,
        step_mode='m',  # 'm' or 's'
        decoding='temporal',  # 'temporal', 'rate', or 'sum'
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.decoding = decoding
        self.step_mode = step_mode
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                layer.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode=step_mode),
                BN(planes[0], step_mode=step_mode),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode=step_mode),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        # Backbone layers
        self.layers = nn.ModuleList()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(planes[idx - 1], planes[idx], stride=2, activation=activation, step_mode=step_mode)
                )
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(planes[idx], num_heads[idx], (img_size // patch_sizes[idx])**2,
                             patch_sizes[idx], activation=activation, step_mode=step_mode)
                    )
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(planes[idx], group_size=group_size, activation=activation, step_mode=step_mode)
                    )
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        # Detection head
        self.detection_head = SpikingDetectionHeadSSD(
            in_channels=planes,
            num_anchors=num_anchors_per_level,
            step_mode=step_mode,
            **kwargs,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == 'm':
            # Multi-step: Input is [B, T, C, H, W] or [B, C, H, W]
            if x.dim() == 4:
                # [B, C, H, W] -> [T, B, C, H, W]
                x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            elif x.dim() == 5:
                # [B, T, C, H, W] -> [T, B, C, H, W]
                x = x.transpose(0, 1)
            else:
                raise ValueError(f"Expected 4D or 5D input for step_mode='m', got {x.dim()}D")

            # Forward pass (all T steps at once)
            x = self.prologue(x)
            feature_maps = []
            for layer in self.layers:
                x = layer(x)
                feature_maps.append(x)
            loc_preds = self.detection_head(feature_maps)
            return loc_preds  # [T, B, 10, 4]

        elif self.step_mode == 's':
            # Input: [B, C, H, W]
            if x.dim() != 4:
                raise ValueError(f"Expected 4D input [B, C, H, W] for step_mode='s'")
            x = self.prologue(x)
            feature_maps = []
            for layer in self.layers:
                x = layer(x)
                feature_maps.append(x)
            loc_preds = self.detection_head(feature_maps)  # [B, 10, 4]
            return loc_preds

        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")

     
class SpikingResformerFPN(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        out_channels=256,
        num_anchors=[9, 9, 9],
        prologue=None,
        group_size=64,
        activation=LIF,
        step_mode='m',  # 'm' or 's'
        decoding='temporal',  # 'temporal', 'rate', or 'sum' (only for step_mode='s')
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.step_mode = step_mode
        self.decoding = decoding
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                layer.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode=step_mode),
                BN(planes[0], step_mode=step_mode),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode=step_mode),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        # Backbone layers
        self.layers = nn.ModuleList()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(planes[idx - 1], planes[idx], stride=2, activation=activation, step_mode=step_mode)
                )
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(planes[idx], num_heads[idx], (img_size // patch_sizes[idx])**2,
                             patch_sizes[idx], activation=activation, step_mode=step_mode)
                    )
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(planes[idx], group_size=group_size, activation=activation, step_mode=step_mode)
                    )
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        # FPN head
        self.fpn_head = SpikingFPNHead(
            in_channels=planes,
            out_channels=out_channels,
            num_anchors=num_anchors,
            step_mode=step_mode,
            **kwargs,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == 'm':
            # Multi-step: Input is [B, T, C, H, W] or [B, C, H, W]
            if x.dim() == 4:
                # [B, C, H, W] -> [T, B, C, H, W]
                x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            elif x.dim() == 5:
                # [B, T, C, H, W] -> [T, B, C, H, W]
                x = x.transpose(0, 1)
            else:
                raise ValueError(f"Expected 4D or 5D input for step_mode='m', got {x.dim()}D")

            # Forward pass (all T steps at once)
            x = self.prologue(x)
            feature_maps = []
            for layer in self.layers:
                x = layer(x)
                feature_maps.append(x)
            loc_preds = self.fpn_head(feature_maps)
            return loc_preds  # [T, B, 10, 4]

        elif self.step_mode == 's':
            # Single-step: Input is [B, C, H, W]
            if x.dim() != 4:
                raise ValueError(f"Expected 4D input for step_mode='s', got {x.dim()}D")

            # Iterate T times to simulate T time steps
            outputs = []
            for _ in range(self.T):
                # Forward pass for one time step
                x_step = x  # Same input for all steps (or modify if needed)
                x_step = self.prologue(x_step)
                feature_maps = []
                for layer in self.layers:
                    x_step = layer(x_step)
                    feature_maps.append(x_step)
                loc_preds_step = self.fpn_head(feature_maps)  # [B, 10, 4]
                outputs.append(loc_preds_step)

            # Stack along time dimension: [T, B, 10, 4]
            return torch.stack(outputs, dim=0)

        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")
