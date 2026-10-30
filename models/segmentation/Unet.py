import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron
from typing import List, Optional
import torch.nn.functional as F
from spikingjelly.activation_based import functional

class SpikingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, step_mode='m', activation=neuron.LIFNode, **kwargs):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size, stride, padding, step_mode=step_mode, bias=False)
        self.bn = layer.BatchNorm2d(out_channels, step_mode=step_mode)
        self.activation = activation(step_mode=step_mode, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: Tensor of shape (N, 4) representing boxes (x1, y1, x2, y2).
        boxes2: Tensor of shape (M, 4) representing boxes (x1, y1, x2, y2).

    Returns:
        IoU matrix of shape (N, M).
    """

    # Extract coordinates
    x11, y11, x12, y12 = boxes1.unbind(dim=-1)
    x21, y21, x22, y22 = boxes2.unbind(dim=-1)

    # Calculate intersection area
    x_left = torch.max(x11[:, None], x21)
    y_top = torch.max(y11[:, None], y21)
    x_right = torch.min(x12[:, None], x22)
    y_bottom = torch.min(y12[:, None], y22)

    intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

    # Calculate areas of each box
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    # Calculate union area
    union = area1[:, None] + area2 - intersection

    # Compute IoU
    iou = intersection / union.clamp(min=1e-6)  # Avoid division by zero

    return iou

class SpikingDetectionHeadSSDnew(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_final_boxes: int = 10,
        step_mode: str = 'm',
        learnable_anchors: bool = True,
        use_nms: bool = True,
        common_channels: int = 38,  # Must match your U-Net's base_channels
        **kwargs,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.num_final_boxes = num_final_boxes
        self.learnable_anchors = learnable_anchors
        self.use_nms = use_nms
        self.num_anchors = num_anchors
        self.common_channels = common_channels

        # Channel adapters: 1x1 conv to reduce each feature map to common_channels
        self.channel_adapters = nn.ModuleList([
            SpikingConv2d(
                in_ch,
                common_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                step_mode=step_mode,
                **kwargs
            )
            for in_ch in in_channels
        ])

        # Shared head: Predicts max_anchors * 4 channels from common_channels
        max_anchors = max(num_anchors)
        self.loc_head = SpikingConv2d(
            common_channels,
            max_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            step_mode=step_mode,
            **kwargs
        )

        # Generate base anchors for each feature map
        self.anchor_boxes = self._generate_base_anchors(num_anchors)
        if learnable_anchors:
            self.anchor_boxes = nn.ParameterList([
                nn.Parameter(anchors.clone()) for anchors in self.anchor_boxes
            ])
        else:
            for i, anchors in enumerate(self.anchor_boxes):
                self.register_buffer(f'anchor_boxes_{i}', anchors)

    def _generate_base_anchors(self, num_anchors: List[int]) -> List[torch.Tensor]:
        """Generate base anchors (centered at 0.5, 0.5) for each feature map."""
        anchor_boxes = []
        aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.333]
        scales = [0.2, 0.4, 0.6, 0.8, 1.0]

        for i, num_a in enumerate(num_anchors):
            ratios = aspect_ratios[:num_a]
            anchors = []
            for ar in ratios:
                scale = scales[i % len(scales)]
                width = scale * (ar ** 0.5)
                height = scale / (ar ** 0.5)
                x1 = 0.5 - width / 2
                y1 = 0.5 - height / 2
                x2 = 0.5 + width / 2
                y2 = 0.5 + height / 2
                anchors.append([x1, y1, x2, y2])
            anchor_boxes.append(torch.tensor(anchors, dtype=torch.float32))
        return anchor_boxes

    def _offsets_to_bboxes(self, offsets, anchors, H, W):
        """Convert offsets (dx, dy, dw, dh) to absolute bboxes (x1, y1, x2, y2)."""
        anchors = anchors.to(offsets.device)
        anchors = anchors.view(1, 1, 1, -1, 4)

        h_indices = torch.arange(H, device=offsets.device).float() / H
        w_indices = torch.arange(W, device=offsets.device).float() / W
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        h_grid = h_grid.view(1, H, W, 1)
        w_grid = w_grid.view(1, H, W, 1)

        anchor_cx_base = (anchors[..., 0] + anchors[..., 2]) / 2
        anchor_cy_base = (anchors[..., 1] + anchors[..., 3]) / 2
        anchor_w = anchors[..., 2] - anchors[..., 0]
        anchor_h = anchors[..., 3] - anchors[..., 1]

        anchor_cx = w_grid + (anchor_cx_base - 0.5)
        anchor_cy = h_grid + (anchor_cy_base - 0.5)

        dx, dy, dw, dh = offsets[..., 0], offsets[..., 1], offsets[..., 2], offsets[..., 3]

        pred_cx = anchor_cx + dx * anchor_w
        pred_cy = anchor_cy + dy * anchor_h
        pred_w = anchor_w * torch.exp(dw)
        pred_h = anchor_h * torch.exp(dh)

        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2

        return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

    def _nms(self, boxes, iou_threshold=0.5):
        """Non-Maximum Suppression (NMS)."""
        if boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        iou_matrix = calculate_iou(boxes, boxes)
        areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        order = areas.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            ious = iou_matrix[i, order[1:]]
            mask = ious < iou_threshold
            order = order[1:][mask]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        all_boxes = []

        for i, feature_map in enumerate(x):
            # --- CRITICAL FIX: Reset spiking neuron states for this feature map ---
            functional.reset_net(self.channel_adapters[i])  # Reset channel adapter state
            functional.reset_net(self.loc_head)              # Reset shared head state

            # Adapt channels to common_channels
            feature_map = self.channel_adapters[i](feature_map)

            # Predict offsets
            loc_pred = self.loc_head(feature_map)

            # Reshape to [T/B, H, W, max_anchors, 4]
            if self.step_mode == 'm':
                T, B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H, W, -1, 4)
            else:
                B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, -1, 4)

            # Slice to actual anchors for this feature map
            anchors = self.anchor_boxes[i]
            A = anchors.shape[0]
            loc_pred = loc_pred[..., :A, :]  # [T/B, H, W, A, 4]

            # Convert offsets to bboxes
            bboxes = self._offsets_to_bboxes(loc_pred, anchors, H, W)

            # Flatten spatial dimensions
            if self.step_mode == 'm':
                bboxes = bboxes.view(T, B, -1, 4)
            else:
                bboxes = bboxes.view(B, -1, 4)

            all_boxes.append(bboxes)

        # Concatenate all boxes
        if self.step_mode == 'm':
            all_boxes = torch.cat(all_boxes, dim=2)  # [T, B, total_boxes, 4]
        else:
            all_boxes = torch.cat(all_boxes, dim=1)  # [B, total_boxes, 4]

        # Select top num_final_boxes
        if self.use_nms:
            if self.step_mode == 'm':
                final_boxes = []
                for t in range(all_boxes.shape[0]):
                    for b in range(all_boxes.shape[1]):
                        boxes = all_boxes[t, b]
                        keep = self._nms(boxes)
                        if len(keep) > self.num_final_boxes:
                            areas = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
                            topk_indices = areas.argsort(descending=True)[:self.num_final_boxes]
                            keep = keep[topk_indices]
                        if len(keep) < self.num_final_boxes:
                            padding = torch.zeros(
                                (self.num_final_boxes - len(keep), 4),
                                device=all_boxes.device,
                                dtype=all_boxes.dtype
                            )
                            keep_boxes = torch.cat([boxes[keep], padding], dim=0)
                        else:
                            keep_boxes = boxes[keep]
                        final_boxes.append(keep_boxes.unsqueeze(0).unsqueeze(0))
                loc_preds = torch.cat(final_boxes, dim=0)  # [T, B, num_final_boxes, 4]
            else:
                final_boxes = []
                for b in range(all_boxes.shape[0]):
                    boxes = all_boxes[b]
                    keep = self._nms(boxes)
                    if len(keep) > self.num_final_boxes:
                        areas = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
                        topk_indices = areas.argsort(descending=True)[:self.num_final_boxes]
                        keep = keep[topk_indices]
                    if len(keep) < self.num_final_boxes:
                        padding = torch.zeros(
                            (self.num_final_boxes - len(keep), 4),
                            device=all_boxes.device,
                            dtype=all_boxes.dtype
                        )
                        keep_boxes = torch.cat([boxes[keep], padding], dim=0)
                    else:
                        keep_boxes = boxes[keep]
                    final_boxes.append(keep_boxes.unsqueeze(0))
                loc_preds = torch.cat(final_boxes, dim=0)  # [B, num_final_boxes, 4]
        else:
            # Fallback to truncation/padding
            if all_boxes.shape[-2] >= self.num_final_boxes:
                loc_preds = all_boxes[..., :self.num_final_boxes, :]
            else:
                if self.step_mode == 'm':
                    padding = torch.zeros(
                        (all_boxes.shape[0], all_boxes.shape[1], self.num_final_boxes - all_boxes.shape[2], 4),
                        device=all_boxes.device,
                        dtype=all_boxes.dtype,
                    )
                    loc_preds = torch.cat([all_boxes, padding], dim=2)
                else:
                    padding = torch.zeros(
                        (all_boxes.shape[0], self.num_final_boxes - all_boxes.shape[1], 4),
                        device=all_boxes.device,
                        dtype=all_boxes.dtype,
                    )
                    loc_preds = torch.cat([all_boxes, padding], dim=1)

        return loc_preds

class SpikingDetectionHeadSSD(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],  # Number of anchors PER SPATIAL LOCATION
        num_final_boxes: int = 10,
        step_mode: str = 'm',
        learnable_anchors: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.num_final_boxes = num_final_boxes
        self.learnable_anchors = learnable_anchors

        # Location heads: Predict 4 offsets (dx, dy, dw, dh) per anchor
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


        # Generate BASE anchors (not flattened) for each feature map: [A, 4]
        self.anchor_boxes = self._generate_base_anchors(num_anchors)
        if learnable_anchors:
            self.anchor_boxes = nn.ParameterList([
                nn.Parameter(anchors.clone()) for anchors in self.anchor_boxes
            ])
        else:
            for i, anchors in enumerate(self.anchor_boxes):
                self.register_buffer(f'anchor_boxes_{i}', anchors)

    def _generate_base_anchors(self, num_anchors: List[int]) -> List[torch.Tensor]:
        """
        Generate BASE anchor boxes (centered at (0.5, 0.5)) for each feature map.
        Returns:
            List of tensors of shape [A, 4] (A = num_anchors[i]).
        """
        anchor_boxes = []
        aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.333]  # Common aspect ratios
        scales = [0.2, 0.4, 0.6, 0.8, 1.0]  # Scales per feature map level

        for i, num_a in enumerate(num_anchors):
            ratios = aspect_ratios[:num_a]
            anchors = []
            for ar in ratios:
                scale = scales[i % len(scales)]
                width = scale * (ar ** 0.5)
                height = scale / (ar ** 0.5)
                # Base anchors centered at (0.5, 0.5)
                x1 = 0.5 - width / 2
                y1 = 0.5 - height / 2
                x2 = 0.5 + width / 2
                y2 = 0.5 + height / 2
                anchors.append([x1, y1, x2, y2])
            anchor_boxes.append(torch.tensor(anchors, dtype=torch.float32))  # [A, 4]
        return anchor_boxes

    def _offsets_to_bboxes(
        self,
        offsets: torch.Tensor,  # [T/B, H, W, A, 4]
        anchors: torch.Tensor,  # [A, 4] (base anchors)
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Convert offsets (dx, dy, dw, dh) to absolute bboxes (x1, y1, x2, y2).
        Args:
            offsets: [T/B, H, W, A, 4] (dx, dy, dw, dh).
            anchors: [A, 4] (base anchors centered at (0.5, 0.5)).
            H, W: Spatial dimensions of the feature map.
        Returns:
            Absolute bboxes: [T/B, H, W, A, 4] (x1, y1, x2, y2).
        """
        # Move anchors to the same device as offsets
        anchors = anchors.to(offsets.device)

        # Expand anchors to [1, 1, 1, A, 4]
        anchors = anchors.view(1, 1, 1, -1, 4)

        # Create full grid of (h, w) positions (normalized to [0,1])
        h_indices = torch.arange(H, device=offsets.device).float() / H  # [H]
        w_indices = torch.arange(W, device=offsets.device).float() / W  # [W]
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')  # [H, W], [H, W]

        # Reshape to [1, H, W, 1] for broadcasting
        h_grid = h_grid.view(1, H, W, 1)  # [1, H, W, 1]
        w_grid = w_grid.view(1, H, W, 1)  # [1, H, W, 1]

        # Base anchor centers (relative to (0.5, 0.5))
        anchor_cx_base = (anchors[..., 0] + anchors[..., 2]) / 2  # [1, 1, 1, A]
        anchor_cy_base = (anchors[..., 1] + anchors[..., 3]) / 2  # [1, 1, 1, A]

        # Adjust centers to spatial grid: [1, H, W, A]
        anchor_cx = w_grid + (anchor_cx_base - 0.5)  # [1, H, W, 1] + [1, 1, 1, A] -> [1, H, W, A]
        anchor_cy = h_grid + (anchor_cy_base - 0.5)  # [1, H, W, 1] + [1, 1, 1, A] -> [1, H, W, A]

        # Anchor widths/heights: [1, 1, 1, A]
        anchor_w = anchors[..., 2] - anchors[..., 0]
        anchor_h = anchors[..., 3] - anchors[..., 1]

        # Extract offsets: [T/B, H, W, A]
        dx, dy, dw, dh = offsets[..., 0], offsets[..., 1], offsets[..., 2], offsets[..., 3]

        # Convert offsets to absolute coordinates
        pred_cx = anchor_cx + dx * anchor_w
        pred_cy = anchor_cy + dy * anchor_h
        pred_w = anchor_w * torch.exp(dw)
        pred_h = anchor_h * torch.exp(dh)

        # Convert to (x1, y1, x2, y2)
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2

        # Stack to [x1, y1, x2, y2]: [T/B, H, W, A, 4]
        bboxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
        return bboxes

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        all_loc_preds = []

        for i, feature_map in enumerate(x):
            # Predict offsets: [T/B, C, H, W]
            loc_pred = self.loc_heads[i](feature_map)
            # Scale offsets to a reasonable range (e.g., [-2, 2] -> [0, 1] via sigmoid)
            loc_pred = loc_pred * 4.0 - 2.0

            # Reshape to [T/B, H, W, A, 4]
            if self.step_mode == 'm':
                T, B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H, W, -1, 4)
            else:
                B, _, H, W = loc_pred.shape
                loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, -1, 4)

            # Get base anchors for this feature map: [A, 4]
            anchors = self.anchor_boxes[i]

            # Convert offsets to absolute bboxes: [T/B, H, W, A, 4]
            bboxes = self._offsets_to_bboxes(loc_pred, anchors, H, W)

            # Flatten spatial dimensions: [T/B, H*W*A, 4]
            if self.step_mode == 'm':
                bboxes = bboxes.view(T, B, -1, 4)  # [T, B, H*W*A, 4]
            else:
                bboxes = bboxes.view(B, -1, 4)  # [B, H*W*A, 4]

            all_loc_preds.append(bboxes)

        # Concatenate all feature map predictions
        if self.step_mode == 'm':
            all_loc_preds = torch.cat(all_loc_preds, dim=2)  # [T, B, total_anchors, 4]
        else:
            all_loc_preds = torch.cat(all_loc_preds, dim=1)  # [B, total_anchors, 4]

        # Truncate or pad to num_final_boxes
        if all_loc_preds.shape[-2] >= self.num_final_boxes:
            if self.step_mode == 'm':
                loc_preds = all_loc_preds[:, :, :self.num_final_boxes, :]
            else:
                loc_preds = all_loc_preds[:, :self.num_final_boxes, :]
        else:
            if self.step_mode == 'm':
                padding = torch.zeros(
                    (all_loc_preds.shape[0], all_loc_preds.shape[1], self.num_final_boxes - all_loc_preds.shape[2], 4),
                    device=all_loc_preds.device,
                    dtype=all_loc_preds.dtype,
                )
                loc_preds = torch.cat([all_loc_preds, padding], dim=2)
            else:
                padding = torch.zeros(
                    (all_loc_preds.shape[0], self.num_final_boxes - all_loc_preds.shape[1], 4),
                    device=all_loc_preds.device,
                    dtype=all_loc_preds.dtype,
                )
                loc_preds = torch.cat([all_loc_preds, padding], dim=1)

        return loc_preds

class SpikingFPNHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_final_boxes: int = 10,
        step_mode: str = 'm',
        **kwargs,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.num_final_boxes = num_final_boxes

        # Shared head for all feature maps
        self.conv = SpikingConv2d(
            in_channels[0],  # Use first feature map's channels
            256,  # Intermediate channels
            kernel_size=3,
            padding=1,
            step_mode=step_mode,
            **kwargs
        )
        self.loc_head = SpikingConv2d(
            256,
            4 * num_final_boxes,  # Predict all boxes at once
            kernel_size=3,
            padding=1,
            step_mode=step_mode,
            **kwargs
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Upsample and add lateral connections (simplified FPN)
        feature_maps = []
        for i, feature_map in enumerate(x):
            if i > 0:
                # Upsample previous feature map
                upsampled = F.interpolate(
                    feature_maps[-1],
                    size=feature_map.shape[-2:],
                    mode='nearest'
                )
                # Add lateral connection
                feature_map = upsampled + feature_map
            # Apply shared conv
            feature_map = self.conv(feature_map)
            feature_maps.append(feature_map)

        # Predict boxes from the first feature map (simplified)
        boxes = self.loc_head(feature_maps[0])

        # Reshape to [T/B, num_final_boxes, 4]
        if self.step_mode == 'm':
            T, B, _, H, W = boxes.shape
            boxes = boxes.permute(0, 1, 3, 4, 2).contiguous().view(T, B, -1, 4)
        else:
            B, _, H, W = boxes.shape
            boxes = boxes.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

        # Select top num_final_boxes (simplified)
        if self.step_mode == 'm':
            boxes = boxes[:, :, :self.num_final_boxes, :]
        else:
            boxes = boxes[:, :self.num_final_boxes, :]

        return boxes



# --- Main Model ---
class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        step_mode: str = 'm',
        T: int = 10,
        head_type: str = 'fpn',  # 'ssd' or 'fpn'
        num_final_boxes: int = 10,
        model_size: float = 1.0, 
        learnable_anchors: bool = False,  # For detection head
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.step_mode = step_mode
        self.T = T
        self.head_type = head_type
        self.num_final_boxes = num_final_boxes


        self.channel_multiplier = model_size

        self.base_channels = int(64 * self.channel_multiplier)
        self.channels = {
            'c1': self.base_channels,  
            'c2': self.base_channels*2,
            'c3': self.base_channels*4,  
            'c4': self.base_channels*8,  
            'c5': self.base_channels*16, 

        }

        # --- Encoder ---
        self.conv1 = SpikingConv2d(in_channels, self.base_channels, step_mode=step_mode, **kwargs)
        self.conv2 = SpikingConv2d(self.base_channels, self.base_channels, step_mode=step_mode, **kwargs)
        self.pooling1 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.pool_d1 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.conv_d1 = SpikingConv2d(
            self.base_channels,
            self.channels['c2'],
            kernel_size=1,
            stride=1,
            padding=0,
            step_mode=step_mode,
            **kwargs
        )

        self.conv3 = SpikingConv2d(self.base_channels, self.channels['c2'], step_mode=step_mode, **kwargs)
        self.conv4 = SpikingConv2d(
            self.channels['c2'] * 2,  # x_d1 + x_2
            self.channels['c2'],
            step_mode=step_mode,
            **kwargs
        )
        self.pooling2 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.pool_d2 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.conv_d2 = SpikingConv2d(
            self.channels['c2'],
            self.channels['c3'],
            kernel_size=1,
            stride=1,
            padding=0,
            step_mode=step_mode,
            **kwargs
        )

        self.conv5 = SpikingConv2d(self.channels['c2'], self.channels['c3'], step_mode=step_mode, **kwargs)
        self.conv6 = SpikingConv2d(
            self.channels['c3'] * 2,  # x_d2 + x_3
            self.channels['c3'],
            step_mode=step_mode,
            **kwargs
        )
        self.pooling3 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.pool_d3 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.conv_d3 = SpikingConv2d(
            self.channels['c3'],
            self.channels['c4'],
            kernel_size=1,
            stride=1,
            padding=0,
            step_mode=step_mode,
            **kwargs
        )

        self.conv7 = SpikingConv2d(self.channels['c3'], self.channels['c4'], step_mode=step_mode, **kwargs)
        self.conv8 = SpikingConv2d(
            self.channels['c4'] * 2,  # x_d3 + x_4
            self.channels['c4'],
            step_mode=step_mode,
            **kwargs
        )
        self.pooling4 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.pool_d4 = layer.AvgPool2d(kernel_size=2, stride=2, step_mode=step_mode)
        self.conv_d4 = SpikingConv2d(
            self.channels['c4'],
            self.channels['c5'],
            kernel_size=1,
            stride=1,
            padding=0,
            step_mode=step_mode,
            **kwargs
        )

        self.conv9 = SpikingConv2d(self.channels['c4'], self.channels['c5'], step_mode=step_mode, **kwargs)
        self.conv10 = SpikingConv2d(
            self.channels['c5'] * 2,  # x_d4 + x_5
            self.channels['c5'],
            step_mode=step_mode,
            **kwargs
        )

        # --- Decoder ---
        self.convtranspose1 = layer.ConvTranspose2d(
            in_channels=self.channels['c5'],
            out_channels=self.channels['c4'],
            stride=2,
            kernel_size=3,
            padding=1,
            output_padding=1,
            bias=False,
            step_mode=step_mode,
        )
        self.conv11 = SpikingConv2d(
            self.channels['c5'],  # x_5 + up1
            self.channels['c4'],
            step_mode=step_mode,
            **kwargs
        )
        self.conv12 = SpikingConv2d(
            self.channels['c4'],
            self.channels['c4'],
            step_mode=step_mode,
            **kwargs
        )

        self.convtranspose2 = layer.ConvTranspose2d(
            in_channels=self.channels['c4'],
            out_channels=self.channels['c3'],
            stride=2,
            kernel_size=3,
            padding=1,
            output_padding=1,
            bias=False,
            step_mode=step_mode,
        )
        self.conv13 = SpikingConv2d(
            self.channels['c4'], 
            self.channels['c3'],
            step_mode=step_mode,
            **kwargs
        )
        self.conv14 = SpikingConv2d(
            self.channels['c3'],
            self.channels['c3'],
            step_mode=step_mode,
            **kwargs
        )

        self.convtranspose3 = layer.ConvTranspose2d(
            in_channels=self.channels['c3'],
            out_channels=self.channels['c2'],
            stride=2,
            kernel_size=3,
            padding=1,
            output_padding=1,
            bias=False,
            step_mode=step_mode,
        )
        self.conv15 = SpikingConv2d(
            self.channels['c3'],  # x_3 + up3
            self.channels['c2'],
            step_mode=step_mode,
            **kwargs
        )
        self.conv16 = SpikingConv2d(
            self.channels['c2'],
            self.channels['c2'],
            step_mode=step_mode,
            **kwargs
        )

        self.convtranspose4 = layer.ConvTranspose2d(
            in_channels=self.channels['c2'],
            out_channels=self.channels['c1'],
            stride=2,
            kernel_size=3,
            padding=1,
            output_padding=1,
            bias=False,
            step_mode=step_mode,
        )
        self.conv17 = SpikingConv2d(
            self.channels['c2'],  # x_2 + up4
            self.channels['c1'],
            step_mode=step_mode,
            **kwargs
        )
        self.conv18 = SpikingConv2d(
            self.channels['c1'],
            self.channels['c1'],
            step_mode=step_mode,
            **kwargs
        )

        # --- Detection Head ---
        # Compute in_channels for the detection head based on the scaled channels
        # The feature maps passed to the head are [x_9, x_8, x_7, x_6, x_5]
        # Their channels are: c18, c16, c14, c12, c10
        detection_in_channels = [
            self.channels['c1'],  # x_9
            self.channels['c2'],  # x_8
            self.channels['c3'],  # x_7
            self.channels['c4'],  # x_6
            self.channels['c5'],  # x_5
        ]

        if head_type == 'ssd':
            self.detection_head = SpikingDetectionHeadSSD(
                in_channels=detection_in_channels,
                num_anchors=[4, 4, 4, 4, 4],
                num_final_boxes=num_final_boxes,
                step_mode=step_mode,
                learnable_anchors=learnable_anchors,
                **kwargs,
            )
        elif head_type == 'fpn':
            self.detection_head = SpikingFPNHead(
                in_channels=detection_in_channels,
                num_anchors=[9, 9, 9, 9, 9],
                num_final_boxes=num_final_boxes,
                step_mode=step_mode,
                learnable_anchors=learnable_anchors,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):
        # Ensure T is present
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            if x.shape[1] != self.T:
                x = x.transpose(0, 1)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        # --- Encoder ---
        x_1 = self.conv1(x)
        x_1 = self.conv2(x_1)
        x_d1 = self.conv_d1(self.pool_d1(x_1))
        down_x1 = self.pooling1(x_1)

        x_2 = self.conv3(down_x1)
        x_2 = torch.cat((x_d1, x_2), dim=2)
        x_2 = self.conv4(x_2)
        x_d2 = self.conv_d2(self.pool_d2(x_2))
        down_x2 = self.pooling2(x_2)

        x_3 = self.conv5(down_x2)
        x_3 = torch.cat((x_d2, x_3), dim=2)
        x_3 = self.conv6(x_3)
        x_d3 = self.conv_d3(self.pool_d3(x_3))
        down_x3 = self.pooling3(x_3)

        x_4 = self.conv7(down_x3)
        x_4 = torch.cat((x_d3, x_4), dim=2)
        x_4 = self.conv8(x_4)
        x_d4 = self.conv_d4(self.pool_d4(x_4))
        down_x4 = self.pooling4(x_4)

        x_5 = self.conv9(down_x4)
        x_5 = torch.cat((x_d4, x_5), dim=2)
        x_5 = self.conv10(x_5)

        # --- Decoder ---
        up1 = self.convtranspose1(x_5)
        concat1 = torch.cat((x_4, up1), dim=2)
        x_6 = self.conv11(concat1)
        x_6 = self.conv12(x_6)

        up2 = self.convtranspose2(x_6)
        concat2 = torch.cat((x_3, up2), dim=2)
        x_7 = self.conv13(concat2)
        x_7 = self.conv14(x_7)

        up3 = self.convtranspose3(x_7)
        concat3 = torch.cat((x_2, up3), dim=2)
        x_8 = self.conv15(concat3)
        x_8 = self.conv16(x_8)

        up4 = self.convtranspose4(x_8)
        concat4 = torch.cat((x_1, up4), dim=2)
        x_9 = self.conv17(concat4)
        x_9 = self.conv18(x_9)

        # --- Detection Head ---
        feature_maps = [x_9, x_8, x_7, x_6, x_5]
        box_preds = self.detection_head(feature_maps)  # [T, B, 10, 4]
        return box_preds
    
if __name__ == "__main__":
    import torch

    def test_model_with_random_tensor(
        model,
        batch_size: int = 2,
        in_channels: int = 3,
        height: int = 256,
        width: int = 256,
        T: int = 10,
        input_is_spikes: bool = True,
        verbose: bool = True,
    ) -> bool:
        """
        Test the model with a random tensor of parametric dimensions.

        Args:
            model: The Denoising_Model instance to test.
            batch_size: Batch size (B).
            in_channels: Number of input channels (C).
            height: Input height (H).
            width: Input width (W).
            T: Number of time steps (for spiking input).
            input_is_spikes: If True, input is binary spikes (0 or 1). If False, use random floats.
            verbose: If True, print input/output shapes and test result.

        Returns:
            bool: True if the test passes, False otherwise.
        """
        # Set model to eval mode
        model.eval()

        # Generate random input tensor
        if input_is_spikes:
            # Binary spikes (0 or 1)
            x = torch.randint(0, 2, (batch_size, in_channels, height, width)).float()
        else:
            # Random floats (e.g., for non-spiking input)
            x = torch.rand(batch_size, in_channels, height, width)

        # Expected output shape: [T, B, 10, 4]
        expected_output_shape = (T, batch_size, 10, 4)

        with torch.no_grad():
            # Forward pass
            output = model(x)

            # Check output shape
            if output.shape != expected_output_shape:
                if verbose:
                    print(f"❌ Test FAILED: Output shape mismatch.")
                    print(f"   Input shape: {x.shape}")
                    print(f"   Expected output shape: {expected_output_shape}")
                    print(f"   Actual output shape: {output.shape}")
                return False

            if verbose:
                print(f"✅ Test PASSED!")
                print(f"   Input shape: {x.shape}")
                print(f"   Output shape: {output.shape}")
                print(f"   Model works with dimensions (B={batch_size}, C={in_channels}, H={height}, W={width}, T={T})")
            return True
    
    model = Unet(in_channels=2, step_mode='m', T=8, head_type='fpn', model_size=0.5)

    # Test with 4D input (B, C, H, W)
    success = test_model_with_random_tensor(
        model,
        batch_size=4,
        in_channels=2,
        height=512,
        width=512,
        T=8,
        input_is_spikes=False,
        verbose=True,
    )
    
    print(f"Test with 4D input {'PASSED' if success else 'FAILED'}")