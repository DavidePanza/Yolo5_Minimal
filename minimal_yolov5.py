"""
Minimal YOLOv5 Implementation
A simplified version focusing on core components for learning object detection from scratch.
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# PART 1: BASIC BUILDING BLOCKS
# ============================================================================

def autopad(k, p=None):
    """Auto-padding to maintain spatial dimensions."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, 
                              autopad(kernel, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck block."""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions - Core YOLOv5 module."""
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv3 = Conv(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0)
                                 for _ in range(num_blocks)))
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast version."""
    def __init__(self, in_channels, out_channels, kernel=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    """Concatenate along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


# ============================================================================
# PART 2: DETECTION HEAD
# ============================================================================

class Detect(nn.Module):
    """YOLOv5 Detection Head - generates predictions."""
    
    def __init__(self, num_classes=80, anchors=(), channels=()):
        super().__init__()
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor (x, y, w, h, obj, classes)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        
        # Output convolutions for each detection layer
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in channels)
        self.stride = None  # computed during build
    
    def forward(self, x):
        """
        Args:
            x: list of feature maps from different scales
        Returns:
            predictions in format (batch, num_predictions, no)
        """
        z = []  # inference output
        
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            # Reshape: (bs, na*no, ny, nx) -> (bs, na, ny, nx, no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                # Decode predictions
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else torch.cat(z, 1)
    
    def _make_grid(self, nx=20, ny=20, i=0):
        """Generate grid for anchor boxes."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


# ============================================================================
# PART 3: YOLOV5 MODEL
# ============================================================================

class YOLOv5(nn.Module):
    """
    Minimal YOLOv5 Model
    
    Architecture:
    - Backbone: Extract features at multiple scales
    - Neck: FPN + PAN for feature fusion
    - Head: Detection predictions at 3 scales
    """
    
    def __init__(self, num_classes=80, channels=3):
        super().__init__()
        
        # Define channel dimensions for each layer
        # Format: [ch_out, kernel, stride]
        self.nc = num_classes
        
        # Anchors for 3 detection scales (small, medium, large objects)
        self.anchors = [
            [10, 13, 16, 30, 33, 23],      # P3/8
            [30, 61, 62, 45, 59, 119],     # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ]
        
        # ==================== BACKBONE ====================
        # Input: 3x640x640
        self.conv1 = Conv(channels, 32, 6, 2, 2)  # 32x320x320
        self.conv2 = Conv(32, 64, 3, 2)            # 64x160x160
        self.c3_1 = C3(64, 64, 1)                  # 64x160x160
        
        self.conv3 = Conv(64, 128, 3, 2)           # 128x80x80
        self.c3_2 = C3(128, 128, 2)                # 128x80x80
        
        self.conv4 = Conv(128, 256, 3, 2)          # 256x40x40
        self.c3_3 = C3(256, 256, 3)                # 256x40x40
        
        self.conv5 = Conv(256, 512, 3, 2)          # 512x20x20
        self.c3_4 = C3(512, 512, 1)                # 512x20x20
        self.sppf = SPPF(512, 512, 5)              # 512x20x20
        
        # ==================== NECK (FPN + PAN) ====================
        # Top-down pathway
        self.conv6 = Conv(512, 256, 1, 1)          # 256x20x20
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.c3_5 = C3(512, 256, 1, False)         # 256x40x40 (concat with P3)
        
        self.conv7 = Conv(256, 128, 1, 1)          # 128x40x40
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.c3_6 = C3(256, 128, 1, False)         # 128x80x80 (concat with P2)
        
        # Bottom-up pathway
        self.conv8 = Conv(128, 128, 3, 2)          # 128x40x40
        self.c3_7 = C3(384, 256, 1, False)         # Takes 128+256=384, outputs 256x40x40

        self.conv9 = Conv(256, 256, 3, 2)          # 256x20x20
        self.c3_8 = C3(768, 512, 1, False)         # Takes 256+512=768, outputs 512x20x20
        
        # ==================== HEAD ====================
        self.detect = Detect(num_classes, self.anchors, (128, 256, 512))
        
        # Initialize stride
        self.stride = torch.tensor([8., 16., 32.])
        self.detect.stride = self.stride
        self._initialize_biases()
    
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor (batch, 3, height, width)
        Returns:
            predictions during training: list of 3 prediction tensors
            predictions during inference: concatenated predictions (batch, num_predictions, no)
        """
        # Backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        
        x = self.conv3(x)
        p3 = self.c3_2(x)  # Save for FPN - 128x80x80
        
        x = self.conv4(p3)
        p4 = self.c3_3(x)  # Save for FPN - 256x40x40
        
        x = self.conv5(p4)
        x = self.c3_4(x)
        p5 = self.sppf(x)  # 512x20x20
        
        # Neck - Top-down
        x = self.conv6(p5)
        x = self.upsample1(x)
        x = torch.cat([x, p4], 1)
        p4_fused = self.c3_5(x)  # 256x40x40 - Save for bottom-up

        x = self.conv7(p4_fused)
        x = self.upsample2(x)
        x = torch.cat([x, p3], 1)
        out1 = self.c3_6(x)  # 128x80x80 - Small objects (P3)

        # Neck - Bottom-up
        x = self.conv8(out1)  # 128 channels
        x = torch.cat([x, p4_fused], 1)  # 128 + 256 = 384 channels
        out2 = self.c3_7(x)  # 256x40x40 - Medium objects (P4)
        
        x = self.conv9(out2)
        x = torch.cat([x, p5], 1)
        out3 = self.c3_8(x)  # 512x20x20 - Large objects (P5)
        
        # Detection head
        return self.detect([out1, out2, out3])
    
    def _initialize_biases(self):
        """Initialize biases for better convergence."""
        for mi, s in zip(self.detect.m, self.detect.stride):
            b = mi.bias.view(self.detect.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99999))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# ============================================================================
# PART 4: LOSS FUNCTION
# ============================================================================

def bbox_iou(box1, box2, eps=1e-7):
    """
    Calculate IoU between box1 and box2.
    Args:
        box1: (N, 4) in format [x, y, w, h]
        box2: (N, 4) in format [x, y, w, h]
    Returns:
        iou: (N,) IoU values
    """
    # Get coordinates
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    return inter / union


class YOLOLoss:
    """Compute loss for YOLOv5."""
    
    def __init__(self, model, hyp=None):
        device = next(model.parameters()).device
        self.device = device
        
        # Hyperparameters
        if hyp is None:
            hyp = {
                'box': 0.05,  # box loss gain
                'cls': 0.5,   # cls loss gain
                'obj': 1.0,   # obj loss gain
                'anchor_t': 4.0,  # anchor-multiple threshold
            }
        self.hyp = hyp
        
        # Get detection layer
        m = model.detect
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.stride = m.stride
        
        # Loss functions
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    
    def __call__(self, predictions, targets):
        """
        Compute loss.
        Args:
            predictions: list of 3 tensors of shape (bs, na, ny, nx, no)
            targets: (num_targets, 6) in format [img_idx, class, x, y, w, h] (normalized)
        Returns:
            total_loss: scalar
            loss_items: (box_loss, obj_loss, cls_loss)
        """
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # Calculate losses for each detection layer
        for i, pi in enumerate(predictions):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            n = b.shape[0]
            if n:
                # Get predictions for this target
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                
                # Box loss
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i])
                lbox += (1.0 - iou).mean()
                
                # Objectness loss
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification loss
                if self.nc > 1:
                    t = torch.zeros_like(pcls, device=self.device)
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.bce_cls(pcls, t)
            
            lobj += self.bce_obj(pi[..., 4], tobj)
        
        # Scale losses
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        
        return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach()
    
    def build_targets(self, predictions, targets):
        """Build targets for loss computation."""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # [1, 1, gx, gy, gw, gh, 1]

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # (na, nt, 7): [img_idx, cls, x, y, w, h, anchor_idx]
        
        g = 0.5  # bias
        off = torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
        ], device=self.device).float() * g
        
        for i in range(self.nl):
            anchors, shape = self.anchors[i], predictions[i].shape
            # shape should be (bs, na, ny, nx, no) which is 5 dimensions
            # Extract grid dimensions: nx=shape[3], ny=shape[2]
            if len(shape) != 5:
                raise ValueError(f"Expected predictions[{i}] to have 5 dimensions (bs, na, ny, nx, no), got shape {shape}")
            gain[2:6] = torch.tensor(shape, device=self.device)[[3, 2, 3, 2]]  # x, y, w, h gain

            t = targets * gain  # Scale targets to grid size
            if nt:
                # Match targets to anchors
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                
                # Offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indices, anch


# ============================================================================
# PART 5: SIMPLE TRAINING LOOP
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Simple training loop for one epoch."""
    model.train()
    loss_fn = YOLOLoss(model)
    
    print(f'\nEpoch {epoch}')
    pbar = enumerate(dataloader)
    
    for i, (imgs, targets) in pbar:
        imgs = imgs.to(device).float() / 255.0  # normalize to 0-1
        targets = targets.to(device)
        
        # Forward
        predictions = model(imgs)
        loss, loss_items = loss_fn(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f'  Batch {i}/{len(dataloader)}: '
                  f'Loss={loss.item():.4f} '
                  f'(box={loss_items[0]:.4f}, obj={loss_items[1]:.4f}, cls={loss_items[2]:.4f})')
    
    return loss.item()


# ============================================================================
# PART 6: EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Minimal YOLOv5 Implementation")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = YOLOv5(num_classes=80, channels=3)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of predictions per image: {output.shape[1]}")
    
    # Show model architecture
    print("\n3. Model Architecture Summary:")
    print("   Backbone: CSPDarknet53")
    print("   Neck: PANet")
    print("   Head: 3 detection layers (P3/8, P4/16, P5/32)")
    print("   Anchors per layer: 3")
    print("   Total predictions: (80x80 + 40x40 + 20x20) x 3 = 25,200")
    
    print("\n" + "=" * 80)
    print("To train this model:")
    print("1. Prepare your dataset in YOLO format")
    print("2. Create a DataLoader")
    print("3. Use the train_one_epoch() function")
    print("4. Add validation and checkpointing")
    print("=" * 80)
