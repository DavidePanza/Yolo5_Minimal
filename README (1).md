# Minimal YOLOv5: Learn Object Detection from Scratch

A minimal, educational implementation of YOLOv5 that retains the core components while being simple enough to understand and modify.

## 📚 What You'll Learn

This implementation helps you understand:

1. **YOLO Architecture Components**
   - Backbone (CSPDarknet53)
   - Neck (PANet for feature fusion)
   - Head (Multi-scale detection)

2. **Key Concepts**
   - Anchor-based detection
   - Feature pyramid networks
   - Loss computation (box, objectness, classification)
   - Target assignment

3. **Training Pipeline**
   - Data loading in YOLO format
   - Training loop with validation
   - Checkpointing and early stopping

## 🏗️ Architecture Overview

```
Input (640x640x3)
       ↓
[BACKBONE: CSPDarknet53]
   ├→ P3 (80x80)   - Small objects
   ├→ P4 (40x40)   - Medium objects  
   └→ P5 (20x20)   - Large objects
       ↓
[NECK: PANet]
   - Feature fusion across scales
       ↓
[HEAD: Detect]
   - 3 detection layers
   - 3 anchors per layer
   - Total: 25,200 predictions
```

## 📁 File Structure

```
minimal-yolov5/
├── minimal_yolov5.py      # Core model implementation
├── train_yolov5.py        # Training script with data loading
├── config.yaml            # Training configuration
├── README.md              # This file
└── dataset/               # Your dataset (YOLO format)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch torchvision numpy opencv-python pillow pyyaml tqdm matplotlib
```

### 2. Prepare Your Dataset

YOLO format expects:
- Images: JPG/PNG files
- Labels: TXT files with same name as images

Label format (one line per object):
```
class_id x_center y_center width height
```

All coordinates are normalized (0-1).

Example `image1.txt`:
```
0 0.5 0.5 0.3 0.4
2 0.3 0.7 0.1 0.2
```

### 3. Test the Model

```python
from minimal_yolov5 import YOLOv5
import torch

# Create model
model = YOLOv5(num_classes=80, channels=3)

# Test forward pass
x = torch.randn(1, 3, 640, 640)
output = model(x)
print(f"Output shape: {output.shape}")  # (1, 25200, 85)
```

### 4. Train the Model

Edit `config.yaml` with your dataset paths and run:

```bash
python train_yolov5.py --config config.yaml --mode train
```

### 5. Run Inference

```bash
python train_yolov5.py --mode detect --weights runs/train/exp/best.pt --source image.jpg
```

## 📖 Detailed Components

### 1. Conv Block (Conv + BatchNorm + SiLU)

```python
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, ...)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
```

**Why?** Standard building block that normalizes and activates features.

### 2. C3 Module (CSP Bottleneck)

```python
class C3(nn.Module):
    # Cross Stage Partial connection
    # Splits features, processes part through bottlenecks,
    # then concatenates for efficient feature extraction
```

**Why?** Reduces computation while maintaining accuracy through partial connections.

### 3. SPPF (Spatial Pyramid Pooling Fast)

```python
class SPPF(nn.Module):
    # Applies multiple max pooling at different scales
    # Captures multi-scale context
```

**Why?** Helps detect objects at different sizes by aggregating features at multiple scales.

### 4. Detect Head

```python
class Detect(nn.Module):
    # Predicts: [x, y, w, h, objectness, class_probs]
    # Uses anchors to predict relative to grid cells
```

**Why?** Converts feature maps to bounding box predictions.

### 5. Loss Function

```python
class YOLOLoss:
    # Combines three losses:
    # 1. Box loss (IoU-based)
    # 2. Objectness loss (confidence)
    # 3. Classification loss
```

**Why?** Trains the model to accurately localize and classify objects.

## 🎯 Training Tips

### For Small Datasets (<1000 images)

```yaml
epochs: 50
batch_size: 8
lr: 0.001
```

### For Medium Datasets (1000-10000 images)

```yaml
epochs: 100
batch_size: 16
lr: 0.01
```

### For Large Datasets (>10000 images)

```yaml
epochs: 300
batch_size: 32
lr: 0.01
```

### GPU Memory Issues?

- Reduce `batch_size`
- Reduce `img_size` to 416 or 320
- Use mixed precision training

## 🔍 Understanding the Code

### Model Forward Pass

```python
def forward(self, x):
    # 1. Backbone extracts features at multiple scales
    p3 = backbone_stage1(x)  # 80x80
    p4 = backbone_stage2(p3) # 40x40
    p5 = backbone_stage3(p4) # 20x20
    
    # 2. Neck fuses features (top-down + bottom-up)
    out1 = neck_upsample_fuse(p5, p4, p3)  # 80x80
    out2 = neck_downsample_fuse(out1, p4)  # 40x40
    out3 = neck_downsample_fuse(out2, p5)  # 20x20
    
    # 3. Head predicts boxes
    return detect([out1, out2, out3])
```

### Loss Computation

```python
def compute_loss(predictions, targets):
    # For each prediction layer:
    for pred in predictions:
        # 1. Match predictions to targets using anchors
        matched = match_targets_to_anchors(targets, anchors)
        
        # 2. Compute losses
        box_loss = iou_loss(pred_boxes, target_boxes)
        obj_loss = bce_loss(pred_objectness, has_object)
        cls_loss = bce_loss(pred_classes, target_classes)
    
    return box_loss + obj_loss + cls_loss
```

## 🛠️ Customization Guide

### Change Number of Classes

```python
model = YOLOv5(num_classes=YOUR_NUM_CLASSES, channels=3)
```

### Modify Architecture

```python
# In YOLOv5.__init__(), change:
self.c3_3 = C3(256, 256, num_blocks=3)  # Increase depth
# or
self.conv4 = Conv(128, 256, 3, 2)       # Change channels
```

### Add Custom Augmentation

```python
# In YOLODataset.__getitem__():
if self.augment:
    img = self.random_flip(img)
    img = self.random_brightness(img)
    # Add more augmentations
```

## 📊 Monitoring Training

Training outputs:
- `best.pt`: Best model weights
- `last.pt`: Latest checkpoint
- `training_history.png`: Loss curves
- `predictions_epoch_X.png`: Visual results

## 🐛 Debugging Tips

### Model Not Learning?

1. Check data loading:
```python
dataloader = DataLoader(dataset, batch_size=1)
imgs, labels = next(iter(dataloader))
print(f"Image shape: {imgs.shape}")
print(f"Labels shape: {labels.shape}")
```

2. Verify loss computation:
```python
loss_fn = YOLOLoss(model)
loss, items = loss_fn(predictions, targets)
print(f"Box: {items[0]}, Obj: {items[1]}, Cls: {items[2]}")
```

3. Check predictions:
```python
model.eval()
with torch.no_grad():
    preds = model(imgs)
print(f"Predictions shape: {preds.shape}")
print(f"Confidence range: {preds[:, :, 4].min()}-{preds[:, :, 4].max()}")
```

### Loss Exploding?

- Reduce learning rate
- Check for NaN in data
- Ensure proper normalization

### Poor Detection?

- Increase training epochs
- Try different anchor sizes
- Add data augmentation
- Check label quality

## 📚 Further Reading

1. **YOLOv5 Paper**: "YOLOv5" by Ultralytics
2. **CSPNet**: "CSPNet: A New Backbone that can Enhance Learning Capability of CNN"
3. **PANet**: "Path Aggregation Network for Instance Segmentation"
4. **Focal Loss**: For handling class imbalance

## 🤝 Contributing

This is an educational project. Feel free to:
- Add comments explaining complex parts
- Implement additional features
- Create tutorials
- Fix bugs

## ⚠️ Limitations

This minimal version:
- No mosaic augmentation
- No automatic anchor optimization
- No mixed precision training
- No model ensembling
- Simplified data augmentation

For production use, consider the full YOLOv5 implementation.

## 📝 License

Educational use. Based on YOLOv5 (AGPL-3.0).

## 🎓 Learning Path

1. **Beginner**: Run the example, understand forward pass
2. **Intermediate**: Modify architecture, train on custom data
3. **Advanced**: Implement new loss functions, add augmentations
4. **Expert**: Optimize for speed, implement new features

---

**Happy Learning! 🚀**

For questions or improvements, please open an issue or submit a pull request.
