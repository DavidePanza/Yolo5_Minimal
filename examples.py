"""
Example Usage Script for Minimal YOLOv5
Demonstrates how to use the model for different tasks
"""

import torch
import numpy as np
from minimal_yolov5 import YOLOv5, YOLOLoss
import matplotlib.pyplot as plt


def example_1_create_model():
    """Example 1: Create and inspect the model."""
    print("=" * 80)
    print("EXAMPLE 1: Create and Inspect Model")
    print("=" * 80)
    
    # Create model
    model = YOLOv5(num_classes=80, channels=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Show architecture
    print(f"\nArchitecture:")
    print(f"  Input: 3 x 640 x 640")
    print(f"  Output: 25,200 predictions x 85 values")
    print(f"  Prediction format: [x, y, w, h, objectness, 80 class probs]")
    
    return model


def example_2_forward_pass():
    """Example 2: Perform a forward pass."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Forward Pass")
    print("=" * 80)
    
    model = YOLOv5(num_classes=80, channels=3)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"  Batch size: {output.shape[0]}")
    print(f"  Predictions per image: {output.shape[1]}")
    print(f"  Values per prediction: {output.shape[2]}")
    
    # Analyze output
    print(f"\nOutput statistics:")
    print(f"  X coordinates: [{output[0, :, 0].min():.2f}, {output[0, :, 0].max():.2f}]")
    print(f"  Y coordinates: [{output[0, :, 1].min():.2f}, {output[0, :, 1].max():.2f}]")
    print(f"  Objectness: [{output[0, :, 4].min():.4f}, {output[0, :, 4].max():.4f}]")
    
    return model, output


def example_3_loss_computation():
    """Example 3: Compute loss."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Loss Computation")
    print("=" * 80)
    
    model = YOLOv5(num_classes=80, channels=3)
    model.train()
    loss_fn = YOLOLoss(model)
    
    # Create dummy data
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640)
    
    # Create dummy targets (format: [img_idx, class, x, y, w, h])
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.2, 0.3],  # Image 0, class 1
        [0, 2, 0.3, 0.7, 0.15, 0.25], # Image 0, class 2
        [1, 0, 0.6, 0.4, 0.3, 0.4],   # Image 1, class 0
    ])
    
    print(f"\nInput: {dummy_input.shape}")
    print(f"Targets: {targets.shape} ({len(targets)} objects)")
    
    # Forward pass
    predictions = model(dummy_input)
    
    # Compute loss
    total_loss, loss_items = loss_fn(predictions, targets)
    
    print(f"\nLoss components:")
    print(f"  Box loss: {loss_items[0]:.4f}")
    print(f"  Objectness loss: {loss_items[1]:.4f}")
    print(f"  Classification loss: {loss_items[2]:.4f}")
    print(f"  Total loss: {total_loss:.4f}")
    
    return total_loss


def example_4_training_step():
    """Example 4: Single training step."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Training Step")
    print("=" * 80)
    
    # Setup
    model = YOLOv5(num_classes=80, channels=3)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    loss_fn = YOLOLoss(model)
    
    # Dummy data
    images = torch.randn(2, 3, 640, 640)
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.2, 0.3],
        [1, 0, 0.6, 0.4, 0.3, 0.4],
    ])
    
    print("Performing training step...")
    
    # Training step
    optimizer.zero_grad()
    predictions = model(images)
    loss, loss_items = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
    
    print(f"\nTraining step complete!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: ✓")
    print(f"  Weights updated: ✓")
    
    return model


def example_5_inference():
    """Example 5: Inference with filtering."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Inference with Filtering")
    print("=" * 80)
    
    model = YOLOv5(num_classes=80, channels=3)
    model.eval()
    
    # Dummy input
    image = torch.randn(1, 3, 640, 640)
    
    print("Running inference...")
    
    with torch.no_grad():
        predictions = model(image)[0]  # (25200, 85)
    
    print(f"\nRaw predictions: {predictions.shape}")
    
    # Filter by confidence
    conf_thresh = 0.5
    high_conf = predictions[predictions[:, 4] > conf_thresh]
    print(f"High confidence (>{conf_thresh}): {len(high_conf)}")
    
    # Filter by class confidence
    class_scores = high_conf[:, 5:].max(dim=1)[0]
    final_preds = high_conf[class_scores > 0.5]
    print(f"After class filtering: {len(final_preds)}")
    
    if len(final_preds) > 0:
        print(f"\nExample prediction:")
        pred = final_preds[0]
        print(f"  Box: ({pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}, {pred[3]:.1f})")
        print(f"  Objectness: {pred[4]:.4f}")
        print(f"  Class: {pred[5:].argmax().item()}")
        print(f"  Class confidence: {pred[5:].max():.4f}")
    
    return predictions


def example_6_model_analysis():
    """Example 6: Analyze model components."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Model Component Analysis")
    print("=" * 80)
    
    model = YOLOv5(num_classes=80, channels=3)
    
    # Analyze detection head
    print("\nDetection Head Configuration:")
    print(f"  Number of classes: {model.detect.nc}")
    print(f"  Outputs per anchor: {model.detect.no}")
    print(f"  Number of anchors: {model.detect.na}")
    print(f"  Number of layers: {model.detect.nl}")
    print(f"  Strides: {model.stride.tolist()}")
    
    print("\nAnchors (3 per scale):")
    for i, anchors in enumerate(model.anchors):
        stride = model.stride[i].item()
        print(f"  P{i+3}/{int(stride)}: {anchors}")
    
    # Layer-wise parameter count
    print("\nParameter Distribution:")
    backbone_params = 0
    neck_params = 0
    head_params = 0
    
    # This is simplified - in practice you'd iterate through named modules
    for name, param in model.named_parameters():
        if 'detect' in name:
            head_params += param.numel()
        elif any(x in name for x in ['conv6', 'conv7', 'conv8', 'conv9', 'upsample']):
            neck_params += param.numel()
        else:
            backbone_params += param.numel()
    
    total = backbone_params + neck_params + head_params
    print(f"  Backbone: {backbone_params:,} ({backbone_params/total*100:.1f}%)")
    print(f"  Neck: {neck_params:,} ({neck_params/total*100:.1f}%)")
    print(f"  Head: {head_params:,} ({head_params/total*100:.1f}%)")


def example_7_feature_maps():
    """Example 7: Visualize intermediate feature maps."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Feature Map Visualization")
    print("=" * 80)
    
    model = YOLOv5(num_classes=80, channels=3)
    model.eval()
    
    # Hook to capture intermediate outputs
    feature_maps = {}
    
    def hook(name):
        def fn(module, input, output):
            feature_maps[name] = output
        return fn
    
    # Register hooks
    model.c3_2.register_forward_hook(hook('P3'))
    model.c3_3.register_forward_hook(hook('P4'))
    model.sppf.register_forward_hook(hook('P5'))
    
    # Forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        _ = model(x)
    
    print("\nFeature map shapes:")
    for name, feat in feature_maps.items():
        print(f"  {name}: {feat.shape} "
              f"({feat.shape[2]}x{feat.shape[3]}, {feat.shape[1]} channels)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, feat) in zip(axes, feature_maps.items()):
        # Show first channel of first feature map
        img = feat[0, 0].cpu().numpy()
        ax.imshow(img, cmap='viridis')
        ax.set_title(f'{name} - First Channel\n{feat.shape[2]}x{feat.shape[3]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/feature_maps.png', dpi=150, bbox_inches='tight')
    print("\nFeature maps saved to: feature_maps.png")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MINIMAL YOLOV5 - EXAMPLE USAGE")
    print("=" * 80)
    
    # Run examples
    example_1_create_model()
    example_2_forward_pass()
    example_3_loss_computation()
    example_4_training_step()
    example_5_inference()
    example_6_model_analysis()
    example_7_feature_maps()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your dataset in YOLO format")
    print("2. Edit config.yaml with your paths")
    print("3. Run: python train_yolov5.py --config config.yaml --mode train")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
