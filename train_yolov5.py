"""
Complete Training Script for Minimal YOLOv5
Includes data loading, training loop, validation, and utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import yaml
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from minimal_yolov5 import YOLOv5, YOLOLoss


# ============================================================================
# DATA LOADING
# ============================================================================

class YOLODataset(Dataset):
    """
    YOLO format dataset loader.
    
    Expected structure:
    dataset/
        images/
            train/
                img1.jpg
                img2.jpg
        labels/
            train/
                img1.txt  (class x_center y_center width height - normalized)
                img2.txt
    """
    
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get all image files
        self.img_files = list(self.img_dir.glob('*.jpg')) + \
                        list(self.img_dir.glob('*.png')) + \
                        list(self.img_dir.glob('*.jpeg'))
        
        print(f"Found {len(self.img_files)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    labels.append([cls, x, y, w, h])
        
        labels = np.array(labels) if labels else np.zeros((0, 5))
        
        # Letterbox resize (maintain aspect ratio)
        img, ratio, (dw, dh) = letterbox(img, self.img_size, auto=False)
        
        # Adjust labels for letterbox
        if len(labels):
            labels[:, 1] = ratio[0] * labels[:, 1] + dw / self.img_size
            labels[:, 2] = ratio[1] * labels[:, 2] + dh / self.img_size
            labels[:, 3] *= ratio[0]
            labels[:, 4] *= ratio[1]
        
        # Convert to tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), torch.from_numpy(labels)


def letterbox(img, new_shape=640, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize image with letterboxing."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def collate_fn(batch):
    """Custom collate function for batching."""
    imgs, labels = zip(*batch)

    # Add sample index to labels and reorder to [img_idx, class, x, y, w, h]
    new_labels = []
    for i, l in enumerate(labels):
        if len(l):
            # Convert to float32 for MPS compatibility
            l = l.float()
            # Prepend image index to create [img_idx, class, x, y, w, h]
            l = torch.cat([torch.full((len(l), 1), i, dtype=torch.float32), l], 1)
            new_labels.append(l)

    # Concatenate all labels (skip empty ones)
    labels = torch.cat(new_labels, 0) if new_labels else torch.zeros((0, 6), dtype=torch.float32)

    return torch.stack(imgs, 0), labels


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataset, device, num_images=4, conf_thresh=0.5):
    """Visualize model predictions."""
    model.eval()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            img, labels = dataset[idx]
            img_display = img.numpy().transpose(1, 2, 0).copy()
            
            # Get predictions
            img_tensor = img.unsqueeze(0).to(device).float() / 255.0
            pred = model(img_tensor)[0]  # (1, num_preds, 85)
            
            # Filter by confidence
            pred = pred[pred[:, 4] > conf_thresh]
            
            # Draw predictions
            ax.imshow(img_display)
            for p in pred:
                x, y, w, h = p[:4].cpu().numpy()
                x1, y1 = x - w/2, y - h/2
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                        edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                conf = p[4].item()
                cls = p[5:].argmax().item()
                ax.text(x1, y1-5, f'C{cls}:{conf:.2f}', 
                       color='red', fontsize=8, weight='bold')
            
            ax.set_title(f'Image {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(cfg_path):
    """
    Main training function.
    
    Config file format (YAML):
    
    train_img_dir: 'dataset/images/train'
    train_label_dir: 'dataset/labels/train'
    val_img_dir: 'dataset/images/val'
    val_label_dir: 'dataset/labels/val'
    
    num_classes: 80
    img_size: 640
    batch_size: 16
    epochs: 100
    lr: 0.01
    momentum: 0.937
    weight_decay: 0.0005
    
    save_dir: 'runs/train/exp'
    """
    
    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = YOLODataset(
        cfg['train_img_dir'],
        cfg['train_label_dir'],
        img_size=cfg['img_size'],
        augment=True
    )
    
    val_dataset = YOLODataset(
        cfg['val_img_dir'],
        cfg['val_label_dir'],
        img_size=cfg['img_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = YOLOv5(num_classes=cfg['num_classes'], channels=3)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['epochs']
    )
    
    # Loss function
    loss_fn = YOLOLoss(model)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=20)
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(cfg['epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        print(f"{'='*80}")
        
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc='Training')
        
        for imgs, targets in pbar:
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            
            # Forward
            predictions = model(imgs)
            loss, loss_items = loss_fn(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_items[0]:.4f}',
                'obj': f'{loss_items[1]:.4f}',
                'cls': f'{loss_items[2]:.4f}'
            })
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc='Validation'):
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
                
                predictions = model(imgs)
                loss, _ = loss_fn(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                save_dir / 'best.pt'
            )
        
        # Save last model
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            save_dir / 'last.pt'
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break
        
        # Visualize predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            fig = visualize_predictions(model, val_dataset, device)
            fig.savefig(save_dir / f'predictions_epoch_{epoch+1}.png')
            plt.close(fig)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_history.png')
    plt.close()
    
    print(f"\nTraining complete! Best val loss: {best_loss:.4f}")
    print(f"Models saved in: {save_dir}")


# ============================================================================
# INFERENCE
# ============================================================================

def detect(model_path, img_path, conf_thresh=0.5, device='cuda'):
    """Run inference on a single image."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = YOLOv5(num_classes=80, channels=3)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, _, _ = letterbox(img, 640)
    img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))[::-1].copy())
    img_tensor = img_tensor.unsqueeze(0).to(device).float() / 255.0
    
    # Predict
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # Filter predictions
    pred = pred[pred[:, 4] > conf_thresh]
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_resized)
    
    for p in pred:
        x, y, w, h = p[:4].cpu().numpy()
        x1, y1 = x - w/2, y - h/2
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        conf = p[4].item()
        cls = p[5:].argmax().item()
        ax.text(x1, y1-5, f'Class {cls}: {conf:.2f}', 
               color='red', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    return fig, pred


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Minimal YOLOv5')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'detect'])
    parser.add_argument('--weights', type=str, help='Path to weights for detection')
    parser.add_argument('--source', type=str, help='Path to image for detection')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'detect':
        fig, preds = detect(args.weights, args.source)
        plt.show()
        print(f"\nDetected {len(preds)} objects")
