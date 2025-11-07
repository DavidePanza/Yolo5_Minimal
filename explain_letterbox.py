"""
Letterbox Resizing Explained
Understanding how YOLO preprocesses images and adjusts bounding box coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw


def visualize_letterbox_concept():
    """
    Visual explanation of letterbox resizing
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Letterbox Resizing: Step-by-Step Explanation', fontsize=16, fontweight='bold')
    
    # ============================================================================
    # STEP 1: Original Image
    # ============================================================================
    ax = axes[0, 0]
    
    # Create example original image (not square)
    original_h, original_w = 480, 640  # Width > Height
    img_original = np.ones((original_h, original_w, 3)) * 0.8
    
    # Draw object (car) as a colored rectangle
    obj_x, obj_y, obj_w, obj_h = 200, 150, 150, 100  # in pixels
    img_original[obj_y:obj_y+obj_h, obj_x:obj_x+obj_w] = [0.2, 0.4, 0.8]  # Blue car
    
    ax.imshow(img_original)
    rect = patches.Rectangle((obj_x, obj_y), obj_w, obj_h, 
                              linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # Show center point
    center_x, center_y = obj_x + obj_w/2, obj_y + obj_h/2
    ax.plot(center_x, center_y, 'r*', markersize=15)
    
    ax.set_title(f'STEP 1: Original Image\nSize: {original_w}×{original_h}', fontweight='bold')
    ax.text(10, 30, f'Object bbox (pixels):\nxy: ({obj_x}, {obj_y})\nwh: ({obj_w}, {obj_h})', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    ax.axis('off')
    
    # ============================================================================
    # STEP 2: Naive Resize (WRONG - Distorts aspect ratio)
    # ============================================================================
    ax = axes[0, 1]
    
    target_size = 640
    # Naive resize stretches the image
    img_naive = np.ones((target_size, target_size, 3)) * 0.8
    
    # Scale factors
    scale_x = target_size / original_w  # 640/640 = 1.0
    scale_y = target_size / original_h  # 640/480 = 1.33
    
    # Scaled object (distorted)
    obj_x_naive = int(obj_x * scale_x)
    obj_y_naive = int(obj_y * scale_y)
    obj_w_naive = int(obj_w * scale_x)
    obj_h_naive = int(obj_h * scale_y)
    
    img_naive[obj_y_naive:obj_y_naive+obj_h_naive, 
              obj_x_naive:obj_x_naive+obj_w_naive] = [0.2, 0.4, 0.8]
    
    ax.imshow(img_naive)
    rect = patches.Rectangle((obj_x_naive, obj_y_naive), obj_w_naive, obj_h_naive,
                              linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    ax.set_title('STEP 2: Naive Resize (WRONG) ❌\nDistorted aspect ratio!', 
                 fontweight='bold', color='red')
    ax.text(10, 30, f'Problem: Image stretched\nscale_x={scale_x:.2f}, scale_y={scale_y:.2f}\nCar looks taller!', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=9)
    ax.axis('off')
    
    # ============================================================================
    # STEP 3: Letterbox Resize (CORRECT)
    # ============================================================================
    ax = axes[0, 2]
    
    # Letterbox maintains aspect ratio
    img_letterbox = np.ones((target_size, target_size, 3)) * 0.5  # Gray padding
    
    # Calculate scale (use minimum to fit inside)
    ratio = min(target_size / original_w, target_size / original_h)  # 640/640 = 1.0
    new_w = int(original_w * ratio)  # 640
    new_h = int(original_h * ratio)  # 480
    
    # Calculate padding
    pad_w = (target_size - new_w) // 2  # (640-640)/2 = 0
    pad_h = (target_size - new_h) // 2  # (640-480)/2 = 80
    
    # Place resized image in center
    img_letterbox[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = 0.8
    
    # Scaled object (correct aspect ratio)
    obj_x_lb = int(obj_x * ratio) + pad_w
    obj_y_lb = int(obj_y * ratio) + pad_h
    obj_w_lb = int(obj_w * ratio)
    obj_h_lb = int(obj_h * ratio)
    
    img_letterbox[obj_y_lb:obj_y_lb+obj_h_lb, 
                  obj_x_lb:obj_x_lb+obj_w_lb] = [0.2, 0.4, 0.8]
    
    ax.imshow(img_letterbox)
    rect = patches.Rectangle((obj_x_lb, obj_y_lb), obj_w_lb, obj_h_lb,
                              linewidth=3, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    
    # Show padding
    ax.axhline(y=pad_h, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=target_size-pad_h, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_title('STEP 3: Letterbox Resize (CORRECT) ✓\nAspect ratio preserved', 
                 fontweight='bold', color='green')
    ax.text(10, 30, f'ratio={ratio:.2f}\nnew_size: {new_w}×{new_h}\npadding: {pad_w},{pad_h}', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)
    ax.axis('off')
    
    # ============================================================================
    # STEP 4: YOLO Format Labels (Normalized coordinates)
    # ============================================================================
    ax = axes[1, 0]
    
    # Original YOLO labels (normalized 0-1)
    yolo_x_orig = (obj_x + obj_w/2) / original_w  # center x normalized
    yolo_y_orig = (obj_y + obj_h/2) / original_h  # center y normalized
    yolo_w_orig = obj_w / original_w
    yolo_h_orig = obj_h / original_h
    
    # Visualization
    ax.text(0.5, 0.7, 'Original YOLO Labels', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.5, f'class: 0 (car)\nx_center: {yolo_x_orig:.3f}\ny_center: {yolo_y_orig:.3f}\n'
                      f'width: {yolo_w_orig:.3f}\nheight: {yolo_h_orig:.3f}',
            ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(0.5, 0.2, '(Normalized to original image size)', ha='center', fontsize=10, style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ============================================================================
    # STEP 5: Label Adjustment for Letterbox
    # ============================================================================
    ax = axes[1, 1]
    
    # Adjusted labels for letterbox
    # This is what the code does:
    yolo_x_adj = ratio * yolo_x_orig + (pad_w / target_size)
    yolo_y_adj = ratio * yolo_y_orig + (pad_h / target_size)
    yolo_w_adj = ratio * yolo_w_orig
    yolo_h_adj = ratio * yolo_h_orig
    
    ax.text(0.5, 0.8, 'Label Adjustment Math', ha='center', fontsize=14, fontweight='bold')
    
    # Show the formulas
    formulas = [
        f"x_new = ratio × x_old + dw/img_size",
        f"      = {ratio:.3f} × {yolo_x_orig:.3f} + {pad_w}/{target_size}",
        f"      = {yolo_x_adj:.3f}",
        "",
        f"y_new = ratio × y_old + dh/img_size",
        f"      = {ratio:.3f} × {yolo_y_orig:.3f} + {pad_h}/{target_size}",
        f"      = {yolo_y_adj:.3f}",
        "",
        f"w_new = ratio × w_old = {yolo_w_adj:.3f}",
        f"h_new = ratio × h_old = {yolo_h_adj:.3f}",
    ]
    
    ax.text(0.5, 0.45, '\n'.join(formulas), ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ============================================================================
    # STEP 6: Why This Matters
    # ============================================================================
    ax = axes[1, 2]
    
    explanation = """
    Why Letterbox + Label Adjustment?
    
    1. PRESERVES ASPECT RATIO
       • Objects maintain correct proportions
       • No distortion in training data
    
    2. CONSISTENT INPUT SIZE
       • All images → 640×640
       • Required for batch processing
    
    3. ACCURATE BOUNDING BOXES
       • Labels adjusted to match new positions
       • Model learns correct spatial relationships
    
    4. BETTER PERFORMANCE
       • Model doesn't learn distorted features
       • Generalizes better to real-world data
    
    Key Insight:
    The padding shifts object positions,
    so we must shift the labels too!
    """
    
    ax.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def code_walkthrough():
    """
    Detailed walkthrough of the letterbox code
    """
    
    print("=" * 80)
    print("LETTERBOX CODE WALKTHROUGH")
    print("=" * 80)
    
    # Example values
    original_w, original_h = 640, 480
    target_size = 640
    
    print(f"\nOriginal image: {original_w}×{original_h}")
    print(f"Target size: {target_size}×{target_size}")
    
    # Step 1: Calculate ratio
    ratio = min(target_size / original_w, target_size / original_h)
    print(f"\nStep 1: Calculate scaling ratio")
    print(f"  ratio = min({target_size}/{original_w}, {target_size}/{original_h})")
    print(f"  ratio = min({target_size/original_w:.3f}, {target_size/original_h:.3f})")
    print(f"  ratio = {ratio:.3f}")
    print(f"  → We use the SMALLER ratio to ensure image fits inside target")
    
    # Step 2: Calculate new unpadded size
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)
    print(f"\nStep 2: Scale image")
    print(f"  new_width = {original_w} × {ratio:.3f} = {new_w}")
    print(f"  new_height = {original_h} × {ratio:.3f} = {new_h}")
    
    # Step 3: Calculate padding
    dw = (target_size - new_w) / 2
    dh = (target_size - new_h) / 2
    print(f"\nStep 3: Calculate padding")
    print(f"  dw (width padding) = ({target_size} - {new_w}) / 2 = {dw:.1f}")
    print(f"  dh (height padding) = ({target_size} - {new_h}) / 2 = {dh:.1f}")
    print(f"  → Padding adds gray bars to center the image")
    
    # Step 4: Adjust labels
    print(f"\nStep 4: Adjust YOLO labels")
    print(f"  Original label: [class, x_norm, y_norm, w_norm, h_norm]")
    print(f"  Example: [0, 0.500, 0.400, 0.200, 0.150]")
    print(f"\n  Adjustment formulas:")
    print(f"    x_new = ratio × x_old + dw / img_size")
    print(f"    y_new = ratio × y_old + dh / img_size")
    print(f"    w_new = ratio × w_old")
    print(f"    h_new = ratio × h_old")
    
    # Example calculation
    x_old, y_old, w_old, h_old = 0.500, 0.400, 0.200, 0.150
    x_new = ratio * x_old + dw / target_size
    y_new = ratio * y_old + dh / target_size
    w_new = ratio * w_old
    h_new = ratio * h_old
    
    print(f"\n  Example calculation:")
    print(f"    x_new = {ratio:.3f} × {x_old:.3f} + {dw:.1f}/{target_size} = {x_new:.3f}")
    print(f"    y_new = {ratio:.3f} × {y_old:.3f} + {dh:.1f}/{target_size} = {y_new:.3f}")
    print(f"    w_new = {ratio:.3f} × {w_old:.3f} = {w_new:.3f}")
    print(f"    h_new = {ratio:.3f} × {h_old:.3f} = {h_new:.3f}")
    
    print(f"\n  Adjusted label: [0, {x_new:.3f}, {y_new:.3f}, {w_new:.3f}, {h_new:.3f}]")
    
    # Why each component
    print(f"\n" + "=" * 80)
    print("WHY EACH ADJUSTMENT?")
    print("=" * 80)
    
    print(f"\n1. MULTIPLY BY RATIO: {ratio:.3f}")
    print(f"   • Scales coordinates to match resized image")
    print(f"   • Image got smaller by factor of {ratio:.3f}")
    print(f"   • So all coordinates must scale by same factor")
    
    print(f"\n2. ADD PADDING OFFSET: dw={dw:.1f}, dh={dh:.1f}")
    print(f"   • Shifts coordinates to account for gray bars")
    print(f"   • Image is centered in 640×640 canvas")
    print(f"   • Object moved right by {dw:.1f}px, down by {dh:.1f}px")
    print(f"   • In normalized coords: +{dw/target_size:.3f}, +{dh/target_size:.3f}")
    
    print(f"\n3. DIVIDE BY TARGET SIZE: /{target_size}")
    print(f"   • Converts pixel offset to normalized coordinates")
    print(f"   • YOLO expects all coords in range [0, 1]")
    print(f"   • Relative to final 640×640 image")
    
    print("\n" + "=" * 80)


def visualize_different_aspect_ratios():
    """
    Show letterbox for different image shapes
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Letterbox Behavior for Different Aspect Ratios', fontsize=16, fontweight='bold')
    
    examples = [
        ("Wide (16:9)", 1920, 1080, "Horizontal padding"),
        ("Square", 640, 640, "No padding needed"),
        ("Tall (9:16)", 1080, 1920, "Vertical padding"),
        ("Very Wide (21:9)", 2560, 1080, "Large horizontal padding"),
        ("Very Tall (4:5)", 640, 800, "Vertical padding"),
        ("Panorama", 3840, 1080, "Extreme horizontal padding"),
    ]
    
    target_size = 640
    
    for idx, (name, w, h, note) in enumerate(examples):
        ax = axes[idx // 3, idx % 3]
        
        # Calculate letterbox
        ratio = min(target_size / w, target_size / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        
        # Visualize
        img = np.ones((target_size, target_size, 3)) * 0.5  # Gray padding
        img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = 0.9  # White image area
        
        ax.imshow(img, cmap='gray')
        
        # Highlight padding
        if pad_h > 0:
            ax.axhline(y=pad_h, color='red', linestyle='--', linewidth=2)
            ax.axhline(y=target_size-pad_h, color='red', linestyle='--', linewidth=2)
        if pad_w > 0:
            ax.axvline(x=pad_w, color='blue', linestyle='--', linewidth=2)
            ax.axvline(x=target_size-pad_w, color='blue', linestyle='--', linewidth=2)
        
        ax.set_title(f'{name}\n{w}×{h} → {new_w}×{new_h}', fontweight='bold')
        ax.text(target_size/2, target_size-20, note, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=9)
        ax.text(10, 30, f'ratio: {ratio:.3f}\npad: ({pad_w},{pad_h})', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Create visualizations
    print("Creating letterbox visualizations...\n")
    
    # Main concept
    fig1 = visualize_letterbox_concept()
    fig1.savefig('/mnt/user-data/outputs/letterbox_explained.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: letterbox_explained.png")
    
    # Code walkthrough
    code_walkthrough()
    
    # Different aspect ratios
    fig2 = visualize_different_aspect_ratios()
    fig2.savefig('/mnt/user-data/outputs/letterbox_aspect_ratios.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: letterbox_aspect_ratios.png")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Letterbox resizing is ESSENTIAL for YOLO because:

1. Maintains aspect ratio → Objects look natural, not distorted
2. Creates uniform input size → Required for batch processing
3. Adds padding → Centers image in target size
4. Adjusts labels → Keeps bounding boxes accurate

Without label adjustment, boxes would be in wrong positions!

The code in train_yolov5.py does:
  labels[:, 1] = ratio * labels[:, 1] + dw / img_size  # x_center
  labels[:, 2] = ratio * labels[:, 2] + dh / img_size  # y_center  
  labels[:, 3] *= ratio  # width
  labels[:, 4] *= ratio  # height

This ensures labels match the letterboxed image coordinates.
    """)
