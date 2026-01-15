# DermAI - AI Model Documentation
## Skin Lesion Segmentation using Attention U-Net

This document provides a comprehensive overview of the AI/Deep Learning components used in the DermAI skin cancer lesion segmentation application.

---

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Training Pipeline](#training-pipeline)
5. [Loss Functions](#loss-functions)
6. [Data Augmentation](#data-augmentation)
7. [Inference Pipeline](#inference-pipeline)
8. [Performance Metrics](#performance-metrics)
9. [Technical Specifications](#technical-specifications)

---

## Overview

The AI system in DermAI is designed for **semantic segmentation** of skin lesions in dermoscopic images. The goal is to precisely identify and delineate the boundaries of potentially cancerous skin lesions, assisting dermatologists in their diagnostic workflow.

### Key Features
- **Encoder-Decoder Architecture**: U-Net style architecture with skip connections
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet as the encoder
- **Attention Mechanism**: Attention gates to focus on lesion regions
- **Deep Supervision**: Multi-scale loss for better gradient flow
- **Boundary Refinement**: Dedicated head for precise edge detection

---

## Model Architecture

### 1. Encoder: MobileNetV2

The encoder uses **MobileNetV2** pre-trained on ImageNet, which provides:

- **Efficient Feature Extraction**: Inverted residual blocks with linear bottlenecks
- **Depthwise Separable Convolutions**: Reduces computational cost
- **Pre-trained Weights**: Transfer learning from ImageNet for robust feature representations

#### Architecture Details:
```
Input: 256 × 256 × 3 (RGB image)
├── Layer 0-1:   16 channels  (skip connection 1)
├── Layer 2-3:   24 channels  (skip connection 2)
├── Layer 4-6:   32 channels  (skip connection 3)
├── Layer 7-13:  96 channels  (skip connection 4)
└── Layer 14-18: 1280 channels (bottleneck)
```

### 2. Decoder: U-Net Style with Skip Connections

The decoder progressively upsamples the feature maps while integrating skip connections from the encoder:

```
Bottleneck (1280 channels)
    ↓ ConvTranspose2d
Decoder Stage 1 (256 channels) + Skip[4] with Attention Gate
    ↓ ConvTranspose2d
Decoder Stage 2 (128 channels) + Skip[3] with Attention Gate
    ↓ ConvTranspose2d
Decoder Stage 3 (64 channels) + Skip[2] with Attention Gate
    ↓ ConvTranspose2d
Decoder Stage 4 (32 channels) + Skip[1] with Attention Gate
    ↓ ConvTranspose2d
Final Stage (16 channels)
    ↓ Conv2d + Sigmoid
Output: 256 × 256 × 1 (binary mask)
```

### 3. Attention Gates

Attention gates are applied to each skip connection to help the decoder focus on relevant features:

```python
class AttentionGate(nn.Module):
    """
    Args:
        F_g: Channels in gating signal (from decoder)
        F_l: Channels in skip connection (from encoder)
        F_int: Intermediate channels
    
    Operation:
        1. Project gating signal: W_g(g) → features
        2. Project skip connection: W_x(x) → features
        3. Combine: ReLU(W_g + W_x)
        4. Generate attention map: Sigmoid(psi)
        5. Apply attention: x * attention_map
    """
```

The attention mechanism learns to:
- **Highlight lesion regions**: Focus on areas with suspicious features
- **Suppress background**: Ignore hair, skin texture, and other noise
- **Improve boundary detection**: Pay attention to lesion edges

### 4. Boundary Refinement Head

A dedicated convolutional head specifically for refining lesion boundaries:

```
Input: 16 channels (from decoder)
    ↓ Conv2d(16 → 64) + ReLU
    ↓ Conv2d(64 → 64) + ReLU
    ↓ Conv2d(64 → 1) + Sigmoid
Output: Boundary-refined mask
```

The final output is the average of the main segmentation output and the boundary-refined output.

### 5. Deep Supervision (Training Only)

During training, auxiliary outputs are generated at multiple decoder scales:

| Stage | Resolution | Weight |
|-------|------------|--------|
| Decoder 1 | 16×16 | 0.5 |
| Decoder 2 | 32×32 | 0.3 |
| Decoder 3 | 64×64 | 0.2 |
| Decoder 4 | 128×128 | 0.1 |
| Final | 256×256 | 1.0 |

Benefits:
- Better gradient flow to early decoder layers
- Faster convergence
- Improved feature learning at multiple scales

---

## Dataset

### ISIC 2018 Challenge Dataset

The model is trained on the **International Skin Imaging Collaboration (ISIC) 2018** dataset:

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 2595 | Model training |
| Validation | 100 | Hyperparameter tuning |
| Test | 1000 | Final evaluation |

### Data Characteristics
- **Image Type**: Dermoscopic images of skin lesions
- **Resolution**: Variable (resized to 256×256 for training)
- **Annotations**: Binary segmentation masks (pixel-level)
- **Lesion Types**: Melanoma, nevus, seborrheic keratosis, etc.

---

## Training Pipeline

### 1. Training Loop

```python
for epoch in range(EPOCHS):
    # Phase 1: Frozen encoder (first 5 epochs)
    if epoch < ENCODER_FREEZE_EPOCHS:
        # Only train decoder layers
        frozen_layers = 7  # First 7 encoder layers frozen
    else:
        # Phase 2: Full fine-tuning
        unfreeze_encoder()
    
    # Train epoch
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass with deep supervision
        main_output, ds_outputs = model(images)
        
        # Compute combined loss
        loss = deep_supervision_loss(main_output, ds_outputs, masks)
        
        loss.backward()
        optimizer.step()
    
    # Validate and update LR scheduler
    scheduler.step(val_loss)
    
    # Early stopping check
    if no_improvement_for(patience=7):
        break
```

### 2. Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 256×256 | Input resolution |
| Batch Size | 32 | Training batch size |
| Learning Rate | 1e-4 | Initial learning rate |
| Optimizer | Adam | Optimizer algorithm |
| Epochs | 50 | Maximum epochs |
| Early Stopping Patience | 7 | Epochs without improvement |
| Encoder Freeze Epochs | 5 | Epochs with frozen encoder |

### 3. Learning Rate Scheduler

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',      # Reduce when val_loss stops decreasing
    patience=3,      # Wait 3 epochs before reducing
    factor=0.5,      # Reduce LR by half
    min_lr=1e-7      # Minimum learning rate
)
```

---

## Loss Functions

### Combined Loss Function

The model is trained using a weighted combination of three loss functions:

```
Total Loss = 0.3 × BCE + 0.4 × Dice + 0.3 × Tversky
```

### 1. Binary Cross-Entropy (BCE) Loss

Measures pixel-wise classification accuracy:

```
BCE = -1/N Σ [y·log(p) + (1-y)·log(1-p)]
```

**Purpose**: Ensures accurate per-pixel predictions

### 2. Dice Loss

Measures overlap between prediction and ground truth:

```
Dice = 2·|P ∩ G| / (|P| + |G|)
Dice Loss = 1 - Dice
```

**Purpose**: Handles class imbalance, optimizes for segmentation quality

### 3. Tversky Loss

A generalization of Dice loss with asymmetric penalization:

```
Tversky = TP / (TP + α·FN + β·FP)
Tversky Loss = 1 - Tversky
```

Configuration:
- **α = 0.7**: Higher penalty for False Negatives (missing lesion parts)
- **β = 0.3**: Lower penalty for False Positives

**Purpose**: Addresses class imbalance by penalizing missed lesions more than false alarms

### Deep Supervision Loss

For multi-scale outputs during training:

```
Total_DS = L_main + Σ(wi × L_i)

Where:
- L_main: Loss on final output (weight = 1.0)
- L_i: Loss on auxiliary output i
- wi: Weight for scale i (0.5, 0.3, 0.2, 0.1)
```

---

## Data Augmentation

The training pipeline uses **Albumentations** for advanced data augmentation:

### Geometric Augmentations

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| HorizontalFlip | - | 0.5 |
| VerticalFlip | - | 0.5 |
| RandomRotate90 | - | 0.5 |
| Affine | translate=±10%, scale=0.8-1.2, rotate=±45° | 0.5 |
| ElasticTransform | alpha=120, sigma=6 | 0.3 |

**Rationale**: Skin lesions can appear at any orientation; dermoscopy captures images from various angles.

### Color Augmentations

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 | 0.5 |
| RandomBrightnessContrast | limit=0.2 | 0.5 |
| HueSaturationValue | hue=±10, sat=±20, val=±10 | 0.3 |

**Rationale**: Handles variations in camera settings, lighting conditions, and skin tones.

### Noise & Blur Augmentations

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| GaussianBlur | blur_limit=3-5 | 0.2 |
| GaussNoise | std=0.02-0.1 | 0.2 |

**Rationale**: Simulates image quality variations and camera noise.

### Cutout/Dropout

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| CoarseDropout | 1-8 holes, size=16-32 pixels | 0.5 |

**Rationale**: Forces the model to learn from partial information, improving robustness to occlusions (hair, rulers, artifacts).

### Normalization

All images are normalized using ImageNet statistics:
```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

---

## Inference Pipeline

### Preprocessing

```python
def preprocess_image(image):
    # 1. Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to 256×256
    image_resized = cv2.resize(image_rgb, (256, 256))
    
    # 3. Normalize with ImageNet stats
    image_normalized = (image_resized / 255.0 - mean) / std
    
    # 4. Convert to tensor [1, 3, 256, 256]
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.float()
```

### Inference

```python
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)  # [1, 1, 256, 256]
        mask = output.cpu().numpy().squeeze()  # [256, 256]
    return mask
```

### Postprocessing

```python
def postprocess_mask(mask, original_size):
    # 1. Apply threshold
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # 2. Resize to original image size
    mask_resized = cv2.resize(binary_mask, 
                               (original_size[1], original_size[0]),
                               interpolation=cv2.INTER_LINEAR)
    
    return mask_resized
```

### Metrics Calculation

```python
# Confidence Score: Mean probability of lesion pixels
confidence = np.mean(mask[mask > 0.5]) if np.any(mask > 0.5) else 0.0

# Lesion Area: Percentage of image covered by lesion
lesion_area = np.mean(mask > 0.5) * 100
```

---

## Performance Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Dice Coefficient** | Overlap between prediction and ground truth | > 0.85 |
| **Accuracy** | Per-pixel classification accuracy | > 0.95 |
| **IoU (Jaccard)** | Intersection over Union | > 0.80 |

### Dice Coefficient Calculation

```python
def dice_coefficient(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum())
    
    return dice.item()
```

### Expected Performance

### Actual Training Results (Epoch 39)

| Split | Dice | Accuracy | Loss |
|-------|------|----------|------|
| Validation | 0.8864 | 0.9405 | 0.1470 |

*Model achieved best validation loss at Epoch 39.*

---

## Technical Specifications

### Model Size

| Component | Parameters |
|-----------|------------|
| MobileNetV2 Encoder | ~3.4M |
| Attention Gates | ~0.2M |
| Decoder | ~2.0M |
| Boundary Head | ~0.1M |
| **Total Parameters** | **5,637,018** |
| **Trainable Parameters** | **5,581,530** |

### Computational Requirements

| Metric | Value |
|--------|-------|
| Input Size | 256 × 256 × 3 |
| FLOPs | ~1.2 GFLOPs |
| GPU Memory (Inference) | ~500 MB |
| Inference Time (GPU) | ~15 ms |
| Inference Time (CPU) | ~200 ms |

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Pre-trained models |
| Albumentations | ≥1.3.0 | Data augmentation |
| OpenCV | ≥4.8.0 | Image processing |
| NumPy | ≥1.24.0 | Numerical operations |

---

## Model Files

### Required Files

```
models/
└── model.pth    # Trained model weights (~23 MB)
```

### Loading the Model

```python
from server.main import AttentionUNetMobileNet

# Initialize model
model = AttentionUNetMobileNet(deep_supervision=False)

# Load trained weights
model.load_state_dict(
    torch.load('models/model.pth', map_location='cpu'),
    strict=False
)

# Set to evaluation mode
model.eval()
```

---

## Future Improvements

1. **Multi-class Segmentation**: Extend to classify lesion types
2. **Ensemble Models**: Combine multiple architectures for robustness
3. **Test-Time Augmentation**: Average predictions over augmented inputs
4. **Model Quantization**: Reduce model size for mobile deployment
5. **Uncertainty Estimation**: Provide confidence intervals for predictions

---

## References

1. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. **Attention U-Net**: Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
3. **MobileNetV2**: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
4. **ISIC 2018**: Codella, N., et al. (2018). Skin Lesion Analysis Toward Melanoma Detection 2018.
5. **Tversky Loss**: Salehi, S.S.M., et al. (2017). Tversky Loss Function for Image Segmentation.

---

## License

This AI model is for **research and educational purposes only**. It should not be used as a substitute for professional medical diagnosis.

---

