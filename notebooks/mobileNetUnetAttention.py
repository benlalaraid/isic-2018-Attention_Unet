import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
TRAIN_IMG_PATH = '/kaggle/input/isic-2018/data/images/train'
TRAIN_MASK_PATH = '/kaggle/input/isic-2018/data/annotations/train'
VAL_IMG_PATH = '/kaggle/input/isic-2018/data/images/val'
VAL_MASK_PATH = '/kaggle/input/isic-2018/data/annotations/val'
TEST_IMG_PATH = '/kaggle/input/isic-2018/data/images/test'
TEST_MASK_PATH = '/kaggle/input/isic-2018/data/annotations/test'

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
ENCODER_FREEZE_EPOCHS = 5  # Number of epochs to freeze encoder


# =============================================
# 1. ADVANCED DATA AUGMENTATION (Albumentations)
# =============================================

def get_train_transforms(img_size=IMG_SIZE):
    """
    Advanced augmentation pipeline for training:
    - Horizontal/Vertical Flips: Lesions are orientation-independent
    - Random Rotation: Skin images can be taken at any angle
    - Color Jitter: To handle different camera lighting and skin tones
    - CoarseDropout (Cutout): Forces model to look at edges if center is obscured
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Using Affine instead of ShiftScaleRotate (recommended in Albumentations v2.x)
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.8, 1.2),
            rotate=(-45, 45),
            border_mode=cv2.BORDER_REFLECT,  # Use border_mode instead of mode
            p=0.5
        ),
        # Elastic deformation for shape variation
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        # Color augmentations
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        # Blur and noise augmentations
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),  # Updated for Albumentations v2.x
        # CoarseDropout (Cutout): Forces the model to look at edges (Albumentations v2.x API)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 16, img_size // 8),
            hole_width_range=(img_size // 16, img_size // 8),
            fill=0.0,       # Fill with black (numeric value, not string)
            fill_mask=0.0,  # Fill mask with 0 (background)
            p=0.5
        ),
        # Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    """Minimal transforms for validation/test (only resize and normalize)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Custom Dataset with Albumentations
class ISICDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, img_size=IMG_SIZE):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize mask to [0, 1]
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            # Fallback if no transform (shouldn't happen)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img / 255.0
            img = torch.FloatTensor(np.transpose(img, (2, 0, 1)))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = torch.FloatTensor(mask)
        
        # Ensure mask has channel dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return img.float(), mask.float()


def get_matching_files(img_dir, mask_dir):
    """Find matching image and mask pairs by filename"""
    img_files = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for path in glob(os.path.join(img_dir, ext)):
            basename = os.path.splitext(os.path.basename(path))[0]
            img_files[basename] = path
    
    mask_files = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for path in glob(os.path.join(mask_dir, ext)):
            basename = os.path.splitext(os.path.basename(path))[0]
            basename = basename.replace('_segmentation', '').replace('_mask', '')
            mask_files[basename] = path
    
    common = set(img_files.keys()) & set(mask_files.keys())
    
    img_paths = [img_files[k] for k in sorted(common)]
    mask_paths = [mask_files[k] for k in sorted(common)]
    
    return img_paths, mask_paths


# =============================================
# 3. ARCHITECTURE REFINEMENTS - Attention Gate
# =============================================

class AttentionGate(nn.Module):
    """
    Attention U-Net gate for skip connections.
    Helps the decoder focus on the lesion and ignore background noise/hairs.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in gating signal (from decoder)
            F_l: Number of channels in skip connection (from encoder)
            F_int: Number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (coarse features)
            x: Skip connection from encoder (fine features)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1's spatial size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


# Boundary Refinement Head
class BoundaryRefinementHead(nn.Module):
    def __init__(self, in_channels, filters=64):
        super(BoundaryRefinementHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(filters, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x


# =============================================
# 3. ARCHITECTURE - Attention U-Net with Deep Supervision
# =============================================

class AttentionUNetMobileNet(nn.Module):
    """
    Enhanced U-Net with:
    - MobileNetV2 encoder (pre-trained)
    - Attention Gates on skip connections
    - Deep Supervision (multi-scale loss)
    - Encoder freezing support for first N epochs
    """
    def __init__(self, deep_supervision=True):
        super(AttentionUNetMobileNet, self).__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder (MobileNetV2 - pretrained)
        mobilenet = mobilenet_v2(weights='DEFAULT')
        self.encoder_features = mobilenet.features
        
        # Attention Gates for skip connections
        self.att1 = AttentionGate(F_g=256, F_l=96, F_int=64)
        self.att2 = AttentionGate(F_g=128, F_l=32, F_int=32)
        self.att3 = AttentionGate(F_g=64, F_l=24, F_int=16)
        self.att4 = AttentionGate(F_g=32, F_l=16, F_int=8)
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(256 + 96, 256, kernel_size=3, padding=1)
        self.dec1_bn1 = nn.BatchNorm2d(256)
        self.dec1_relu1 = nn.ReLU(inplace=True)
        self.dec1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec1_bn2 = nn.BatchNorm2d(256)
        self.dec1_relu2 = nn.ReLU(inplace=True)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(128 + 32, 128, kernel_size=3, padding=1)
        self.dec2_bn1 = nn.BatchNorm2d(128)
        self.dec2_relu1 = nn.ReLU(inplace=True)
        self.dec2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec2_bn2 = nn.BatchNorm2d(128)
        self.dec2_relu2 = nn.ReLU(inplace=True)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(64 + 24, 64, kernel_size=3, padding=1)
        self.dec3_bn1 = nn.BatchNorm2d(64)
        self.dec3_relu1 = nn.ReLU(inplace=True)
        self.dec3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec3_bn2 = nn.BatchNorm2d(64)
        self.dec3_relu2 = nn.ReLU(inplace=True)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv2d(32 + 16, 32, kernel_size=3, padding=1)
        self.dec4_bn1 = nn.BatchNorm2d(32)
        self.dec4_relu1 = nn.ReLU(inplace=True)
        self.dec4_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec4_bn2 = nn.BatchNorm2d(32)
        self.dec4_relu2 = nn.ReLU(inplace=True)
        
        # Additional upsampling to reach 256x256
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec5_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dec5_bn1 = nn.BatchNorm2d(16)
        self.dec5_relu1 = nn.ReLU(inplace=True)
        self.dec5_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dec5_bn2 = nn.BatchNorm2d(16)
        self.dec5_relu2 = nn.ReLU(inplace=True)
        
        # Main segmentation output
        self.seg_output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Boundary refinement head
        self.boundary_head = BoundaryRefinementHead(16, filters=64)
        
        # Deep supervision outputs (at different scales)
        if self.deep_supervision:
            self.ds_out1 = nn.Sequential(
                nn.Conv2d(256, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.ds_out2 = nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.ds_out3 = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.ds_out4 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
    
    def freeze_encoder(self, num_layers=7):
        """Freeze the first N layers of the encoder to preserve pretrained features"""
        for idx, layer in enumerate(self.encoder_features):
            if idx < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Froze first {num_layers} encoder layers")
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder layers for fine-tuning"""
        for layer in self.encoder_features:
            for param in layer.parameters():
                param.requires_grad = True
        print("Unfroze all encoder layers")
        
    def forward(self, x):
        # Encoder with skip connections
        skips = []
        for idx, layer in enumerate(self.encoder_features):
            x = layer(x)
            if idx in [1, 3, 6, 13, 18]:  # Save skip connections
                skips.append(x)
        
        ds_outputs = []  # Deep supervision outputs
        
        # Decoder Stage 1
        d1 = self.up1(x)
        skip1_att = self.att1(d1, skips[3])  # Apply attention
        d1 = torch.cat([d1, skip1_att], dim=1)
        d1 = self.dec1_relu1(self.dec1_bn1(self.dec1_conv1(d1)))
        d1 = self.dec1_relu2(self.dec1_bn2(self.dec1_conv2(d1)))
        if self.deep_supervision and self.training:
            ds_outputs.append(self.ds_out1(d1))
        
        # Decoder Stage 2
        d2 = self.up2(d1)
        skip2_att = self.att2(d2, skips[2])
        d2 = torch.cat([d2, skip2_att], dim=1)
        d2 = self.dec2_relu1(self.dec2_bn1(self.dec2_conv1(d2)))
        d2 = self.dec2_relu2(self.dec2_bn2(self.dec2_conv2(d2)))
        if self.deep_supervision and self.training:
            ds_outputs.append(self.ds_out2(d2))
        
        # Decoder Stage 3
        d3 = self.up3(d2)
        skip3_att = self.att3(d3, skips[1])
        d3 = torch.cat([d3, skip3_att], dim=1)
        d3 = self.dec3_relu1(self.dec3_bn1(self.dec3_conv1(d3)))
        d3 = self.dec3_relu2(self.dec3_bn2(self.dec3_conv2(d3)))
        if self.deep_supervision and self.training:
            ds_outputs.append(self.ds_out3(d3))
        
        # Decoder Stage 4
        d4 = self.up4(d3)
        skip4_att = self.att4(d4, skips[0])
        d4 = torch.cat([d4, skip4_att], dim=1)
        d4 = self.dec4_relu1(self.dec4_bn1(self.dec4_conv1(d4)))
        d4 = self.dec4_relu2(self.dec4_bn2(self.dec4_conv2(d4)))
        if self.deep_supervision and self.training:
            ds_outputs.append(self.ds_out4(d4))
        
        # Final upsampling to 256x256
        d5 = self.up5(d4)
        d5 = self.dec5_relu1(self.dec5_bn1(self.dec5_conv1(d5)))
        d5 = self.dec5_relu2(self.dec5_bn2(self.dec5_conv2(d5)))
        
        # Main output
        seg_output = self.seg_output(d5)
        
        # Boundary refinement
        boundary_output = self.boundary_head(d5)
        
        # Average outputs
        final_output = (seg_output + boundary_output) / 2
        
        if self.deep_supervision and self.training:
            return final_output, ds_outputs
        return final_output


# =============================================
# 4. LOSS FUNCTION TUNING - Tversky Loss + BCE/Dice weighting
# =============================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss: A generalization of Dice loss that allows you to 
    penalize False Negatives (missing part of a lesion) more than False Positives.
    
    Useful for class imbalance where lesion is much smaller than background.
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-7):
        """
        Args:
            alpha: Weight for False Negatives (higher = penalize FN more)
            beta: Weight for False Positives
            smooth: Smoothing factor to avoid division by zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Negatives, False Positives
        TP = (pred * target).sum()
        FN = (target * (1 - pred)).sum()
        FP = ((1 - target) * pred).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        return 1 - tversky


# Dice Coefficient metric
def dice_coefficient(pred, target, threshold=0.5, smooth=1e-7):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()


# Combined Loss with Tversky + BCE + Dice
class CombinedLoss(nn.Module):
    """
    Enhanced loss function combining:
    - BCE Loss (pixel-wise accuracy)
    - Dice Loss (overlap quality)  
    - Tversky Loss (class imbalance handling)
    
    Weights: 0.3 BCE / 0.4 Dice / 0.3 Tversky (prioritizing Dice for overlap quality)
    """
    def __init__(self, bce_weight=0.3, dice_weight=0.4, tversky_weight=0.3, 
                 tversky_alpha=0.7, tversky_beta=0.3):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        tversky_loss = self.tversky(pred, target)
        return (self.bce_weight * bce_loss + 
                self.dice_weight * dice_loss + 
                self.tversky_weight * tversky_loss)


class DeepSupervisionLoss(nn.Module):
    """
    Loss function for deep supervision with multi-scale outputs.
    Calculates loss at different decoder scales and averages them.
    """
    def __init__(self, base_criterion, weights=None):
        super(DeepSupervisionLoss, self).__init__()
        self.base_criterion = base_criterion
        # Weights for each deep supervision output (shallower = less weight)
        self.weights = weights or [0.5, 0.3, 0.2, 0.1]
        
    def forward(self, main_output, ds_outputs, target):
        # Main output loss (full weight)
        loss = self.base_criterion(main_output, target)
        
        # Deep supervision losses at different scales
        for i, ds_out in enumerate(ds_outputs):
            # Upsample deep supervision output to target size
            ds_upsampled = nn.functional.interpolate(
                ds_out, size=target.shape[2:], mode='bilinear', align_corners=True
            )
            loss += self.weights[i] * self.base_criterion(ds_upsampled, target)
        
        return loss


# =============================================
# TRAINING AND VALIDATION FUNCTIONS
# =============================================

def train_epoch(model, dataloader, criterion, optimizer, device, deep_supervision=True):
    """Training function with deep supervision support"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if deep_supervision:
            outputs, ds_outputs = model(images)
            loss = criterion(outputs, ds_outputs, masks)
        else:
            outputs = model(images)
            loss = criterion.base_criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        # Metrics (use main output)
        running_loss += loss.item()
        acc = ((outputs > 0.5).float() == masks).float().mean()
        running_acc += acc.item()
        running_dice += dice_coefficient(outputs, masks)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validation function (no deep supervision during eval)"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    
    # Get base criterion for validation (no deep supervision)
    if hasattr(criterion, 'base_criterion'):
        val_criterion = criterion.base_criterion
    else:
        val_criterion = criterion
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)  # No deep supervision during eval
            loss = val_criterion(outputs, masks)
            
            running_loss += loss.item()
            acc = ((outputs > 0.5).float() == masks).float().mean()
            running_acc += acc.item()
            running_dice += dice_coefficient(outputs, masks)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_dice


# =============================================
# MAIN EXECUTION
# =============================================

# Load datasets
print("Loading training data...")
train_img_paths, train_mask_paths = get_matching_files(TRAIN_IMG_PATH, TRAIN_MASK_PATH)
print(f"Found {len(train_img_paths)} training pairs")

print("\nLoading validation data...")
val_img_paths, val_mask_paths = get_matching_files(VAL_IMG_PATH, VAL_MASK_PATH)
print(f"Found {len(val_img_paths)} validation pairs")

print("\nLoading test data...")
test_img_paths, test_mask_paths = get_matching_files(TEST_IMG_PATH, TEST_MASK_PATH)
print(f"Found {len(test_img_paths)} test pairs")

# Create datasets with augmentation transforms
train_dataset = ISICDataset(train_img_paths, train_mask_paths, transform=get_train_transforms())
val_dataset = ISICDataset(val_img_paths, val_mask_paths, transform=get_val_transforms())
test_dataset = ISICDataset(test_img_paths, test_mask_paths, transform=get_val_transforms())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Build model with Attention U-Net and Deep Supervision
print("\nBuilding Attention U-Net model with Deep Supervision...")
model = AttentionUNetMobileNet(deep_supervision=True).to(device)

# Freeze encoder for first N epochs
model.freeze_encoder(num_layers=7)

# Loss function with improved weighting (0.3 BCE / 0.4 Dice / 0.3 Tversky)
base_criterion = CombinedLoss(
    bce_weight=0.3, 
    dice_weight=0.4, 
    tversky_weight=0.3,
    tversky_alpha=0.7,  # Higher penalty for False Negatives
    tversky_beta=0.3     # Lower penalty for False Positives
)
criterion = DeepSupervisionLoss(base_criterion)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',          # Reduce LR when val_loss stops decreasing
    patience=3,          # Wait 3 epochs before reducing
    factor=0.5,          # Reduce LR by half
    min_lr=1e-7          # Minimum LR
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model built. Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Training loop with early stopping, LR scheduling, and encoder unfreezing
print("\nStarting training...")
print(f"Encoder will be unfrozen after epoch {ENCODER_FREEZE_EPOCHS}")
history = {
    'train_loss': [], 'train_acc': [], 'train_dice': [],
    'val_loss': [], 'val_acc': [], 'val_dice': [],
    'learning_rate': []
}

best_val_loss = float('inf')
patience = 7  # Increased patience due to LR scheduling
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Unfreeze encoder after N epochs (for fine-tuning)
    if epoch == ENCODER_FREEZE_EPOCHS:
        model.unfreeze_encoder()
        print(f"Starting fine-tuning of encoder layers")
    
    # Log current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    history['learning_rate'].append(current_lr)
    print(f"Current LR: {current_lr:.2e}")
    
    # Train
    train_loss, train_acc, train_dice = train_epoch(
        model, train_loader, criterion, optimizer, device, deep_supervision=True
    )
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_dice'].append(train_dice)
    
    # Validate
    val_loss, val_acc, val_dice = validate_epoch(model, val_loader, criterion, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_dice'].append(val_dice)
    
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Dice: {train_dice:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Dice: {val_dice:.4f}")
    
    # Step the learning rate scheduler based on validation loss
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("✓ Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Model Loss (BCE + Dice + Tversky)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Dice Coefficient
axes[1, 0].plot(history['train_dice'], label='Train Dice')
axes[1, 0].plot(history['val_dice'], label='Val Dice')
axes[1, 0].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Dice')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='orange')
axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc, test_dice = validate_epoch(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Dice Coefficient: {test_dice:.4f}")

# Visualize 3 test predictions
print("\nGenerating predictions...")
model.eval()
test_samples = []
for i in range(3):
    img, mask = test_dataset[i]
    test_samples.append((img, mask))

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Denormalization for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

with torch.no_grad():
    for i, (img, mask) in enumerate(test_samples):
        img_input = img.unsqueeze(0).to(device)
        pred = model(img_input).cpu().squeeze()
        
        # Denormalize image for display
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_display = img_denorm.permute(1, 2, 0).numpy()
        mask_display = mask.squeeze().numpy()
        pred_display = pred.numpy()
        
        # Original image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title('Original Image', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(mask_display, cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(pred_display, cmap='gray')
        axes[i, 2].set_title('Prediction', fontweight='bold')
        axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Training completed successfully!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("\n=== ENHANCEMENTS APPLIED ===")
print("1. ✓ Advanced Data Augmentation (Albumentations)")
print("   - HorizontalFlip, VerticalFlip, RandomRotate90")
print("   - ShiftScaleRotate, ElasticTransform")
print("   - ColorJitter, RandomBrightnessContrast, HueSaturationValue")
print("   - GaussianBlur, GaussNoise")
print("   - CoarseDropout (Cutout)")
print("2. ✓ Learning Rate Scheduler (ReduceLROnPlateau)")
print("   - Patience: 3, Factor: 0.5, Min LR: 1e-7")
print("3. ✓ Architecture Refinements")
print("   - Attention Gates on skip connections")
print("   - Deep Supervision at multiple scales")
print(f"   - Encoder freezing for first {ENCODER_FREEZE_EPOCHS} epochs")
print("4. ✓ Loss Function Tuning")
print("   - Combined: 0.3 BCE + 0.4 Dice + 0.3 Tversky")
print("   - Tversky: alpha=0.7 (penalize FN), beta=0.3 (FP)")