"""
Skin Cancer Lesion Segmentation - FastAPI Backend
================================================
This API provides endpoints for skin lesion segmentation using an
Attention U-Net model with MobileNetV2 encoder.

Features:
- Real-time segmentation of dermoscopic images
- Confidence score calculation
- Lesion area analysis
- Support for multiple image formats (JPEG, PNG, WebP)
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================
# CONFIGURATION
# =============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pth")
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================
# MODEL ARCHITECTURE (must match training)
# =============================================

class AttentionGate(nn.Module):
    """Attention U-Net gate for skip connections."""
    def __init__(self, F_g, F_l, F_int):
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
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class BoundaryRefinementHead(nn.Module):
    """Refines segmentation boundaries."""
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


class AttentionUNetMobileNet(nn.Module):
    """
    Attention U-Net with MobileNetV2 encoder for skin lesion segmentation.
    """
    def __init__(self, deep_supervision=False):
        super(AttentionUNetMobileNet, self).__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder (MobileNetV2 - pretrained)
        mobilenet = mobilenet_v2(weights='DEFAULT')
        self.encoder_features = mobilenet.features
        
        # Attention Gates
        self.att1 = AttentionGate(F_g=256, F_l=96, F_int=64)
        self.att2 = AttentionGate(F_g=128, F_l=32, F_int=32)
        self.att3 = AttentionGate(F_g=64, F_l=24, F_int=16)
        self.att4 = AttentionGate(F_g=32, F_l=16, F_int=8)
        
        # Decoder
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
        
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec5_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dec5_bn1 = nn.BatchNorm2d(16)
        self.dec5_relu1 = nn.ReLU(inplace=True)
        self.dec5_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dec5_bn2 = nn.BatchNorm2d(16)
        self.dec5_relu2 = nn.ReLU(inplace=True)
        
        # Main output
        self.seg_output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Boundary head
        self.boundary_head = BoundaryRefinementHead(16, filters=64)
        
        # Deep supervision outputs (only used during training)
        if self.deep_supervision:
            self.ds_out1 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())
            self.ds_out2 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.Sigmoid())
            self.ds_out3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())
            self.ds_out4 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
        
    def forward(self, x):
        # Encoder
        skips = []
        for idx, layer in enumerate(self.encoder_features):
            x = layer(x)
            if idx in [1, 3, 6, 13, 18]:
                skips.append(x)
        
        # Decoder Stage 1
        d1 = self.up1(x)
        skip1_att = self.att1(d1, skips[3])
        d1 = torch.cat([d1, skip1_att], dim=1)
        d1 = self.dec1_relu1(self.dec1_bn1(self.dec1_conv1(d1)))
        d1 = self.dec1_relu2(self.dec1_bn2(self.dec1_conv2(d1)))
        
        # Decoder Stage 2
        d2 = self.up2(d1)
        skip2_att = self.att2(d2, skips[2])
        d2 = torch.cat([d2, skip2_att], dim=1)
        d2 = self.dec2_relu1(self.dec2_bn1(self.dec2_conv1(d2)))
        d2 = self.dec2_relu2(self.dec2_bn2(self.dec2_conv2(d2)))
        
        # Decoder Stage 3
        d3 = self.up3(d2)
        skip3_att = self.att3(d3, skips[1])
        d3 = torch.cat([d3, skip3_att], dim=1)
        d3 = self.dec3_relu1(self.dec3_bn1(self.dec3_conv1(d3)))
        d3 = self.dec3_relu2(self.dec3_bn2(self.dec3_conv2(d3)))
        
        # Decoder Stage 4
        d4 = self.up4(d3)
        skip4_att = self.att4(d4, skips[0])
        d4 = torch.cat([d4, skip4_att], dim=1)
        d4 = self.dec4_relu1(self.dec4_bn1(self.dec4_conv1(d4)))
        d4 = self.dec4_relu2(self.dec4_bn2(self.dec4_conv2(d4)))
        
        # Final upsampling
        d5 = self.up5(d4)
        d5 = self.dec5_relu1(self.dec5_bn1(self.dec5_conv1(d5)))
        d5 = self.dec5_relu2(self.dec5_bn2(self.dec5_conv2(d5)))
        
        # Main output
        seg_output = self.seg_output(d5)
        boundary_output = self.boundary_head(d5)
        final_output = (seg_output + boundary_output) / 2
        
        return final_output


# =============================================
# PREPROCESSING
# =============================================

def get_inference_transforms(img_size=IMG_SIZE):
    """Transform pipeline for inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model inference."""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transform = get_inference_transforms()
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0)
    
    return img_tensor


def postprocess_mask(mask: np.ndarray, original_size: tuple) -> np.ndarray:
    """Postprocess model output to original image size."""
    # Resize to original size
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), 
                               interpolation=cv2.INTER_LINEAR)
    return mask_resized


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert numpy mask to base64 string."""
    # Normalize to 0-255
    mask_uint8 = (mask * 255).astype(np.uint8)
    # Encode as PNG
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of mask on original image."""
    # Create colored mask (magenta for lesion)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = int(255 * 0.94)  # R
    colored_mask[:, :, 1] = int(255 * 0.38)  # G
    colored_mask[:, :, 2] = int(255 * 0.57)  # B
    
    # Apply mask
    mask_3ch = np.stack([mask] * 3, axis=-1)
    overlay = image * (1 - mask_3ch * alpha) + colored_mask * (mask_3ch * alpha)
    
    return overlay.astype(np.uint8)


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


# =============================================
# FASTAPI APP
# =============================================

app = FastAPI(
    title="Skin Lesion Segmentation API",
    description="AI-powered skin cancer lesion segmentation using Attention U-Net",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Using device: {DEVICE}")
    
    try:
        model = AttentionUNetMobileNet(deep_supervision=False).to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            print("✓ Model loaded successfully!")
        else:
            print(f"⚠ Model file not found at {MODEL_PATH}")
            print("  The API will work but predictions will use untrained weights.")
        
        model.eval()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = AttentionUNetMobileNet(deep_supervision=False).to(DEVICE)
        model.eval()


class SegmentationResponse(BaseModel):
    """Response model for segmentation endpoint."""
    success: bool
    mask_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    confidence: Optional[float] = None
    lesion_area_percent: Optional[float] = None
    message: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Redirect to frontend."""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/static/index.html" />
        </head>
        <body>
            <p>Redirecting to the application...</p>
        </body>
    </html>
    """


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


@app.post("/api/segment", response_model=SegmentationResponse)
async def segment_image(file: UploadFile = File(...)):
    """
    Segment a skin lesion from an uploaded dermoscopic image.
    
    Args:
        file: Image file (JPEG, PNG, WebP)
    
    Returns:
        SegmentationResponse with mask and overlay in base64
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        return SegmentationResponse(
            success=False,
            message=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        
        # Preprocess
        img_tensor = preprocess_image(image_rgb).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            mask = output.cpu().numpy().squeeze()
        
        # Calculate metrics
        confidence = float(np.mean(mask[mask > 0.5])) if np.any(mask > 0.5) else 0.0
        lesion_area = float(np.mean(mask > 0.5) * 100)
        
        # Threshold mask
        mask_binary = (mask > 0.5).astype(np.float32)
        
        # Resize to original size
        mask_resized = postprocess_mask(mask_binary, original_size)
        
        # Create overlay
        overlay = create_overlay(image_rgb, mask_resized, alpha=0.4)
        
        return SegmentationResponse(
            success=True,
            mask_base64=mask_to_base64(mask_resized),
            overlay_base64=image_to_base64(overlay),
            confidence=round(confidence * 100, 2),
            lesion_area_percent=round(lesion_area, 2)
        )
        
    except Exception as e:
        return SegmentationResponse(
            success=False,
            message=f"Error processing image: {str(e)}"
        )


# Mount static files (frontend)
client_path = os.path.join(os.path.dirname(__file__), "..", "client")
if os.path.exists(client_path):
    app.mount("/static", StaticFiles(directory=client_path), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
