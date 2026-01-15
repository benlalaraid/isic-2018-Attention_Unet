# DermAI - Web Application
## Skin Lesion Segmentation System

A modern web application for AI-powered skin lesion segmentation using Attention U-Net deep learning model.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone/Navigate to the project:**
   ```bash
   cd "/home/raid/Desktop/isic2018 skin cancer app"
   ```

2. **Activate the virtual environment:**
   ```bash
   source ~/cv-env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r server/requirements.txt
   ```

4. **Add your trained model:**
   ```bash
   # Copy your trained model to:
   models/model.pth
   ```

5. **Run the server:**
   ```bash
   cd server
   python main.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Open in browser:**
   ```
   http://localhost:8000
   ```

---

## 📁 Project Structure

```
isic2018-skin-cancer-app/
├── client/                     # Frontend (HTML/CSS/JS)
│   ├── index.html              # Main HTML page
│   ├── styles.css              # CSS styles (dark medical theme)
│   └── app.js                  # JavaScript (file upload, API calls)
│
├── server/                     # Backend (FastAPI)
│   ├── main.py                 # FastAPI application & model
│   └── requirements.txt        # Python dependencies
│
├── models/                     # Trained model weights
│   └── best_model.pth          # (add your trained model here)
│
├── notebooks/                  # Training notebooks
│   └── mobileNetUnetAttention.py
│
├── README.md                   # This file
└── README_AI.md               # AI/Model documentation
```

---

## 🖥️ Features

### Frontend
- **Modern Medical Theme**: Dark mode with purple/pink gradient accents
- **Drag & Drop Upload**: Easy image upload with drag-and-drop support
- **Real-time Results**: Instant visualization of segmentation results
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Download Results**: Export combined analysis as PNG image

### Backend (API)
- **FastAPI Framework**: High-performance async Python server
- **CORS Enabled**: Cross-origin requests supported
- **Health Check**: API status monitoring endpoint
- **Image Validation**: Supports JPEG, PNG, WebP formats

---

## 🔌 API Endpoints

### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Segmentation
```http
POST /api/segment
Content-Type: multipart/form-data

file: <image_file>
```
**Response:**
```json
{
  "success": true,
  "mask_base64": "iVBORw0KGgo...",
  "overlay_base64": "iVBORw0KGgo...",
  "confidence": 85.5,
  "lesion_area_percent": 12.3
}
```

---

## 🎨 Design System

### Color Palette

| Color | HSL | Usage |
|-------|-----|-------|
| Primary Purple | `hsl(250, 89%, 65%)` | Buttons, accents |
| Accent Pink | `hsl(330, 81%, 60%)` | Highlights, gradients |
| Success Green | `hsl(142, 71%, 45%)` | Positive indicators |
| Background Dark | `hsl(240, 20%, 4%)` | Main background |

### Typography
- **Primary Font**: Inter (Google Fonts)
- **Monospace Font**: JetBrains Mono
- **Headings**: 700-800 weight
- **Body**: 400-500 weight

### Effects
- **Glassmorphism**: Blur + transparency on cards
- **Gradient Orbs**: Animated background blobs
- **Smooth Transitions**: 250ms ease animations
- **Hover States**: Lift + glow effects

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + O` | Open file dialog |
| `Ctrl/Cmd + S` | Download results |
| `Escape` | Reset analysis |

---

## 🛠️ Configuration

### Server Configuration (server/main.py)

```python
# Model path
MODEL_PATH = "../models/best_model.pth"

# Image size (must match training)
IMG_SIZE = 256

# Device (auto-detected)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Running on Different Port

```bash
uvicorn server.main:app --port 3000
```

### Production Mode

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🧪 Testing

### API Test with cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Segmentation
curl -X POST http://localhost:8000/api/segment \
  -F "file=@test_image.jpg"
```

### API Test with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/api/health")
print(response.json())

# Segmentation
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/segment",
        files={"file": f}
    )
print(response.json())
```

---

## 📦 Dependencies

### Python (Backend)
```
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=10.0.0
albumentations>=1.3.0
numpy>=1.24.0
pydantic>=2.0.0
```

### Frontend
- Vanilla HTML5
- Vanilla CSS3
- Vanilla JavaScript (ES6+)
- Google Fonts (Inter, JetBrains Mono)

---

## 🔒 Security Notes

- All image processing is done locally (no external API calls)
- Images are processed in memory and not stored
- CORS is enabled for development (restrict in production)

---

## 🐛 Troubleshooting

### Model not found
```
⚠ Model file not found at models/best_model.pth
```
**Solution**: Add your trained model to the `models/` directory.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU:
```python
DEVICE = torch.device('cpu')
```

### Port already in use
```
OSError: [Errno 98] Address already in use
```
**Solution**: Kill the existing process or use a different port:
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

### Static files not serving
**Solution**: Ensure the client directory exists and contains index.html:
```bash
ls -la client/
```

---

## 📄 License

This project is for **research and educational purposes only**. 

⚠️ **Medical Disclaimer**: This tool should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues and questions, please open a GitHub issue.

---

*Version 1.0.0 | Built with FastAPI, PyTorch & ❤️*
