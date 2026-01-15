/**
 * DermAI - Skin Lesion Segmentation Application
 * Frontend JavaScript
 * ============================================
 */

// ===== DOM ELEMENTS =====
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const statusBadge = document.getElementById('status-badge');

// Result elements
const originalImage = document.getElementById('original-image');
const maskImage = document.getElementById('mask-image');
const overlayImage = document.getElementById('overlay-image');
const confidenceValue = document.getElementById('confidence-value');
const areaValue = document.getElementById('area-value');

// Action buttons
const newAnalysisBtn = document.getElementById('new-analysis-btn');
const downloadBtn = document.getElementById('download-btn');

// ===== CONFIGURATION =====
const API_BASE_URL = window.location.origin;

// ===== STATE =====
let currentResults = null;
let currentOriginalImage = null;

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
});

function initializeEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== uploadBtn) {
            fileInput.click();
        }
    });

    // Upload button click
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Action buttons
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    downloadBtn.addEventListener('click', downloadResults);

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
}

// ===== API HEALTH CHECK =====
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusBadge.classList.add('online');
            statusBadge.classList.remove('offline');
            statusBadge.querySelector('.status-text').textContent = 
                data.model_loaded ? 'Model Ready' : 'API Online';
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        console.error('API health check failed:', error);
        statusBadge.classList.add('offline');
        statusBadge.classList.remove('online');
        statusBadge.querySelector('.status-text').textContent = 'Offline';
    }
}

// ===== DRAG AND DROP HANDLERS =====
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// ===== FILE HANDLING =====
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showNotification('Invalid file type. Please upload JPEG, PNG, or WebP images.', 'error');
        return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File too large. Maximum size is 10MB.', 'error');
        return;
    }

    // Store original image for download
    const reader = new FileReader();
    reader.onload = (e) => {
        currentOriginalImage = e.target.result;
        originalImage.src = currentOriginalImage;
    };
    reader.readAsDataURL(file);

    // Send to API
    analyzeImage(file);
}

// ===== API COMMUNICATION =====
async function analyzeImage(file) {
    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/segment`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentResults = data;
            displayResults(data);
        } else {
            throw new Error(data.message || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// ===== RESULTS DISPLAY =====
function displayResults(data) {
    // Set images
    maskImage.src = `data:image/png;base64,${data.mask_base64}`;
    overlayImage.src = `data:image/png;base64,${data.overlay_base64}`;

    // Set metrics
    confidenceValue.textContent = `${data.confidence}%`;
    areaValue.textContent = `${data.lesion_area_percent}%`;

    // Apply color coding to confidence
    if (data.confidence >= 80) {
        confidenceValue.style.color = 'var(--success-400)';
    } else if (data.confidence >= 50) {
        confidenceValue.style.color = '#fbbf24';
    } else {
        confidenceValue.style.color = '#f87171';
    }

    // Show results section
    resultsSection.classList.remove('hidden');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===== RESET ANALYSIS =====
function resetAnalysis() {
    // Hide results
    resultsSection.classList.add('hidden');

    // Clear images
    originalImage.src = '';
    maskImage.src = '';
    overlayImage.src = '';

    // Reset metrics
    confidenceValue.textContent = '--';
    areaValue.textContent = '--';
    confidenceValue.style.color = '';

    // Clear state
    currentResults = null;
    currentOriginalImage = null;

    // Reset file input
    fileInput.value = '';

    // Scroll to upload
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ===== DOWNLOAD RESULTS =====
function downloadResults() {
    if (!currentResults || !currentOriginalImage) {
        showNotification('No results to download', 'error');
        return;
    }

    // Create canvas to composite results
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Load all images
    const images = {
        original: new Image(),
        mask: new Image(),
        overlay: new Image()
    };

    images.original.src = currentOriginalImage;
    images.mask.src = `data:image/png;base64,${currentResults.mask_base64}`;
    images.overlay.src = `data:image/png;base64,${currentResults.overlay_base64}`;

    // Wait for all images to load
    Promise.all([
        new Promise(resolve => images.original.onload = resolve),
        new Promise(resolve => images.mask.onload = resolve),
        new Promise(resolve => images.overlay.onload = resolve)
    ]).then(() => {
        // Calculate dimensions
        const imgWidth = images.original.width;
        const imgHeight = images.original.height;
        const padding = 20;
        const headerHeight = 60;
        const footerHeight = 80;

        canvas.width = imgWidth * 3 + padding * 4;
        canvas.height = imgHeight + headerHeight + footerHeight + padding * 2;

        // Background
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Header
        ctx.fillStyle = '#fafafa';
        ctx.font = 'bold 24px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('DermAI - Skin Lesion Analysis Results', canvas.width / 2, 40);

        // Images
        const y = headerHeight + padding;
        
        // Original
        ctx.drawImage(images.original, padding, y, imgWidth, imgHeight);
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '14px Inter, sans-serif';
        ctx.fillText('Original', padding + imgWidth / 2, y - 10);

        // Mask
        ctx.drawImage(images.mask, imgWidth + padding * 2, y, imgWidth, imgHeight);
        ctx.fillText('Segmentation Mask', imgWidth + padding * 2 + imgWidth / 2, y - 10);

        // Overlay
        ctx.drawImage(images.overlay, imgWidth * 2 + padding * 3, y, imgWidth, imgHeight);
        ctx.fillText('Lesion Overlay', imgWidth * 2 + padding * 3 + imgWidth / 2, y - 10);

        // Footer with metrics
        const footerY = y + imgHeight + padding + 20;
        ctx.fillStyle = '#fafafa';
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
            `Confidence: ${currentResults.confidence}% | Lesion Area: ${currentResults.lesion_area_percent}% | Model: Attention U-Net`,
            canvas.width / 2,
            footerY
        );

        ctx.fillStyle = '#71717a';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText(
            `Generated: ${new Date().toLocaleString()} | DermAI - For research purposes only`,
            canvas.width / 2,
            footerY + 25
        );

        // Download
        const link = document.createElement('a');
        link.download = `dermai-analysis-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();

        showNotification('Results downloaded successfully!', 'success');
    });
}

// ===== UTILITY FUNCTIONS =====
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    } else {
        loadingOverlay.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span class="notification-icon">${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ'}</span>
        <span class="notification-message">${message}</span>
    `;

    // Add styles
    Object.assign(notification.style, {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        background: type === 'success' ? 'rgba(34, 197, 94, 0.9)' : 
                   type === 'error' ? 'rgba(239, 68, 68, 0.9)' : 
                   'rgba(99, 102, 241, 0.9)',
        color: 'white',
        borderRadius: '12px',
        boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        fontSize: '0.9rem',
        fontWeight: '500',
        zIndex: '9999',
        animation: 'slideIn 0.3s ease',
        backdropFilter: 'blur(10px)'
    });

    // Add animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(100px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        @keyframes slideOut {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100px);
            }
        }
    `;
    document.head.appendChild(style);

    document.body.appendChild(notification);

    // Auto remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 4000);
}

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + O to open file
    if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault();
        fileInput.click();
    }

    // Ctrl/Cmd + S to download results
    if ((e.ctrlKey || e.metaKey) && e.key === 's' && currentResults) {
        e.preventDefault();
        downloadResults();
    }

    // Escape to reset
    if (e.key === 'Escape' && !resultsSection.classList.contains('hidden')) {
        resetAnalysis();
    }
});

// ===== WINDOW EVENTS =====
// Prevent accidental page leave when results are present
window.addEventListener('beforeunload', (e) => {
    if (currentResults) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// Re-check API health periodically
setInterval(checkAPIHealth, 30000);
