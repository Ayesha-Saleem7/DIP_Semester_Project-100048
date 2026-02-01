import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


def order_points(pts):
    """Order points in correct sequence: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect


def four_point_transform(image, pts):
    """Apply perspective transformation"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calculate height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Ensure minimum size
    if maxWidth < 50 or maxHeight < 50:
        return None
    
    # Create destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def detect_document_advanced(image):
    """Advanced document detection with multiple methods"""
    orig = image.copy()
    
    # Resize for processing
    ratio = image.shape[0] / 800.0
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Try multiple edge detection methods
    contours_list = []
    
    # Method 1: Standard Canny with blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged1 = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed1 = cv2.morphologyEx(edged1, cv2.MORPH_CLOSE, kernel)
    cnts1, _ = cv2.findContours(closed1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list.extend(cnts1)
    
    # Method 2: Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    cnts2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list.extend(cnts2)
    
    # Method 3: Morphological gradient
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh_gradient = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts3, _ = cv2.findContours(thresh_gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list.extend(cnts3)
    
    # Method 4: Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    _, thresh_sobel = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    cnts4, _ = cv2.findContours(thresh_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list.extend(cnts4)
    
    # Combine all contours and sort by area
    all_contours = sorted(contours_list, key=cv2.contourArea, reverse=True)
    
    # Find best document contour
    image_area = image_resized.shape[0] * image_resized.shape[1]
    
    for contour in all_contours[:20]:  # Check top 20 contours
        area = cv2.contourArea(contour)
        
        # Skip if too small (less than 5% of image)
        if area < image_area * 0.05:
            continue
            
        # Skip if too large (more than 95% of image)
        if area > image_area * 0.95:
            continue
        
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Look for 4-sided contour
        if len(approx) == 4:
            # Check if it's roughly rectangular
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # Most documents have aspect ratio between 0.3 and 3.0
            if 0.3 < aspect_ratio < 3.0:
                # Scale back to original size
                screenCnt = approx.reshape(4, 2) * ratio
                return orig, screenCnt, True
    
    # If no contour found, return None
    return orig, None, False


def enhance_document(image):
    """Enhanced document processing with better text clarity"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Remove noise with bilateral filter
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)
    
    # Sharpen the image
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(contrasted, -1, kernel_sharpening)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,  # Larger block size for better results
        C=10
    )
    
    # Clean up small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


# Streamlit UI Configuration
st.set_page_config(
    page_title="Advanced Document Scanner",
    page_icon="📄",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        color: #155724;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        color: #721c24;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📄 Advanced Document Scanner")
st.markdown("**AI-Powered Document Detection & Enhancement**")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload your document image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "pdf"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display original
    st.subheader("📸 Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Scan button
    if st.button("🔍 Scan Document"):
        with st.spinner("🔄 Detecting and processing document..."):
            try:
                # Detect document
                orig, contour, found = detect_document_advanced(image)
                
                if found and contour is not None:
                    # Apply perspective transform
                    warped = four_point_transform(orig, contour)
                    
                    if warped is not None:
                        # Enhance the document
                        scanned = enhance_document(warped)
                        
                        # Display success message
                        st.markdown('<div class="success-message">✅ Document scanned successfully!</div>', 
                                  unsafe_allow_html=True)
                        
                        # Display result
                        st.subheader("📄 Scanned Result")
                        st.image(scanned, use_container_width=True)
                        
                        # Prepare downloads
                        scanned_pil = Image.fromarray(scanned)
                        
                        # PNG download
                        png_buf = io.BytesIO()
                        scanned_pil.save(png_buf, format="PNG", optimize=True)
                        png_data = png_buf.getvalue()
                        
                        # PDF download
                        pdf_buf = io.BytesIO()
                        scanned_pil.save(pdf_buf, format="PDF", resolution=100.0)
                        pdf_data = pdf_buf.getvalue()
                        
                        # Download buttons
                        st.markdown("### 📥 Download Options")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download PNG",
                                data=png_data,
                                file_name="scanned_document.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        with col2:
                            st.download_button(
                                label="📄 Download PDF",
                                data=pdf_data,
                                file_name="scanned_document.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    else:
                        st.markdown('<div class="error-message">❌ Could not transform document properly</div>', 
                                  unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">❌ Could not detect document boundaries</div>', 
                              unsafe_allow_html=True)
                    
                    st.warning("""
                    ### 💡 Tips to Improve Detection:
                    
                    **For best results:**
                    - ✓ Ensure all 4 corners of the document are visible
                    - ✓ Use a contrasting background (dark for white paper)
                    - ✓ Ensure good, even lighting without harsh shadows
                    - ✓ Keep the document flat (no wrinkles or folds)
                    - ✓ Fill 50-80% of the frame with the document
                    - ✓ Take photo from directly above if possible
                    - ✓ Avoid reflections on glossy paper
                    """)
                    
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error processing image: {str(e)}</div>', 
                          unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "⚡ Powered by OpenCV & Streamlit | Built with ❤️ for Digital Image Processing"
    "</div>",
    unsafe_allow_html=True
)
