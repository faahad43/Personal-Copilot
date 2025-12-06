from typing import Optional
from PIL import Image
import pytesseract
import pdfplumber


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file using pdfplumber.

    Returns the concatenated text of all pages. If pdfplumber fails,
    returns an empty string.
    """
    text_chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                    # fallback: try to extract text from images on the page
                    if not page_text:
                        try:
                            im = page.to_image(resolution=150).original
                            page_text = pytesseract.image_to_string(im)
                        except Exception:
                            page_text = ""
                    text_chunks.append(page_text)
                except Exception:
                    continue
    except Exception:
        return ""

    return "\n\n".join([t for t in text_chunks if t])


def extract_text_from_image(path: str, lang: Optional[str] = None) -> str:
    """Extract text from an image file using pytesseract (Tesseract OCR).

    Extracts text from images using optical character recognition.
    Tries multiple OCR configurations for better accuracy.
    
    Args:
        path: Path to the image file
        lang: Optional Tesseract language code (e.g., 'eng', 'fra')
    
    Returns:
        Extracted text from the image, or fallback description if OCR fails
    """
    try:
        img = Image.open(path)
        filename = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
        
        print(f"📷 Processing image: {filename}")
        
        # Prepare OCR arguments
        args = {}
        if lang:
            args['lang'] = lang
        
        # Try multiple OCR configurations for better accuracy
        ocr_configs = [
            r'--oem 3 --psm 6',  # Primary: Default with best accuracy
            r'--oem 1 --psm 3',  # Fallback: Alternative engine
            r'--psm 6',          # Fallback: Simple config
        ]
        
        text = ""
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(img, config=config, **args)
                if text and text.strip():
                    print(f"✓ OCR successful with config: {config}")
                    return text.strip()
            except Exception as config_error:
                print(f"  ⚠️  Config {config} failed: {str(config_error)}")
                continue
        
        # If all OCR attempts failed or returned empty, log and create fallback
        print(f"⚠️  OCR could not extract readable text from '{filename}'")
        
        # Create informative fallback description with image metadata
        try:
            width, height = img.size if hasattr(img, 'size') else (0, 0)
            format_str = img.format if hasattr(img, 'format') else 'Unknown'
            
            # Include image properties to help the model understand what was uploaded
            fallback = f"Image: {filename}\nFile format: {format_str}\nDimensions: {width}x{height} pixels\n\nNote: This image was uploaded but OCR could not extract readable text. It may contain visual content such as photos, diagrams, logos, or handwriting that are not machine-readable text."
            
            print(f"  Using fallback description (image properties)")
            return fallback
        except Exception as inner_e:
            print(f"⚠️  Error creating fallback: {str(inner_e)}")
            return f"Image: {filename}\n\nNote: Image uploaded successfully but text extraction not available."
            
    except Exception as e:
        print(f"❌ Image processing error: {str(e)}")
        # Last resort fallback
        try:
            filename = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
            return f"Image: {filename}\n\nImage file uploaded but could not be processed."
        except:
            return "Image file uploaded but could not be processed."
