import base64
import io
from typing import List, Tuple
from PIL import Image
import fitz

def pil_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to base64 string.
    
    Args:
        image (PIL.Image.Image): The PIL Image to convert
        format (str, optional): Image format to save as. Defaults to "PNG".
    
    Returns:
        str: Base64 encoded string of the image
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def pdf_to_images(pdf_path: str, zoom: float = 2.0) -> List[Tuple[int, Image.Image, fitz.Page]]:
    """Convert a PDF file to a list of PIL Images using PyMuPDF.
    
    Args:
        pdf_path (str): Path to PDF file
        zoom (float, optional): Zoom factor for rendering. Defaults to 2.0.
    
    Returns:
        List[Tuple[int, Image.Image, fitz.Page]]: List of tuples containing page number, PIL Image, and PyMuPDF Page object
    """
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((i, img, page))
    return images 