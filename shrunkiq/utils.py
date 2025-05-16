import base64
import io
import os
import tempfile
import subprocess
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

def compress_pil(img, quality):
    """Compress a PIL Image using JPEG compression.
    
    Args:
        img (PIL.Image.Image): The image to compress
        quality (int): JPEG quality (0-100)
        
    Returns:
        PIL.Image.Image: Compressed image
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf)

def compress_pdf(input_pdf: str, output_pdf: str, dpi: int = 100, quality: int = 60) -> str:
    """Compress a PDF file by reducing image resolution and quality using pure Python.
    
    This function uses PyMuPDF (fitz) to:
    1. Extract and compress images to the specified quality
    2. Optimize the PDF structure
    
    Args:
        input_pdf (str): Path to input PDF file
        output_pdf (str): Path to save compressed PDF file
        dpi (int, optional): Target DPI for images. Defaults to 100.
        quality (int, optional): JPEG quality (0-100). Defaults to 60.
    
    Returns:
        str: Path to the compressed PDF file
    """
    # Ensure quality is within valid range
    quality = max(0, min(100, quality))
    
    # Calculate scaling factor based on DPI (72 DPI is standard PDF resolution)
    scale_factor = min(1.0, dpi / 72.0)
    
    # Open the input PDF
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()  # Create a new empty PDF
    
    # First pass: Render pages to images and compress them
    for page_num in range(len(doc)):
        src_page = doc[page_num]
        
        # Set the rendering matrix (controls resolution)
        zoom = 2.0  # Higher for better quality
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap (image)
        pix = src_page.get_pixmap(matrix=matrix)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize if needed based on scaling factor
        if scale_factor < 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Compress the image using the provided function
        compressed_img = compress_pil(img, quality)
        
        # Create a new page
        new_page = new_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
        
        # Convert back to bytes
        img_bytes = io.BytesIO()
        compressed_img.save(img_bytes, format="JPEG", optimize=True)
        img_bytes.seek(0)
        
        # Insert the compressed image
        new_page.insert_image(new_page.rect, stream=img_bytes)
    
    # Save the new PDF
    new_doc.save(
        output_pdf,
        garbage=4,         # Maximum garbage collection
        deflate=True,      # Compress streams
        pretty=False,      # Don't use pretty format (more compact)
    )
    
    # Close both documents
    doc.close()
    new_doc.close()
    
    # Get file sizes for comparison
    input_size = os.path.getsize(input_pdf)
    output_size = os.path.getsize(output_pdf)
    compression_ratio = (1 - (output_size / input_size)) * 100
    
    print(f"Compressed PDF: {input_size:,} bytes → {output_size:,} bytes "
          f"({compression_ratio:.1f}% reduction)")
    
    return output_pdf 