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
    """Compress a PDF file by rendering each page to an image, then compressing
    that image. This method converts entire pages to images.
    
    NOTE: This function converts all page content (including text and vector graphics)
    into raster images. For PDFs that are primarily text or vector-based, this may
    result in an increase in file size or no significant reduction. It is most
    effective for image-heavy PDFs (e.g., scanned documents) or when full-page
    rasterization at a specific DPI and quality is the desired outcome.
    
    Args:
        input_pdf (str): Path to input PDF file
        output_pdf (str): Path to save compressed PDF file
        dpi (int, optional): Target DPI for rendering pages to images. Defaults to 100.
        quality (int, optional): JPEG quality for image compression (0-100). Defaults to 60.
    
    Returns:
        str: Path to the compressed PDF file
    """
    # Ensure quality is within valid range
    quality = max(0, min(100, quality))
    # Ensure DPI is positive
    dpi = max(1, dpi) # DPI must be positive
        
    # Open the input PDF
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()  # Create a new empty PDF
    
    for page_num in range(len(doc)):
        src_page = doc[page_num]
        
        # Render page to pixmap (image) using the target DPI directly
        pix = src_page.get_pixmap(dpi=dpi)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Compress the image using the provided function
        # This step applies JPEG compression to the rendered page image.
        compressed_img = compress_pil(img, quality)
        
        # Create a new page in the output PDF with original dimensions
        # Using src_page.rect.width/height preserves original page size.
        new_page = new_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
        
        # Convert compressed PIL image back to bytes for PDF insertion
        img_bytes = io.BytesIO()
        # Ensure the image mode is compatible if it changed during compress_pil, though unlikely for JPEG
        if compressed_img.mode != "RGB":
             compressed_img = compressed_img.convert("RGB")
        compressed_img.save(img_bytes, format="JPEG", optimize=True)
        img_bytes.seek(0)
        
        # Insert the compressed image onto the new page, filling its rectangle
        new_page.insert_image(new_page.rect, stream=img_bytes)
    
    # Save the new PDF with compression options
    new_doc.save(
        output_pdf,
        garbage=4,         # Maximum garbage collection (compact)
        deflate=True,      # Compress all uncompressed streams
        pretty=False,      # Don't use pretty format (more compact)
    )
    
    # Close both documents
    doc.close()
    new_doc.close()
    
    # Get file sizes for comparison
    input_size = os.path.getsize(input_pdf)
    output_size = os.path.getsize(output_pdf)
    
    if input_size > 0:
        compression_ratio = (1 - (output_size / input_size)) * 100
        print(f"Compressed PDF: {input_size:,} bytes → {output_size:,} bytes "
              f"({compression_ratio:.1f}% reduction)")
    else:
        print(f"Compressed PDF: input size 0 bytes → {output_size:,} bytes")
    
    return output_pdf 