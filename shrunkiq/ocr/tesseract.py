import pytesseract
import os
from typing import Union, Optional
from PIL import Image
from shrunkiq.ocr.base import BaseOCR

class TesseractOCR(BaseOCR):
    """Tesseract OCR implementation."""
    
    def __init__(self, lang: str = "eng", config: Optional[str] = None, tesseract_cmd: Optional[str] = None):
        """Initialize TesseractOCR.
        
        Args:
            lang (str, optional): Language for OCR. Defaults to "eng".
            config (Optional[str], optional): Custom Tesseract configuration. Defaults to None.
            tesseract_cmd (Optional[str], optional): Path to Tesseract executable. Defaults to None.
        """
        self.lang = lang
        self.config = config
        
        # Set Tesseract executable path
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.getenv('TESSERACT_CMD') is not None:
            pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
   
    
    def extract_text(self, image: Union[Image.Image, str], **kwargs) -> str:
        """Extract text from an image using Tesseract.
        
        Args:
            image (Union[Image.Image, str]): PIL Image or path to image file
            **kwargs: Additional arguments passed to pytesseract.image_to_string
            
        Returns:
            str: Extracted text from the image
        """
        image = self._prepare_image(image)
        
        # Override default language if provided in kwargs
        lang = kwargs.pop('lang', self.lang)
        config = kwargs.pop('config', self.config)
        
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=config,
            **kwargs
        )
        return text.strip() 