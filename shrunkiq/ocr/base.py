from abc import ABC, abstractmethod
from typing import Union
from PIL import Image
from shrunkiq.utils import pil_image_to_base64

class BaseOCR(ABC):
    """Base class for OCR implementations."""
    
    @abstractmethod
    def extract_text(self, image: Union[Image.Image, str], **kwargs) -> str:
        """Extract text from an image.
        
        Args:
            image (Union[Image.Image, str]): PIL Image or path to image file
            **kwargs: Additional arguments for specific OCR implementation
            
        Returns:
            str: Extracted text from the image
        """
        pass
    
    def _prepare_image(self, image: Union[Image.Image, str], base64: bool = False) -> Image.Image:
        """Prepare image for OCR by converting to RGB if needed.
        
        Args:
            image (Union[Image.Image, str]): PIL Image or path to image file
            
        Returns:
            Image.Image: Processed PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if base64:
            return pil_image_to_base64(image)
        return image 