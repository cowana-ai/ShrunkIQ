import os

import pytesseract
from PIL import Image

from shrunkiq.ocr.base import BaseOCR


class TesseractOCR(BaseOCR):
    """Tesseract OCR implementation."""

    def __init__(self, lang: str = "eng", config: str | None = None, tesseract_cmd: str | None = None):
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

    def get_text_and_confidence(self, image: Image.Image) -> tuple[str, float]:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        confidences = []

        for word, conf in zip(data["text"], data["conf"]):
            if word.strip() != "" and conf > 0:
                words.append(word)
                confidences.append(conf)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return " ".join(words), mean_conf / 100

    def extract_text(self, image: Image.Image | str, return_confidence: bool = False, **kwargs) -> str:
        """Extract text from an image using Tesseract.

        Args:
            image (Union[Image.Image, str]): PIL Image or path to image file
            **kwargs: Additional arguments

        Returns:
            str: Extracted text from the image
        """
        image = self._prepare_image(image)

        text, conf = self.get_text_and_confidence(image)
        if not return_confidence:
            return text.strip()
        return text.strip(), conf
