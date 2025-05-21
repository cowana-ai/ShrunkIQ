from langchain_openai import ChatOpenAI
from PIL import Image

from shrunkiq.ocr.base import BaseOCR


class LLMOCR(BaseOCR):
    """LLM OCR implementation."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.,
        api_key: str | None = None
    ):
        """Initialize LLMOCR.

        Args:
            model_name (str, optional): OpenAI model name. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.prompt =  """
        Act as OCR and output what's written in the following image.
        Output only what's visible and don't use your knowledge to predict what's written.
        - You MUST answer *only* with text verbatim from the passage below.
        - You MUST NOT infer, fill gaps, or use any world knowledge.
        """

    def extract_text(self, image: Image.Image | str, **kwargs) -> str:
        """Extract text from an image using LLM.

        Args:
            image (Union[Image.Image, str]): PIL Image or path to image file
            **kwargs: Additional arguments passed to pytesseract.image_to_string

        Returns:
            str: Extracted text from the image
        """
        image = self._prepare_image(image, base64=True)
        message = {
        "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.prompt,
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image,
                    "mime_type": "image/jpeg",
                },
            ],
        }
        response = self.llm.invoke([message])
        return response.content.strip()
