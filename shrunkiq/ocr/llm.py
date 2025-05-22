from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from PIL import Image
from pydantic import BaseModel, Field

from shrunkiq.ocr.base import BaseOCR


class OCROutput(BaseModel):
    """Structured output for OCR results."""

    text: str = Field(
        description="The extracted text from the image, or 'Image is not clear' if unreadable"
    )
    is_clear: bool = Field(
        default=True,
        description="Whether the image is clear enough to read for human"
    )

    @property
    def is_unclear_response(self) -> bool:
        """Check if this is an 'Image is not clear' response."""
        return self.text.lower().strip() == "image is not clear"

    def __str__(self) -> str:
        if self.is_unclear_response:
            return self.text
        return (f"Text: {self.text}\n"
                f"Clear: {self.is_clear}")


class OCRPromptTemplate:
    """Template for OCR prompts with consistent formatting."""

    SYSTEM_TEMPLATE = """You are a precise OCR system that extracts text from images.
    Your task is to accurately transcribe visible text without making assumptions or inferences.

    Key Rules:
    1. Only output text that is clearly visible in the image
    2. Never use external knowledge or context
    3. Never guess or fill in unclear parts
    4. If text is not clearly visible, respond with "Image is not clear"
    """

    HUMAN_TEMPLATE = """Please extract the text from the provided image.

    Requirements:
    - Extract ONLY text that is clearly visible
    - Do NOT use any external knowledge
    - Do NOT make any assumptions about unclear text
    - If the image is not clear enough to read, output exactly "Image is not clear"

    Output Format:
    {format_instructions}
    """

    @classmethod
    def create(cls, output_parser: PydanticOutputParser) -> ChatPromptTemplate:
        """Create a formatted chat prompt template.

        Args:
            output_parser: Parser for structuring the output

        Returns:
            ChatPromptTemplate: Formatted prompt template
        """
        system_message = SystemMessagePromptTemplate.from_template(cls.SYSTEM_TEMPLATE)
        human_message = HumanMessagePromptTemplate.from_template(
            cls.HUMAN_TEMPLATE,
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )

        return ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])


class LLMOCR(BaseOCR):
    """LLM OCR implementation."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        temperature: float = 0.,
        api_key: str | None = None
    ):
        """Initialize LLMOCR.

        Args:
            model_name: OpenAI model name. Defaults to "gpt-4o-mini".
            temperature: Sampling temperature. Defaults to 0.
            api_key: OpenAI API key. Defaults to None.
        """

        self.llm = init_chat_model(model_name, model_provider=model_provider, temperature=temperature)
        self.model_provider = model_provider
        # Set up output parsing and prompt template
        self.output_parser = PydanticOutputParser(pydantic_object=OCROutput)
        self.prompt_template = OCRPromptTemplate.create(self.output_parser)

    def extract_text(self, image: Image.Image | str, **kwargs) -> OCROutput:
        """Extract text from an image using LLM.

        Args:
            image: PIL Image or path to image file
            **kwargs: Additional arguments passed to the model

        Returns:
            OCROutput: Structured output containing extracted text and clarity status
        """
        image_data = self._prepare_image(image, base64=True)
        if self.model_provider == "openai":
            image_content = {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_data,
                    "mime_type": "image/jpeg",
                }
        else:
            image_content = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            }
        # Create message with image
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.prompt_template.format(),
                },
                image_content
            ],
        }

        response = self.llm.invoke([message])
        return self.output_parser.parse(response.content)
