from typing import List, Optional
from .base import BaseQA
from PIL import Image
from shrunkiq.utils import pil_image_to_base64
from .models import QAPair, QAResponse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class QAGenerator(BaseQA):
    """Generate question templates for document comprehension."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=QAResponse)
    
    def generate_qa_pairs(
        self,
        page_img: Image.Image,
        num_pairs: int = 3,
        context: Optional[str] = None
    ) -> QAResponse:
        """Generate question-answer templates for document comprehension.
        
        Args:
            page_img (Image.Image): The page image to analyze
            num_pairs (int, optional): Number of QA pairs to generate. Defaults to 3.
            context (Optional[str], optional): Additional context about document type/topic. Defaults to None.
            
        Returns:
            QAResponse: Structured response containing QA pairs
        """
        prompt = self._build_qa_generation_prompt(num_pairs, context)
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": pil_image_to_base64(page_img),
                    "mime_type": "image/jpeg",
                },
            ],
        }
        response = self.llm.invoke([message])
        return self.output_parser.parse(response.content)
    
    def _build_qa_generation_prompt(
        self,
        num_pairs: int,
        context: Optional[str]
    ) -> str:
        """Build prompt for QA generation.
        
        Args:
            num_pairs (int): Number of QA pairs to generate
            context (Optional[str]): Additional context about document type/topic
            
        Returns:
            str: Generated prompt
        """
        format_instructions = self.output_parser.get_format_instructions()
        
        template = """You're a helpful assistant.

Read the page of the document provided and generate {num_pairs} question-answer pairs that would be used for comprehension verification of the compressed document.
The questions should cover the important parts of the content and act as assertions for content preservation.

{format_instructions}

{context_str}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["num_pairs"],
            partial_variables={
                "format_instructions": format_instructions,
                "context_str": f"\nContext:\n{context}" if context else ""
            }
        )
        
        return prompt.format(num_pairs=num_pairs) 