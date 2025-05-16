from typing import Optional, Dict, Any
from shrunkiq.qa.base import BaseQA
from shrunkiq.qa.models import ExtractiveQAResponse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class QuestionAnswerer(BaseQA):
    """Answer questions based on provided context using extractive QA."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=ExtractiveQAResponse)
    
    def answer_question(
        self,
        question: str,
        context: str,
        max_tokens: int = 500,
        **kwargs: Any
    ) -> ExtractiveQAResponse:
        """Answer a question based on the provided context using extractive QA.
        
        Args:
            question (str): The question to answer
            context (str): The context to use for answering
            max_tokens (int, optional): Maximum tokens in answer. Defaults to 150.
            **kwargs: Additional arguments for the LLM
            
        Returns:
            ExtractiveQAResponse: Structured response containing answer and evidence
        """
        prompt = self._build_answer_prompt(question, context)
        response = self.llm.invoke(prompt, max_tokens=max_tokens, **kwargs)
        return self.output_parser.parse(response.content)
    
    def _build_answer_prompt(self, question: str, context: str) -> str:
        """Build prompt for extractive question answering.
        
        Args:
            question (str): The question to answer
            context (str): The context to use for answering
            
        Returns:
            str: Generated prompt
        """
        format_instructions = self.output_parser.get_format_instructions()
        
        template = """You are an extractive QA assistant.
- You MUST answer *only* with text verbatim from the passage below.
- You MUST NOT infer, fill gaps, or use any world knowledge.
- If the answer is not in the passage, respond with "⊘" (no answer).
- After your answer, always provide the exact character index span(s) from the passage under "Evidence:".

{format_instructions}

Given the following passage, answer the question exactly how it's written in the text without the use of any prior knowledge.

Passage:
{context}

Question:
{question}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "context"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        return prompt.format(question=question, context=context) 