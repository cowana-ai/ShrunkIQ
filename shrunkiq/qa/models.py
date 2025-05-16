from typing import List
from pydantic import BaseModel, Field

class QAPair(BaseModel):
    """A question-answer pair with assertion."""
    question: str = Field(..., description="The question to ask")
    answer: str = Field(..., description="The expected answer or pattern")
    assertion: str = Field(..., description="What aspect this QA pair verifies")

class QAResponse(BaseModel):
    """Response containing multiple QA pairs."""
    qa_pairs: List[QAPair] = Field(..., description="List of question-answer pairs")

class ExtractiveQAResponse(BaseModel):
    """Response from extractive question answering."""
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The answer extracted verbatim from the text")
    evidence: str = Field(..., description="The evidence span from the text, or ⊘ if no answer found") 