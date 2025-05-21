from abc import ABC

from langchain_openai import ChatOpenAI


class BaseQA(ABC):
    """Base class for QA components."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        api_key: str | None = None
    ):
        """Initialize BaseQA.

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

    def _format_qa_pair(self, question: str, answer: str) -> dict[str, str]:
        """Format a QA pair.

        Args:
            question (str): The question
            answer (str): The answer

        Returns:
            Dict[str, str]: Formatted QA pair
        """
        return {
            "question": question.strip(),
            "answer": answer.strip()
        }
