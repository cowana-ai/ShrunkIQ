from dataclasses import dataclass


@dataclass
class HallucinationPoint:
    """Represents the point at which hallucination occurred."""
    font_size: int
    compression_quality: int
    is_human_readable: bool
    llm_prediction: str
    tesseract_prediction: str
    source_text: str
    target_text: str

    def __str__(self) -> str:
        return (
            f"Hallucination(font={self.font_size}, compression={self.compression_quality}, "
            f"readable={self.is_human_readable})"
        )

@dataclass
class ProbeMetrics:
    """Metrics collected during LLM probing."""
    # Basic stats
    total_samples: int
    successful_hallucinations: int
    failed_attempts: int

    # Font size statistics
    avg_hallucination_font_size: float
    min_hallucination_font_size: int
    max_hallucination_font_size: int

    # Compression statistics
    avg_hallucination_compression: float
    min_hallucination_compression: int
    max_hallucination_compression: int

    # Readability metrics
    human_readable_hallucinations: int
    human_unreadable_hallucinations: int

    # Detailed hallucination points
    hallucination_points: list[HallucinationPoint]

    # Error cases
    error_cases: list[dict[str, str]]

    @property
    def hallucination_rate(self) -> float:
        """Rate of successful hallucinations."""
        return self.successful_hallucinations / self.total_samples if self.total_samples > 0 else 0

    @property
    def human_readable_rate(self) -> float:
        """Rate of hallucinations that were human-readable."""
        return (self.human_readable_hallucinations / self.successful_hallucinations
                if self.successful_hallucinations > 0 else 0)

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary format."""
        return {
            "total_samples": self.total_samples,
            "successful_hallucinations": self.successful_hallucinations,
            "failed_attempts": self.failed_attempts,
            "hallucination_rate": self.hallucination_rate,
            "human_readable_rate": self.human_readable_rate,
            "avg_hallucination_font_size": self.avg_hallucination_font_size,
            "min_hallucination_font_size": self.min_hallucination_font_size,
            "max_hallucination_font_size": self.max_hallucination_font_size,
            "avg_hallucination_compression": self.avg_hallucination_compression,
            "min_hallucination_compression": self.min_hallucination_compression,
            "max_hallucination_compression": self.max_hallucination_compression,
            "human_readable_hallucinations": self.human_readable_hallucinations,
            "human_unreadable_hallucinations": self.human_unreadable_hallucinations
        }

def analyze_readibility_from_keywords(keywords: list[str], reconstructed_text: str, threshold: float = 0.5) -> float:
    """Analyze the readability of a reconstructed text based on keywords.

    Args:
        keywords: List of keywords to check for in the reconstructed text
        reconstructed_text: The reconstructed text to analyze

    Returns:
        A score between 0 and 1 indicating the readability of the text
    """
    return sum(keyword in reconstructed_text for keyword in keywords) / len(keywords) >= threshold
