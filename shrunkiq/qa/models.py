from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from shrunkiq.qa.evaluator import PDFEvaluator # For type hinting, if needed later

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

class PageContent(BaseModel):
    """Content extracted from a PDF page."""
    page_number: int = Field(..., description="The 1-indexed page number")
    text: str = Field(..., description="Text extracted from the page using OCR")

class Question(BaseModel):
    """A question to be answered."""
    id: str = Field(..., description="Unique identifier for the question")
    text: str = Field(..., description="The question text")
    category: Optional[str] = Field(None, description="Optional category or type of question")

class GroundTruthAnswer(BaseModel):
    """Ground truth answer to a question."""
    question_id: str = Field(..., description="ID of the question being answered")
    answer: str = Field(..., description="The ground truth answer")
    evidence: Optional[str] = Field(None, description="Supporting evidence or context")
    source_page: Optional[int] = Field(None, description="Page number where the answer was found")

class Prediction(BaseModel):
    """Predicted answer to a question."""
    question_id: str = Field(..., description="ID of the question being answered")
    answer: str = Field(..., description="The predicted answer")
    evidence: str = Field(..., description="Evidence supporting the answer")
    confidence: Optional[float] = Field(None, description="Confidence score for the prediction")
    source_page: Optional[int] = Field(None, description="Page number where the answer was found")

class GroundTruth(BaseModel):
    """Collection of ground truth answers."""
    questions: List[Question] = Field(..., description="List of questions")
    answers: List[GroundTruthAnswer] = Field(..., description="List of ground truth answers")
    
    def get_answer_for_question(self, question_id: str) -> Optional[GroundTruthAnswer]:
        """Get the ground truth answer for a specific question."""
        for answer in self.answers:
            if answer.question_id == question_id:
                return answer
        return None

class Predictions(BaseModel):
    """Collection of predicted answers."""
    predictions: List[Prediction] = Field(..., description="List of predictions")
    
    def get_prediction_for_question(self, question_id: str) -> Optional[Prediction]:
        """Get the prediction for a specific question."""
        for prediction in self.predictions:
            if prediction.question_id == question_id:
                return prediction
        return None

class EvaluationMetric(BaseModel):
    """Evaluation metric result."""
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Computed value of the metric")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details for the metric")

class QuestionEvaluation(BaseModel):
    """Evaluation results for a single question."""
    question_id: str = Field(..., description="ID of the evaluated question")
    question_text: str = Field(..., description="Text of the evaluated question")
    ground_truth: str = Field(..., description="Ground truth answer")
    prediction: str = Field(..., description="Predicted answer")
    metrics: List[EvaluationMetric] = Field(..., description="Evaluation metrics for this question")

class EvaluationResult(BaseModel):
    """Overall evaluation results."""
    overall_metrics: List[EvaluationMetric] = Field(..., description="Overall evaluation metrics")
    question_evaluations: List[QuestionEvaluation] = Field(..., description="Per-question evaluation results")
    
    @property
    def average_f1(self) -> float:
        """Get the average F1 score (bertscore_f1_mean) from the overall metrics."""
        for metric in self.overall_metrics:
            if metric.name == "bertscore_f1_mean":
                return metric.value
        return 0.0

    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Retrieve a specific metric value from overall_metrics."""
        for metric in self.overall_metrics:
            if metric.name == metric_name:
                return metric.value
        return None

    def normalize_scores(self, baseline_eval_result: 'EvaluationResult') -> Dict[str, Any]:
        """
        Normalizes the current evaluation scores against a baseline evaluation result.
        Calculates ratios for key metrics like 'bertscore_f1_mean' and 'exact_match_accuracy'.

        Args:
            baseline_eval_result (EvaluationResult): The baseline evaluation result to normalize against.

        Returns:
            Dict[str, Any]: A dictionary containing normalized scores. 
                            Metrics that cannot be normalized (e.g., baseline is 0) will be omitted
                            or set to a specific value like None or an error string.
        """
        normalized_metrics: Dict[str, Any] = {}

        metrics_to_normalize = [
            "bertscore_f1_mean",
            "exact_match_accuracy",
            "answer_rate"
        ]

        for metric_name in metrics_to_normalize:
            current_value = self.get_metric_value(metric_name)
            baseline_value = baseline_eval_result.get_metric_value(metric_name)

            if current_value is not None and baseline_value is not None:
                if baseline_value == 0:
                    # Avoid division by zero. 
                    # If current is also 0, could be 1.0 (no change), or undefined.
                    # If current is non-zero, it's an infinite improvement (or decline if negative).
                    # For now, let's mark as 'undefined' or skip.
                    normalized_metrics[f"normalized_{metric_name}"] = None
                else:
                    normalized_metrics[f"normalized_{metric_name}"] = current_value / baseline_value
            else:
                raise ValueError(f"Metric {metric_name} is not present in both evaluation results")
            # If current_value is None, it won't be included, which is fine.

        # You could also calculate absolute differences or other comparative stats here.
        return normalized_metrics 