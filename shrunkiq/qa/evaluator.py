import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from shrunkiq.metrics.bertscore import compute_bertscore
from shrunkiq.qa.models import (EvaluationMetric, EvaluationResult,
                                GroundTruth, GroundTruthAnswer, PageContent,
                                Prediction, Predictions, Question,
                                QuestionEvaluation)
from shrunkiq.utils import pdf_to_images


class PDFEvaluator:
    """Evaluate PDF documents using OCR and QA capabilities."""

    def __init__(
        self,
        config: DictConfig,
    ):
        """Initialize PDFEvaluator.

        Args:
            config (DictConfig): Configuration for the evaluator
        """
        self.ocr = instantiate(config.ocr.model)
        self.generator = instantiate(config.qa.generator)
        self.answerer = instantiate(config.qa.answerer)
        self.ground_truth_for_comparison: GroundTruth | None = None
        self.baseline_evaluation_result: EvaluationResult | None = None

    def extract_pdf_content(self, pdf_path: str, zoom: float = 2.0) -> list[PageContent]:
        """Extract text content from a PDF document using OCR.

        Args:
            pdf_path (str): Path to the PDF file
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.

        Returns:
            List[PageContent]: List of page content with extracted text
        """
        pages = pdf_to_images(pdf_path, zoom=zoom)
        page_contents = []

        for page_num, page_img, _ in pages:
            page_text = self.ocr.extract_text(page_img)
            page_contents.append(
                PageContent(
                    page_number=page_num + 1,
                    text=page_text
                )
            )

        return page_contents

    def generate_ground_truth(
        self,
        pdf_path: str,
        num_questions_per_page: int = 3,
        zoom: float = 2.0
    ) -> GroundTruth:
        """Generate ground truth QA pairs from a PDF document.

        Args:
            pdf_path (str): Path to the PDF file
            num_questions_per_page (int, optional): Number of questions per page. Defaults to 3.
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.

        Returns:
            GroundTruth: Ground truth with questions and answers
        """
        pages = pdf_to_images(pdf_path, zoom=zoom)
        questions = []
        answers = []

        for page_num, page_img, _ in pages:
            # First extract text using OCR
            page_text = self.ocr.extract_text(page_img)

            # Generate questions based on the image
            qa_pairs = self.generator.generate_qa_pairs(
                page_img=page_img,
                num_pairs=num_questions_per_page
            )

            # Answer each generated question using the OCR text
            for qa in qa_pairs.qa_pairs:
                # Create a unique ID for the question
                question_id = f"q_{page_num+1}_{len(questions)+1}"

                # Add the question
                questions.append(
                    Question(
                        id=question_id,
                        text=qa.question,
                        category=qa.assertion
                    )
                )

                # Get extractive answer from OCR text
                answer = self.answerer.answer_question(
                    question=qa.question,
                    context=page_text
                )

                # Add the ground truth answer
                answers.append(
                    GroundTruthAnswer(
                        question_id=question_id,
                        answer=answer.answer,
                        evidence=answer.evidence,
                        source_page=page_num + 1
                    )
                )

        return GroundTruth(
            questions=questions,
            answers=answers
        )

    def predict_answers(
        self,
        pdf_path: str,
        questions: list[Question],
        zoom: float = 2.0,
    ) -> Predictions:
        """Predict answers to questions from a PDF document.

        Args:
            pdf_path (str): Path to the PDF file
            questions (List[Union[str, Question]]): List of questions or Question objects
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.

        Returns:
            Predictions: Collection of predicted answers
        """
        # Extract text from PDF
        pages = self.extract_pdf_content(pdf_path, zoom)

        predictions = []

        # Try to answer each question from each page
        for page in pages:
            for question in questions:
                answer = self.answerer.answer_question(
                    question=question.text,
                    context=page.text
                )

                # Only include answers that were actually found
                predictions.append(
                    Prediction(
                        question_id=question.id,
                        answer=answer.answer,
                        evidence=answer.evidence,
                        source_page=page.page_number
                    )
                )

        return Predictions(predictions=predictions)

    def evaluate(
        self,
        ground_truth: GroundTruth,
        predictions: Predictions
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            ground_truth (GroundTruth): Ground truth answers
            predictions (Predictions): Predicted answers

        Returns:
            EvaluationResult: Evaluation results with metrics
        """
        # Collect evaluation for each question
        question_evaluations = []
        all_bertscore_f1 = []
        matched_answers = 0
        total_questions = len(ground_truth.questions)

        for question in ground_truth.questions:
            # Get ground truth answer
            gt_answer = ground_truth.get_answer_for_question(question.id)

            # Get prediction for this question
            prediction = predictions.get_prediction_for_question(question.id)

            if gt_answer is None or prediction is None:
                raise ValueError(f"No ground truth or prediction found for question {question.id}")

            # Compute metrics
            metrics = []

            # Add exact match metric
            exact_match = int(gt_answer.answer.strip().lower() == prediction.answer.strip().lower())
            if exact_match:
                matched_answers += 1

            metrics.append(
                EvaluationMetric(
                    name="exact_match",
                    value=exact_match
                )
            )

            # Add BERTScore metric
            bertscore = compute_bertscore(gt_answer.answer, prediction.answer)
            all_bertscore_f1.append(bertscore["f1"])

            metrics.append(
                EvaluationMetric(
                    name="bertscore_f1",
                    value=bertscore["f1"],
                    details={
                        "precision": bertscore["precision"],
                        "recall": bertscore["recall"]
                    }
                )
            )

            # Add to question evaluations
            question_evaluations.append(
                QuestionEvaluation(
                    question_id=question.id,
                    question_text=question.text,
                    ground_truth=gt_answer.answer,
                    prediction=prediction.answer,
                    metrics=metrics
                )
            )

        # Compute overall metrics
        overall_metrics = [
            EvaluationMetric(
                name="exact_match_accuracy",
                value=matched_answers / total_questions if total_questions > 0 else 0.0
            ),
            EvaluationMetric(
                name="bertscore_f1_mean", # Renamed for clarity
                value=np.mean(all_bertscore_f1) if all_bertscore_f1 else 0.0
            ),
            EvaluationMetric(
                name="answer_rate",
                value=len([p for p in predictions.predictions if p.answer != "⊘"]) / total_questions if total_questions > 0 else 0.0
            )
        ]

        return EvaluationResult(
            overall_metrics=overall_metrics,
            question_evaluations=question_evaluations
        )

    def establish_baseline(
        self,
        pdf_path: str,
        num_questions_per_page: int = 3,
        zoom: float = 2.0
    ) -> EvaluationResult:
        """Establishes a baseline for evaluation by generating ground truth from the given PDF, predicting answers on
        it, and evaluating them. The ground truth and baseline evaluation result are stored in the evaluator instance.

        Args:
            pdf_path (str): Path to the PDF file to use for generating ground truth and baseline.
            num_questions_per_page (int, optional): Number of questions per page for GT. Defaults to 3.
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.

        Returns:
            EvaluationResult: The evaluation result for the baseline.
        """
        print(f"Establishing baseline for: {pdf_path}")
        # Step 1: Generate ground truth
        self.ground_truth_for_comparison = self.generate_ground_truth(
            pdf_path, num_questions_per_page, zoom
        )
        if not self.ground_truth_for_comparison.questions:
            print("Warning: No questions generated for ground truth. Baseline evaluation will be empty.")
            self.baseline_evaluation_result = EvaluationResult(overall_metrics=[], question_evaluations=[])
            return self.baseline_evaluation_result

        # Step 2: Make initial prediction with original document
        predictions = self.predict_answers(
            pdf_path, self.ground_truth_for_comparison.questions, zoom
        )

        # Step 3: Evaluate against ground truth
        self.baseline_evaluation_result = self.evaluate(
            self.ground_truth_for_comparison, predictions
        )
        print(f"Baseline established. Average BERTScore F1: {self.baseline_evaluation_result.average_f1}")
        return self.baseline_evaluation_result

    def evaluate_document_against_baseline(
        self,
        pdf_path_to_evaluate: str,
        zoom: float = 2.0
    ) -> EvaluationResult:
        """Evaluates a given PDF document against the ground truth established during the baseline phase.

        Args:
            pdf_path_to_evaluate (str): Path to the PDF file to evaluate.
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.

        Returns:
            EvaluationResult: The evaluation result for the given PDF against the baseline GT.

        Raises:
            RuntimeError: If a baseline has not been established first.
        """
        if self.ground_truth_for_comparison is None or not self.ground_truth_for_comparison.questions:
            raise RuntimeError(
                "Baseline ground truth not established or empty. "
                "Call 'establish_baseline' first with a document that yields questions."
            )

        print(f"Evaluating document against baseline: {pdf_path_to_evaluate}")
        # Predict answers for the new PDF using questions from the stored ground truth
        predictions = self.predict_answers(
            pdf_path_to_evaluate, self.ground_truth_for_comparison.questions, zoom
        )

        # Evaluate these predictions against the stored ground truth
        evaluation_result = self.evaluate(
            self.ground_truth_for_comparison, predictions
        )

        normalized_evaluation_result: dict[str, float] = evaluation_result.normalize_scores(self.baseline_evaluation_result)
        metrics = list(normalized_evaluation_result.keys())
        for metric in metrics:
            relative_degradation = 1 - max(normalized_evaluation_result[metric], 0)
            normalized_evaluation_result[f"relative_degradation_{metric}"] = relative_degradation
        return normalized_evaluation_result
