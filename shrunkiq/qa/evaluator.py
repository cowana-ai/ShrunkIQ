from typing import List, Dict, Optional, Any, Union, Tuple
import uuid
import numpy as np
from shrunkiq.ocr import BaseOCR, TesseractOCR
from shrunkiq.qa.generator import QAGenerator
from shrunkiq.qa.answerer import QuestionAnswerer
from shrunkiq.utils import pdf_to_images
from shrunkiq.qa.models import (
    PageContent, Question, GroundTruthAnswer, Prediction,
    GroundTruth, Predictions, EvaluationMetric, QuestionEvaluation, EvaluationResult
)
from shrunkiq.metrics import compute_bertscore

class PDFEvaluator:
    """Evaluate PDF documents using OCR and QA capabilities."""
    
    def __init__(
        self,
        ocr_engine: Optional[BaseOCR] = None,
        qa_generator: Optional[QAGenerator] = None,
        qa_answerer: Optional[QuestionAnswerer] = None,
        **kwargs: Any
    ):
        """Initialize PDFEvaluator.
        
        Args:
            ocr_engine (Optional[BaseOCR], optional): OCR engine. Defaults to TesseractOCR.
            qa_generator (Optional[QAGenerator], optional): QA generator. Defaults to new instance.
            qa_answerer (Optional[QuestionAnswerer], optional): QA answerer. Defaults to new instance.
            **kwargs: Additional arguments for QA components
        """
        self.ocr = ocr_engine or TesseractOCR()
        self.generator = qa_generator or QAGenerator(**kwargs)
        self.answerer = qa_answerer or QuestionAnswerer(**kwargs)
    
    def extract_pdf_content(self, pdf_path: str, zoom: float = 2.0) -> List[PageContent]:
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
        questions: List[Union[str, Question]],
        zoom: float = 2.0,
    ) -> Predictions:
        """Predict answers to questions from a PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            questions (List[Union[str, Question]]): List of questions or Question objects
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.
            use_whole_document (bool, optional): Whether to use whole document as context.
                                              Defaults to True.
            
        Returns:
            Predictions: Collection of predicted answers
        """
        # Extract text from PDF
        pages = self.extract_pdf_content(pdf_path, zoom)
        
        # Convert string questions to Question objects if needed
        processed_questions = []
        for i, q in enumerate(questions):
            if isinstance(q, str):
                processed_questions.append(
                    Question(
                        id=f"q_{i+1}",
                        text=q
                    )
                )
            else:
                processed_questions.append(q)
        
        predictions = []
        
        # Try to answer each question from each page
        for page in pages:
            for question in processed_questions:
                answer = self.answerer.answer_question(
                    question=question.text,
                    context=page.text
                )
                
                # Only include answers that were actually found
                if answer.answer != "⊘":
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
            
            # Skip questions with no ground truth
            if not gt_answer:
                continue
                
            # Use empty prediction if none found
            if not prediction:
                pred_answer = "⊘"
                pred_evidence = ""
            else:
                pred_answer = prediction.answer
                pred_evidence = prediction.evidence
            
            # Compute metrics
            metrics = []
            
            # Add exact match metric
            exact_match = int(gt_answer.answer == pred_answer)
            if exact_match:
                matched_answers += 1
                
            metrics.append(
                EvaluationMetric(
                    name="exact_match",
                    value=exact_match
                )
            )
            
            # Add BERTScore metric
            bertscore = compute_bertscore(gt_answer.answer, pred_answer)
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
                    prediction=pred_answer,
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
                name="bertscore_f1",
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
    