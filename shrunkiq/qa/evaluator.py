from typing import List, Dict, Optional, Any
from shrunkiq.ocr import BaseOCR, TesseractOCR
from shrunkiq.qa.generator import QAGenerator
from shrunkiq.qa.answerer import QuestionAnswerer
from shrunkiq.utils import pdf_to_images

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
    
    def evaluate_pdf(
        self,
        pdf_path: str,
        num_questions: int = 5,
        zoom: float = 2.0
    ) -> Dict[str, Any]:
        """Evaluate a PDF document by generating and answering questions.
        
        Args:
            pdf_path (str): Path to the PDF file
            num_questions (int, optional): Number of questions to generate. Defaults to 5.
            zoom (float, optional): Zoom factor for PDF rendering. Defaults to 2.0.
            
        Returns:
            Dict[str, Any]: Evaluation results containing extracted text and QA pairs
        """
        # Extract text from PDF
        pages = pdf_to_images(pdf_path, zoom=zoom)
        extracted_text = []
        
        for page_num, page_img, _ in pages:
            # Generate QA pairs
            qa_pairs = self.generator.generate_qa_pairs(
                page_img=page_img,
                num_pairs=num_questions
            )
        
            page_text = self.ocr.extract_text(page_img)

            # Answer generated questions
            answered_pairs = []
            for qa in qa_pairs.qa_pairs:
                answer = self.answerer.answer_question(
                    question=qa.question,
                    context=page_text
                )
                answered_pairs.append({
                    "question": qa.question,
                    "generated_answer": qa.answer,
                    "model_answer": answer.answer,
                     "model_evidence": answer.evidence,
                })
        
        return {
            "pages": extracted_text,
            "qa_pairs": answered_pairs
        } 