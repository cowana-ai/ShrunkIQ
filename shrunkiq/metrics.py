from typing import List, Dict, Union, Tuple, Optional
import numpy as np
from shrunkiq.qa.models import DocumentEvaluation, PageEvaluation

def compute_bertscore(
    generated_answer: str, 
    predicted_answer: str,
    model_name: str = "roberta-large",
    lang: str = "en"
) -> Dict[str, float]:
    """
    Compute BERTScore between generated and predicted answers.
    
    Args:
        generated_answer (str): Answer provided by the generator
        predicted_answer (str): Answer predicted/extracted from the document
        model_name (str, optional): Model to use for computing BERTScore. 
                                   Defaults to "microsoft/deberta-xlarge-mnli".
    
    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 scores
    """
    try:
        import bert_score
    except ImportError:
        raise ImportError(
            "bert-score package is required for this function. "
            "Install it with `pip install bert-score transformers`."
        )
    
    # Handle empty or invalid inputs
    if not generated_answer or not predicted_answer:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if generated_answer == "⊘" or predicted_answer == "⊘":
        # If either answer is "no answer", check if they match
        if generated_answer == predicted_answer:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        else:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Compute BERTScore
    P, R, F1 = bert_score.score(
        [generated_answer], 
        [predicted_answer],
        model_type=model_name,
        lang=lang,
        rescale_with_baseline=True
    )
    
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }

def evaluate_document_bertscore(
    doc_eval: DocumentEvaluation,
    aggregation: str = "mean"
) -> Dict[str, Union[float, Dict[int, Dict[str, float]]]]:
    """
    Evaluate BERTScore for all QA pairs in a document evaluation.
    
    Args:
        doc_eval (DocumentEvaluation): Document evaluation containing QA pairs
        aggregation (str, optional): Aggregation method, one of 'mean', 'median', 'min', 'max'.
                                    Defaults to "mean".
    
    Returns:
        Dict[str, Union[float, Dict]]: Evaluation metrics including:
            - overall: Dict with precision, recall, f1 (aggregated)
            - by_page: Dict mapping page numbers to score dicts
            - by_question: Dict mapping question indices to score dicts
    """
    all_scores = []
    page_scores = {}
    question_scores = {}
    
    for i, page in enumerate(doc_eval.pages):
        page_scores[page.page_number] = []
        
        for j, qa_pair in enumerate(page.qa_pairs):
            score = compute_bertscore(
                qa_pair.generated_answer,
                qa_pair.predicted_answer
            )
            
            all_scores.append(score)
            page_scores[page.page_number].append(score)
            question_key = f"p{page.page_number}_q{j+1}"
            question_scores[question_key] = score
    
    # Aggregate scores
    agg_func = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max
    }.get(aggregation, np.mean)
    
    # Overall aggregation
    precision_values = [s["precision"] for s in all_scores]
    recall_values = [s["recall"] for s in all_scores]
    f1_values = [s["f1"] for s in all_scores]
    
    overall = {
        "precision": float(agg_func(precision_values)),
        "recall": float(agg_func(recall_values)),
        "f1": float(agg_func(f1_values))
    }
    
    # Aggregate by page
    by_page = {}
    for page_num, scores in page_scores.items():
        if not scores:
            continue
        page_precision = [s["precision"] for s in scores]
        page_recall = [s["recall"] for s in scores]
        page_f1 = [s["f1"] for s in scores]
        
        by_page[page_num] = {
            "precision": float(agg_func(page_precision)),
            "recall": float(agg_func(page_recall)),
            "f1": float(agg_func(page_f1))
        }
    
    return {
        "overall": overall,
        "by_page": by_page,
        "by_question": question_scores
    } 