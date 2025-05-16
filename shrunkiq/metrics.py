from typing import Dict

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