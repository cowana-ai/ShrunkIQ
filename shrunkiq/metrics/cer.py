import editdistance


def cer(pred: str, truth: str) -> float:
    """Compute Character Error Rate (CER) between predicted and ground-truth text.

    Args:
        pred (str): Predicted text from OCR
        truth (str): Ground-truth text

    Returns:
        float: CER value (0.0 is perfect match, higher is worse)
    """
    if not truth:
        return 0.0 if not pred else 1.0
    dist = editdistance.eval(pred, truth)
    return dist / len(truth)
