from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import *
from .log import logger

def classification_metrics(preds: Sequence[int],
                           labels: Sequence[int],
                           metric: Optional[str] = "micro-f1",
                          ) -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """
    
    if metric == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score