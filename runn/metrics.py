"""
Useful metrics for evaluating the performance of the neural network models.
"""

import numpy as np


def accuracy(proba: np.ndarray, y: np.ndarray) -> float:
    """Accuracy metric.

    Args:
        proba: Matrix of predicted choice probabilities. Each row corresponds to a sample and each column to an
        alternative.
        y: Array of true choices. Each element corresponds to the index of the chosen alternative.

    Returns:
        Accuracy metric.
    """
    return np.mean(np.argmax(proba, axis=1) == y)


def AMPCA(proba: np.ndarray, y: np.ndarray) -> float:
    """Arithmetic Mean Probability of Correct Assignment (AMPCA) metric.

    Args:
        proba: Matrix of predicted choice probabilities. Each row corresponds to a sample and each column to an
        alternative.
        y: Array of true choices. Each element corresponds to the index of the chosen alternative.

    Returns:
        AMPCA metric.
    """
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + proba[i, sel_mode]
        i += 1
    N = i - 1
    return sum / N


def CEL(proba: np.ndarray, y: np.ndarray) -> float:
    """Cross-Entropy Loss (CEL) metric.

    Args:
        proba: Matrix of predicted choice probabilities. Each row corresponds to a sample and each column to an
        alternative.
        y: Array of true choices. Each element corresponds to the index of the chosen alternative.

    Returns:
        CEL metric.
    """
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + np.log(proba[i, sel_mode])
        i += 1
    N = i - 1
    return -sum / N


def GMPCA(proba: np.ndarray, y: np.ndarray) -> float:
    """Geometric Mean Probability of Correct Assignment (GMPCA) metric.

    Args:
        proba: Matrix of predicted choice probabilities. Each row corresponds to a sample and each column to an
        alternative.
        y: Array of true choices. Each element corresponds to the index of the chosen alternative.

    Returns:
        GMPCA metric.
    """
    return np.exp(-CEL(proba, y))
