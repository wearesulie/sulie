import numpy as np
from typing import List


def weighted_quantile_loss(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> float:
    """Calculate the Weighted Quantile Loss (WQL) for a set of predictions.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.
        quantiles (list): List of quantiles to calculate.

    Returns:
        float: Weighted Quantile Loss.
    """
    losses = []
    for q in quantiles:
        errors = y_true - y_pred
        loss = np.maximum(q * errors, (q - 1) * errors)
        losses.append(loss.mean())
    return sum(losses) / len(quantiles)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).
    
    Args:
        actual (array-like): Array of actual values.
        predicted (array-like): Array of predicted values.
    
    Returns:
        float: WAPE value as a percentage.
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")
        
    if np.sum(np.abs(actual)) == 0:
        raise ValueError("Sum of actual values cannot be zero")
    
    wape = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    
    return wape


def mae(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between actual and forecasted values.

    Args:
        actuals (np.ndarray): The actual observed values.
        forecasts (np.ndarray): The forecasted values.

    Returns:
        float: The Mean Absolute Error (MAE).
    """
    absolute_errors = np.abs(actuals - forecasts)
    return np.mean(absolute_errors)