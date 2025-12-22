"""
Probability calibration module for Nostrus.

Implements temperature scaling and other calibration methods to ensure
predicted probabilities are reliable.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """
    Temperature scaling for probability calibration.

    Adjusts predicted probabilities using a single temperature parameter
    that scales the log-odds before applying softmax.
    """

    def __init__(self):
        """Initialize the temperature scaler."""
        self.temperature = 1.0
        self.is_fitted = False

    def _apply_temperature(
        self,
        probs: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.

        Args:
            probs: Array of probabilities (N, 3) for H/D/A
            temperature: Temperature parameter

        Returns:
            Calibrated probabilities
        """
        # Convert to log-odds, scale, and apply softmax
        log_probs = np.log(probs + 1e-10)
        scaled_log_probs = log_probs / temperature
        calibrated = softmax(scaled_log_probs, axis=1)
        return calibrated

    def _neg_log_likelihood(
        self,
        temperature: float,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute negative log-likelihood for temperature optimization.

        Args:
            temperature: Temperature parameter
            probs: Predicted probabilities
            labels: True labels (0=H, 1=D, 2=A)

        Returns:
            Negative log-likelihood
        """
        calibrated = self._apply_temperature(probs, temperature)

        # Cross-entropy loss
        n_samples = len(labels)
        log_lik = 0.0

        for i in range(n_samples):
            log_lik += np.log(calibrated[i, labels[i]] + 1e-10)

        return -log_lik / n_samples

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        bounds: Tuple[float, float] = (0.1, 10.0)
    ) -> "TemperatureScaler":
        """
        Fit temperature parameter using validation data.

        Args:
            probs: Predicted probabilities (N, 3) for H/D/A
            labels: True labels (0=H, 1=D, 2=A)
            bounds: Search bounds for temperature

        Returns:
            Fitted model (self)
        """
        logger.info("Fitting temperature scaling...")

        result = minimize_scalar(
            self._neg_log_likelihood,
            args=(probs, labels),
            bounds=bounds,
            method="bounded"
        )

        self.temperature = result.x
        self.is_fitted = True

        logger.info(f"Optimal temperature: {self.temperature:.4f}")

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predictions.

        Args:
            probs: Predicted probabilities (N, 3) or (3,)

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Scaler not fitted, returning original probabilities")
            return probs

        # Handle single prediction
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            return self._apply_temperature(probs, self.temperature)[0]

        return self._apply_temperature(probs, self.temperature)


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics.

    Args:
        probs: Predicted probabilities (N, 3)
        labels: True labels (0, 1, or 2)
        n_bins: Number of bins for reliability diagram

    Returns:
        Dictionary with calibration metrics
    """
    n_samples = len(labels)

    # Expected Calibration Error (ECE)
    ece = 0.0
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []

    for outcome in range(3):
        # Binary predictions for this outcome
        outcome_probs = probs[:, outcome]
        outcome_labels = (labels == outcome).astype(int)

        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins

            # Samples in this bin
            in_bin = (outcome_probs >= bin_lower) & (outcome_probs < bin_upper)
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                bin_accuracy = np.mean(outcome_labels[in_bin])
                bin_confidence = np.mean(outcome_probs[in_bin])
                ece += n_in_bin * np.abs(bin_accuracy - bin_confidence)

                bin_counts.append(n_in_bin)
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)

    ece = ece / (n_samples * 3)  # Normalize by total predictions

    # Maximum Calibration Error (MCE)
    if len(bin_accuracies) > 0:
        mce = max(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
    else:
        mce = 0.0

    # Brier score
    brier_scores = []
    for outcome in range(3):
        outcome_labels = (labels == outcome).astype(int)
        brier_scores.append(np.mean((probs[:, outcome] - outcome_labels) ** 2))
    brier_score = np.mean(brier_scores)

    # Log-loss
    log_loss = 0.0
    for i in range(n_samples):
        log_loss -= np.log(probs[i, labels[i]] + 1e-10)
    log_loss = log_loss / n_samples

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "n_samples": n_samples
    }


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    outcome: int = 0,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Generate data for a reliability diagram.

    Args:
        probs: Predicted probabilities (N, 3)
        labels: True labels
        outcome: Which outcome to plot (0=H, 1=D, 2=A)
        n_bins: Number of bins

    Returns:
        DataFrame with bin data for plotting
    """
    outcome_probs = probs[:, outcome]
    outcome_labels = (labels == outcome).astype(int)

    bin_data = []

    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        bin_mid = (bin_lower + bin_upper) / 2

        in_bin = (outcome_probs >= bin_lower) & (outcome_probs < bin_upper)
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            accuracy = np.mean(outcome_labels[in_bin])
            confidence = np.mean(outcome_probs[in_bin])
        else:
            accuracy = None
            confidence = None

        bin_data.append({
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "bin_mid": bin_mid,
            "count": n_in_bin,
            "accuracy": accuracy,
            "confidence": confidence
        })

    return pd.DataFrame(bin_data)
