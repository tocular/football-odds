"""
Ensemble model combining Dixon-Coles with GBM corrections.

Provides a unified interface for training, predicting, and saving
the complete model pipeline.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .dixon_coles import DixonColesModel
from .gbm_correction import GBMCorrector
from .calibration import TemperatureScaler, compute_calibration_metrics

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model combining Dixon-Coles Poisson with GBM corrections.

    Pipeline:
    1. Dixon-Coles base model predicts lambda_home, lambda_away
    2. GBM corrector adjusts lambdas based on situational features
    3. Temperature scaler calibrates final probabilities
    """

    def __init__(
        self,
        time_decay: float = 0.0018,
        gbm_params: Optional[dict] = None,
        version: str = "v1.0"
    ):
        """
        Initialize the ensemble model.

        Args:
            time_decay: Time decay for Dixon-Coles model
            gbm_params: Parameters for GBM corrector
            version: Model version string
        """
        self.version = version

        # Initialize components
        self.dixon_coles = DixonColesModel(time_decay=time_decay)
        self.gbm_corrector = GBMCorrector(**(gbm_params or {}))
        self.calibrator = TemperatureScaler()

        self.is_fitted = False
        self.feature_columns = None

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list,
        val_df: Optional[pd.DataFrame] = None,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        date_col: str = "date"
    ) -> "EnsembleModel":
        """
        Fit the complete ensemble model.

        Args:
            train_df: Training data with features and targets
            feature_columns: List of feature columns for GBM
            val_df: Optional validation data for calibration
            home_team_col: Column name for home team
            away_team_col: Column name for away team
            home_goals_col: Column name for home goals
            away_goals_col: Column name for away goals
            date_col: Column name for match date

        Returns:
            Fitted model (self)
        """
        self.feature_columns = feature_columns

        logger.info("Fitting ensemble model...")

        # Step 1: Fit Dixon-Coles base model
        logger.info("Step 1: Fitting Dixon-Coles model...")
        self.dixon_coles.fit(
            train_df,
            home_team_col=home_team_col,
            away_team_col=away_team_col,
            home_goals_col=home_goals_col,
            away_goals_col=away_goals_col,
            date_col=date_col
        )

        # Generate base predictions for training data
        train_with_preds = self.dixon_coles.predict_batch(train_df)

        # Step 2: Fit GBM corrector on residuals
        logger.info("Step 2: Fitting GBM corrector...")
        self.gbm_corrector.fit(
            train_with_preds,
            feature_columns=feature_columns,
            lambda_home_col="lambda_home",
            lambda_away_col="lambda_away",
            home_goals_col=home_goals_col,
            away_goals_col=away_goals_col
        )

        # Step 3: Fit calibrator on validation data (or last portion of training)
        logger.info("Step 3: Fitting probability calibrator...")

        if val_df is not None and len(val_df) > 0:
            cal_df = val_df
        else:
            # Use last 20% of training data for calibration
            cal_df = train_df.tail(int(len(train_df) * 0.2))

        # Generate calibration predictions
        cal_preds = self._predict_uncalibrated(cal_df)

        # Create labels (0=H, 1=D, 2=A)
        labels = np.where(
            cal_df[home_goals_col] > cal_df[away_goals_col], 0,
            np.where(cal_df[home_goals_col] == cal_df[away_goals_col], 1, 2)
        )

        probs = np.column_stack([
            cal_preds["prob_home"],
            cal_preds["prob_draw"],
            cal_preds["prob_away"]
        ])

        self.calibrator.fit(probs, labels)

        self.is_fitted = True
        logger.info("Ensemble model fitting complete")

        return self

    def _predict_uncalibrated(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions without calibration.

        Args:
            df: DataFrame with match features

        Returns:
            DataFrame with predictions
        """
        # Dixon-Coles base predictions
        result = self.dixon_coles.predict_batch(df)

        # Apply GBM corrections
        result = self.gbm_corrector.correct_predictions(result)

        # Recalculate probabilities with corrected lambdas
        probs = []
        for _, row in result.iterrows():
            prob_h, prob_d, prob_a = self._lambda_to_probs(
                row["lambda_home_corrected"],
                row["lambda_away_corrected"]
            )
            probs.append({
                "prob_home": prob_h,
                "prob_draw": prob_d,
                "prob_away": prob_a
            })

        probs_df = pd.DataFrame(probs)
        result["prob_home"] = probs_df["prob_home"]
        result["prob_draw"] = probs_df["prob_draw"]
        result["prob_away"] = probs_df["prob_away"]

        return result

    def _lambda_to_probs(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10
    ) -> Tuple[float, float, float]:
        """
        Convert lambda values to 1X2 probabilities using Dixon-Coles adjustment.

        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            max_goals: Maximum goals to consider

        Returns:
            Tuple of (prob_home, prob_draw, prob_away)
        """
        rho = self.dixon_coles.rho

        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self.dixon_coles._tau(i, j, lambda_home, lambda_away, rho)
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away) * tau

                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p

        total = prob_home + prob_draw + prob_away
        return prob_home / total, prob_draw / total, prob_away / total

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate calibrated predictions for matches.

        Args:
            df: DataFrame with match features

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get uncalibrated predictions
        result = self._predict_uncalibrated(df)

        # Apply calibration
        probs = np.column_stack([
            result["prob_home"],
            result["prob_draw"],
            result["prob_away"]
        ])

        calibrated = self.calibrator.calibrate(probs)

        result["prob_home_calibrated"] = calibrated[:, 0]
        result["prob_draw_calibrated"] = calibrated[:, 1]
        result["prob_away_calibrated"] = calibrated[:, 2]

        # Use calibrated as final
        result["prob_home"] = result["prob_home_calibrated"]
        result["prob_draw"] = result["prob_draw_calibrated"]
        result["prob_away"] = result["prob_away_calibrated"]

        return result

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        features: Optional[dict] = None
    ) -> dict:
        """
        Predict a single match.

        Args:
            home_team: Home team name
            away_team: Away team name
            features: Optional dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        # Create single-row DataFrame
        row = {"home_team": home_team, "away_team": away_team}

        if features:
            row.update(features)
        else:
            # Use neutral features if not provided
            for col in self.feature_columns:
                row[col] = 0

        df = pd.DataFrame([row])
        result = self.predict(df)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": result["lambda_home_corrected"].iloc[0],
            "lambda_away": result["lambda_away_corrected"].iloc[0],
            "prob_home": result["prob_home"].iloc[0],
            "prob_draw": result["prob_draw"].iloc[0],
            "prob_away": result["prob_away"].iloc[0]
        }

    def evaluate(
        self,
        test_df: pd.DataFrame,
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals"
    ) -> dict:
        """
        Evaluate model on test data.

        Args:
            test_df: Test data
            home_goals_col: Column with actual home goals
            away_goals_col: Column with actual away goals

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(test_df)

        # Create labels
        labels = np.where(
            test_df[home_goals_col] > test_df[away_goals_col], 0,
            np.where(test_df[home_goals_col] == test_df[away_goals_col], 1, 2)
        )

        probs = np.column_stack([
            predictions["prob_home"],
            predictions["prob_draw"],
            predictions["prob_away"]
        ])

        # Calibration metrics
        cal_metrics = compute_calibration_metrics(probs, labels)

        # Accuracy
        predicted_outcomes = np.argmax(probs, axis=1)
        accuracy = np.mean(predicted_outcomes == labels)

        return {
            "accuracy": accuracy,
            "log_loss": cal_metrics["log_loss"],
            "brier_score": cal_metrics["brier_score"],
            "ece": cal_metrics["ece"],
            "n_samples": len(test_df)
        }

    def save(self, path: str = "models/current") -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model components
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "dixon_coles.pkl", "wb") as f:
            pickle.dump(self.dixon_coles, f)

        with open(path / "gbm_corrector.pkl", "wb") as f:
            pickle.dump(self.gbm_corrector, f)

        with open(path / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator, f)

        # Save metadata
        metadata = {
            "version": self.version,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/current") -> "EnsembleModel":
        """
        Load model from disk.

        Args:
            path: Directory containing model components

        Returns:
            Loaded model
        """
        path = Path(path)

        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        model = cls(version=metadata["version"])
        model.feature_columns = metadata["feature_columns"]
        model.is_fitted = metadata["is_fitted"]

        with open(path / "dixon_coles.pkl", "rb") as f:
            model.dixon_coles = pickle.load(f)

        with open(path / "gbm_corrector.pkl", "rb") as f:
            model.gbm_corrector = pickle.load(f)

        with open(path / "calibrator.pkl", "rb") as f:
            model.calibrator = pickle.load(f)

        logger.info(f"Model loaded from {path}")
        return model
