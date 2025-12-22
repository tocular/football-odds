"""
Gradient Boosting Machine correction layer for Dixon-Coles predictions.

Uses LightGBM to learn residual patterns that the Poisson model misses,
particularly situational factors like form, rest, and fixture congestion.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class GBMCorrector:
    """
    GBM-based correction layer for Poisson model predictions.

    Learns to predict residuals between Poisson predictions and actual
    outcomes, capturing patterns the base model misses.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize the GBM corrector.

        Args:
            n_estimators: Number of boosting iterations
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            min_child_samples: Minimum samples per leaf
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            random_state: Random seed
        """
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1
        }

        self.model_home = None
        self.model_away = None
        self.feature_columns = None

    def compute_residuals(
        self,
        df: pd.DataFrame,
        lambda_home_col: str = "lambda_home",
        lambda_away_col: str = "lambda_away",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals"
    ) -> pd.DataFrame:
        """
        Compute log residuals between Poisson predictions and actual goals.

        Residuals are computed as: log(actual + 0.5) - log(predicted)
        Using log scale so corrections are multiplicative.

        Args:
            df: DataFrame with predictions and actual goals
            lambda_home_col: Column with predicted home lambda
            lambda_away_col: Column with predicted away lambda
            home_goals_col: Column with actual home goals
            away_goals_col: Column with actual away goals

        Returns:
            DataFrame with residual columns added
        """
        result = df.copy()

        # Add small constant to avoid log(0)
        result["residual_home"] = (
            np.log(result[home_goals_col] + 0.5) -
            np.log(result[lambda_home_col])
        )
        result["residual_away"] = (
            np.log(result[away_goals_col] + 0.5) -
            np.log(result[lambda_away_col])
        )

        return result

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        lambda_home_col: str = "lambda_home",
        lambda_away_col: str = "lambda_away",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        early_stopping_rounds: int = 10,
        validation_fraction: float = 0.2
    ) -> "GBMCorrector":
        """
        Fit GBM models to predict residuals.

        Args:
            df: DataFrame with features, predictions, and actual goals
            feature_columns: List of feature column names to use
            lambda_home_col: Column with predicted home lambda
            lambda_away_col: Column with predicted away lambda
            home_goals_col: Column with actual home goals
            away_goals_col: Column with actual away goals
            early_stopping_rounds: Early stopping patience
            validation_fraction: Fraction of data for validation

        Returns:
            Fitted model (self)
        """
        self.feature_columns = feature_columns

        # Compute residuals
        df_residuals = self.compute_residuals(
            df, lambda_home_col, lambda_away_col, home_goals_col, away_goals_col
        )

        # Prepare features
        X = df_residuals[feature_columns].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Split for validation (use last portion for time-series validity)
        split_idx = int(len(X) * (1 - validation_fraction))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]

        y_home_train = df_residuals["residual_home"].iloc[:split_idx]
        y_home_val = df_residuals["residual_home"].iloc[split_idx:]

        y_away_train = df_residuals["residual_away"].iloc[:split_idx]
        y_away_val = df_residuals["residual_away"].iloc[split_idx:]

        logger.info(f"Training GBM corrector on {len(X_train)} samples, validating on {len(X_val)}")

        # Train home model
        self.model_home = lgb.LGBMRegressor(**self.params)
        self.model_home.fit(
            X_train, y_home_train,
            eval_set=[(X_val, y_home_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )

        # Train away model
        self.model_away = lgb.LGBMRegressor(**self.params)
        self.model_away.fit(
            X_train, y_away_train,
            eval_set=[(X_val, y_away_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )

        # Log performance
        home_rmse = np.sqrt(np.mean((self.model_home.predict(X_val) - y_home_val) ** 2))
        away_rmse = np.sqrt(np.mean((self.model_away.predict(X_val) - y_away_val) ** 2))
        logger.info(f"GBM validation RMSE - Home: {home_rmse:.4f}, Away: {away_rmse:.4f}")

        return self

    def predict_correction(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict correction factors for new matches.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (home_correction, away_correction) arrays
            Corrections are multiplicative factors for lambda
        """
        if self.model_home is None or self.model_away is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())

        # Predict log corrections
        log_correction_home = self.model_home.predict(X)
        log_correction_away = self.model_away.predict(X)

        # Convert to multiplicative factors
        correction_home = np.exp(log_correction_home)
        correction_away = np.exp(log_correction_away)

        return correction_home, correction_away

    def correct_predictions(
        self,
        df: pd.DataFrame,
        lambda_home_col: str = "lambda_home",
        lambda_away_col: str = "lambda_away"
    ) -> pd.DataFrame:
        """
        Apply corrections to Poisson predictions.

        Args:
            df: DataFrame with Poisson predictions and features
            lambda_home_col: Column with predicted home lambda
            lambda_away_col: Column with predicted away lambda

        Returns:
            DataFrame with corrected predictions
        """
        result = df.copy()

        correction_home, correction_away = self.predict_correction(df)

        result["lambda_home_corrected"] = result[lambda_home_col] * correction_home
        result["lambda_away_corrected"] = result[lambda_away_col] * correction_away

        # Clip to reasonable range
        result["lambda_home_corrected"] = result["lambda_home_corrected"].clip(0.1, 5.0)
        result["lambda_away_corrected"] = result["lambda_away_corrected"].clip(0.1, 5.0)

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from both models.

        Returns:
            DataFrame with feature importance scores
        """
        if self.model_home is None:
            raise ValueError("Model not fitted. Call fit() first.")

        importance_home = self.model_home.feature_importances_
        importance_away = self.model_away.feature_importances_

        df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance_home": importance_home,
            "importance_away": importance_away,
            "importance_avg": (importance_home + importance_away) / 2
        })

        return df.sort_values("importance_avg", ascending=False)

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        n_splits: int = 5
    ) -> dict:
        """
        Perform time-series cross-validation.

        Args:
            df: DataFrame with features and targets
            feature_columns: List of feature columns
            n_splits: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        df_residuals = self.compute_residuals(df)
        X = df_residuals[feature_columns].fillna(df_residuals[feature_columns].median())

        tscv = TimeSeriesSplit(n_splits=n_splits)

        home_rmses = []
        away_rmses = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_home_train = df_residuals["residual_home"].iloc[train_idx]
            y_home_val = df_residuals["residual_home"].iloc[val_idx]
            y_away_train = df_residuals["residual_away"].iloc[train_idx]
            y_away_val = df_residuals["residual_away"].iloc[val_idx]

            # Fit models
            model_home = lgb.LGBMRegressor(**self.params)
            model_home.fit(X_train, y_home_train)

            model_away = lgb.LGBMRegressor(**self.params)
            model_away.fit(X_train, y_away_train)

            # Evaluate
            home_rmse = np.sqrt(np.mean((model_home.predict(X_val) - y_home_val) ** 2))
            away_rmse = np.sqrt(np.mean((model_away.predict(X_val) - y_away_val) ** 2))

            home_rmses.append(home_rmse)
            away_rmses.append(away_rmse)

        return {
            "home_rmse_mean": np.mean(home_rmses),
            "home_rmse_std": np.std(home_rmses),
            "away_rmse_mean": np.mean(away_rmses),
            "away_rmse_std": np.std(away_rmses),
            "fold_results": list(zip(home_rmses, away_rmses))
        }
