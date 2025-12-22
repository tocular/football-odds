#!/usr/bin/env python3
"""
Nostrus Betting Model Pipeline

Trains the Dixon-Coles + GBM ensemble model and runs backtesting
on the 2023-2024 validation season.
"""

import logging
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.builder import FeatureBuilder
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_correction import GBMCorrector
from src.models.calibration import TemperatureScaler, compute_calibration_metrics
from src.betting.kelly import KellyCalculator
from src.betting.backtest import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """
    Load and prepare data for training and validation.

    Returns:
        Tuple of (train_df, val_df, feature_columns)
    """
    logger.info("Loading data from database...")

    builder = FeatureBuilder()

    # Build complete feature matrix
    df = builder.build_feature_matrix()

    logger.info(f"Total matches loaded: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Define train/validation split based on config
    train_end_season = config["validation"]["train_seasons_end"]
    val_season = config["validation"]["validation_season"]

    # Get season order for filtering
    season_order = [
        "2015-2016", "2016-2017", "2017-2018", "2018-2019",
        "2019-2020", "2020-2021", "2021-2022", "2022-2023",
        "2023-2024", "2024-2025"
    ]

    train_seasons = season_order[:season_order.index(train_end_season) + 1]

    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"] == val_season].copy()

    logger.info(f"Training data: {len(train_df)} matches ({train_seasons[0]} to {train_end_season})")
    logger.info(f"Validation data: {len(val_df)} matches ({val_season})")

    # Define feature columns (excluding identifiers, targets, and raw odds)
    feature_columns = builder.get_feature_columns()

    # Filter to columns that exist and have data
    available_features = [c for c in feature_columns if c in df.columns]
    logger.info(f"Using {len(available_features)} features: {available_features}")

    return train_df, val_df, available_features


def train_model(train_df: pd.DataFrame, feature_columns: list, config: dict) -> dict:
    """
    Train the Dixon-Coles + GBM ensemble model.

    Returns:
        Dictionary containing trained model components
    """
    logger.info("=" * 60)
    logger.info("TRAINING PHASE")
    logger.info("=" * 60)

    # Step 1: Fit Dixon-Coles base model
    logger.info("\n[1/3] Fitting Dixon-Coles Poisson model...")

    dc_model = DixonColesModel(time_decay=0.0018)
    dc_model.fit(
        train_df,
        home_team_col="home_team",
        away_team_col="away_team",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
        date_col="date"
    )

    logger.info(f"  Home advantage: {dc_model.home_advantage:.3f}")
    logger.info(f"  Rho (low-score correlation): {dc_model.rho:.3f}")
    logger.info(f"  Number of teams: {len(dc_model.teams)}")

    # Get team ratings for analysis
    team_ratings = dc_model.get_team_ratings()
    logger.info(f"\n  Top 5 teams by overall rating:")
    for _, row in team_ratings.head(5).iterrows():
        logger.info(f"    {row['team']}: attack={row['attack']:.3f}, defense={row['defense']:.3f}")

    # Step 2: Generate base predictions and fit GBM corrector
    logger.info("\n[2/3] Fitting GBM correction layer...")

    # Filter training data to rows with valid features
    train_with_features = train_df.dropna(subset=feature_columns, how="all").copy()

    # Generate base predictions
    train_preds = dc_model.predict_batch(train_with_features)

    # Fit GBM corrector
    gbm_corrector = GBMCorrector(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=20
    )

    gbm_corrector.fit(
        train_preds,
        feature_columns=feature_columns,
        lambda_home_col="lambda_home",
        lambda_away_col="lambda_away",
        home_goals_col="home_goals",
        away_goals_col="away_goals"
    )

    # Feature importance analysis
    importance = gbm_corrector.get_feature_importance()
    logger.info(f"\n  Top 5 features by importance:")
    for _, row in importance.head(5).iterrows():
        logger.info(f"    {row['feature']}: {row['importance_avg']:.1f}")

    # Step 3: Fit temperature calibrator
    logger.info("\n[3/3] Fitting probability calibrator...")

    # Use last 20% of training for calibration
    cal_start = int(len(train_preds) * 0.8)
    cal_df = train_preds.iloc[cal_start:].copy()

    # Apply GBM corrections to calibration set
    cal_corrected = gbm_corrector.correct_predictions(cal_df)

    # Recalculate probabilities with corrected lambdas
    from scipy.stats import poisson

    def lambda_to_probs(lambda_h, lambda_a, rho, max_goals=10):
        prob_h, prob_d, prob_a = 0.0, 0.0, 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = dc_model._tau(i, j, lambda_h, lambda_a, rho)
                p = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) * tau
                if i > j:
                    prob_h += p
                elif i == j:
                    prob_d += p
                else:
                    prob_a += p
        total = prob_h + prob_d + prob_a
        return prob_h/total, prob_d/total, prob_a/total

    probs_list = []
    for _, row in cal_corrected.iterrows():
        ph, pd, pa = lambda_to_probs(
            row["lambda_home_corrected"],
            row["lambda_away_corrected"],
            dc_model.rho
        )
        probs_list.append([ph, pd, pa])

    probs_array = np.array(probs_list)

    # Create labels
    labels = np.where(
        cal_df["home_goals"] > cal_df["away_goals"], 0,
        np.where(cal_df["home_goals"] == cal_df["away_goals"], 1, 2)
    )

    # Fit calibrator
    calibrator = TemperatureScaler()
    calibrator.fit(probs_array, labels)

    logger.info(f"  Temperature: {calibrator.temperature:.4f}")

    # Compute calibration metrics before and after
    before_metrics = compute_calibration_metrics(probs_array, labels)
    calibrated_probs = calibrator.calibrate(probs_array)
    after_metrics = compute_calibration_metrics(calibrated_probs, labels)

    logger.info(f"\n  Calibration improvement:")
    logger.info(f"    Log-loss: {before_metrics['log_loss']:.4f} -> {after_metrics['log_loss']:.4f}")
    logger.info(f"    ECE: {before_metrics['ece']:.4f} -> {after_metrics['ece']:.4f}")

    return {
        "dixon_coles": dc_model,
        "gbm_corrector": gbm_corrector,
        "calibrator": calibrator,
        "team_ratings": team_ratings,
        "feature_importance": importance
    }


def generate_predictions(
    model_components: dict,
    df: pd.DataFrame,
    feature_columns: list
) -> pd.DataFrame:
    """
    Generate predictions for a dataset using the trained model.
    """
    dc_model = model_components["dixon_coles"]
    gbm_corrector = model_components["gbm_corrector"]
    calibrator = model_components["calibrator"]

    from scipy.stats import poisson

    # Filter to rows with valid features
    valid_df = df.dropna(subset=["home_team", "away_team"]).copy()

    # Fill missing features with median (from training)
    for col in feature_columns:
        if col in valid_df.columns:
            valid_df[col] = valid_df[col].fillna(valid_df[col].median())

    # Step 1: Dixon-Coles base predictions
    result = dc_model.predict_batch(valid_df)

    # Step 2: Apply GBM corrections
    result = gbm_corrector.correct_predictions(result)

    # Step 3: Recalculate probabilities with corrected lambdas
    def lambda_to_probs(lambda_h, lambda_a, rho, max_goals=10):
        prob_h, prob_d, prob_a = 0.0, 0.0, 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = dc_model._tau(i, j, lambda_h, lambda_a, rho)
                p = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) * tau
                if i > j:
                    prob_h += p
                elif i == j:
                    prob_d += p
                else:
                    prob_a += p
        total = prob_h + prob_d + prob_a
        return prob_h/total, prob_d/total, prob_a/total

    probs_list = []
    for _, row in result.iterrows():
        ph, pd, pa = lambda_to_probs(
            row["lambda_home_corrected"],
            row["lambda_away_corrected"],
            dc_model.rho
        )
        probs_list.append([ph, pd, pa])

    probs_array = np.array(probs_list)

    # Step 4: Apply calibration
    calibrated_probs = calibrator.calibrate(probs_array)

    result["prob_home"] = calibrated_probs[:, 0]
    result["prob_draw"] = calibrated_probs[:, 1]
    result["prob_away"] = calibrated_probs[:, 2]

    return result


def evaluate_model(predictions: pd.DataFrame) -> dict:
    """
    Evaluate model prediction quality.
    """
    # Create labels
    labels = np.where(
        predictions["home_goals"] > predictions["away_goals"], 0,
        np.where(predictions["home_goals"] == predictions["away_goals"], 1, 2)
    )

    probs = np.column_stack([
        predictions["prob_home"],
        predictions["prob_draw"],
        predictions["prob_away"]
    ])

    # Calibration metrics
    cal_metrics = compute_calibration_metrics(probs, labels)

    # Accuracy
    predicted = np.argmax(probs, axis=1)
    accuracy = np.mean(predicted == labels)

    # Accuracy by outcome
    home_mask = labels == 0
    draw_mask = labels == 1
    away_mask = labels == 2

    home_accuracy = np.mean(predicted[home_mask] == 0) if home_mask.sum() > 0 else 0
    draw_accuracy = np.mean(predicted[draw_mask] == 1) if draw_mask.sum() > 0 else 0
    away_accuracy = np.mean(predicted[away_mask] == 2) if away_mask.sum() > 0 else 0

    return {
        "n_matches": len(predictions),
        "accuracy": accuracy,
        "home_accuracy": home_accuracy,
        "draw_accuracy": draw_accuracy,
        "away_accuracy": away_accuracy,
        "log_loss": cal_metrics["log_loss"],
        "brier_score": cal_metrics["brier_score"],
        "ece": cal_metrics["ece"]
    }


def run_backtest(predictions: pd.DataFrame, config: dict) -> tuple:
    """
    Run backtesting simulation on predictions.

    Returns:
        Tuple of (bets_df, summary_metrics)
    """
    logger.info("=" * 60)
    logger.info("BACKTESTING PHASE")
    logger.info("=" * 60)

    model_config = config["model"]

    backtester = Backtester(
        kelly_fraction=model_config["kelly_fraction"],
        max_bet_fraction=model_config["max_bet_pct"],
        min_edge=model_config["min_edge_threshold"],
        min_odds=1.30,
        max_odds=model_config.get("max_odds", 10.0),
        exclude_draws=model_config.get("exclude_draws", False),
        initial_bankroll=1.0
    )

    bets_df, summary = backtester.run_backtest(
        predictions,
        home_goals_col="home_goals",
        away_goals_col="away_goals",
        prob_home_col="prob_home",
        prob_draw_col="prob_draw",
        prob_away_col="prob_away",
        home_odds_col="home_odds",
        draw_odds_col="draw_odds",
        away_odds_col="away_odds"
    )

    logger.info(f"\nBacktest Results ({config['validation']['validation_season']}):")
    logger.info(f"  Total bets: {summary['n_bets']}")
    logger.info(f"  Total staked: {summary['total_staked']:.2%} of bankroll")
    logger.info(f"  Total profit: {summary['total_profit']:.2%} of bankroll")
    logger.info(f"  ROI: {summary['roi']:.2%}")
    logger.info(f"  Hit rate: {summary['hit_rate']:.2%}")
    logger.info(f"  Average odds: {summary['avg_odds']:.2f}")
    logger.info(f"  Average edge: {summary['avg_edge']:.2%}")
    logger.info(f"  Max drawdown: {summary['max_drawdown']:.2%}")
    logger.info(f"  Final bankroll: {summary['final_bankroll']:.2%}")

    # CLV analysis (key indicator of edge sustainability)
    if summary.get('bets_with_clv', 0) > 0:
        logger.info(f"\nClosing Line Value (CLV) Analysis:")
        logger.info(f"  Bets with CLV data: {summary['bets_with_clv']}")
        logger.info(f"  Average CLV: {summary['avg_clv']:.2%}")
        logger.info(f"  CLV+ rate: {summary['clv_positive_pct']:.1%} (bets beating closing line)")

    # Analyze by bet type
    if len(bets_df) > 0:
        by_type = backtester.analyze_by_bet_type(bets_df)
        logger.info("\nPerformance by bet type:")
        for _, row in by_type.iterrows():
            logger.info(
                f"  {row['bet_outcome']}: {row['n_bets']} bets, "
                f"ROI={row['roi']:.2%}, hit_rate={row['hit_rate']:.2%}"
            )

        # Monthly analysis
        by_month = backtester.analyze_by_period(bets_df, period="M")
        logger.info("\nMonthly performance:")
        for _, row in by_month.iterrows():
            logger.info(
                f"  {row['period']}: {row['n_bets']} bets, "
                f"P/L={row['profit']:.4f}, ROI={row['roi']:.2%}"
            )

        # Confidence intervals
        ci = backtester.calculate_confidence_interval(bets_df, n_simulations=1000)
        logger.info(f"\n95% Confidence Interval for ROI:")
        logger.info(f"  Lower: {ci['roi_lower']:.2%}")
        logger.info(f"  Mean: {ci['roi_mean']:.2%}")
        logger.info(f"  Upper: {ci['roi_upper']:.2%}")
        logger.info(f"  P(profitable): {ci['prob_profitable']:.2%}")

    return bets_df, summary


def save_results(
    model_components: dict,
    predictions: pd.DataFrame,
    bets_df: pd.DataFrame,
    summary: dict,
    eval_metrics: dict,
    config: dict,
    output_dir: str = "models"
):
    """
    Save all results to the models directory.
    """
    logger.info("=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model components
    with open(output_path / "dixon_coles.pkl", "wb") as f:
        pickle.dump(model_components["dixon_coles"], f)

    with open(output_path / "gbm_corrector.pkl", "wb") as f:
        pickle.dump(model_components["gbm_corrector"], f)

    with open(output_path / "calibrator.pkl", "wb") as f:
        pickle.dump(model_components["calibrator"], f)

    logger.info(f"  Model components saved to {output_path}")

    # Save team ratings
    model_components["team_ratings"].to_csv(
        output_path / "team_ratings.csv", index=False
    )

    # Save feature importance
    model_components["feature_importance"].to_csv(
        output_path / "feature_importance.csv", index=False
    )

    # Save predictions
    predictions.to_csv(output_path / "validation_predictions.csv", index=False)

    # Save bets
    if len(bets_df) > 0:
        bets_df.to_csv(output_path / "backtest_bets.csv", index=False)

    # Save summary report
    report = {
        "timestamp": timestamp,
        "config": config,
        "evaluation_metrics": eval_metrics,
        "backtest_summary": summary,
        "model_params": {
            "home_advantage": float(model_components["dixon_coles"].home_advantage),
            "rho": float(model_components["dixon_coles"].rho),
            "temperature": float(model_components["calibrator"].temperature),
            "n_teams": len(model_components["dixon_coles"].teams)
        }
    }

    with open(output_path / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"  Report saved to {output_path / 'report.json'}")

    # Generate summary text report
    summary_text = f"""
================================================================================
NOSTRUS BETTING MODEL - VALIDATION REPORT
================================================================================

Run timestamp: {timestamp}
Validation season: {config['validation']['validation_season']}

MODEL PARAMETERS
----------------
Dixon-Coles:
  - Home advantage: {model_components['dixon_coles'].home_advantage:.4f}
  - Rho (correlation): {model_components['dixon_coles'].rho:.4f}
  - Teams trained: {len(model_components['dixon_coles'].teams)}

Calibration:
  - Temperature: {model_components['calibrator'].temperature:.4f}

Kelly Strategy:
  - Kelly fraction: {config['model']['kelly_fraction']}
  - Min edge threshold: {config['model']['min_edge_threshold']:.1%}
  - Max bet size: {config['model']['max_bet_pct']:.1%}

PREDICTION METRICS
------------------
Matches evaluated: {eval_metrics['n_matches']}
Overall accuracy: {eval_metrics['accuracy']:.2%}
  - Home win accuracy: {eval_metrics['home_accuracy']:.2%}
  - Draw accuracy: {eval_metrics['draw_accuracy']:.2%}
  - Away win accuracy: {eval_metrics['away_accuracy']:.2%}
Log-loss: {eval_metrics['log_loss']:.4f}
Brier score: {eval_metrics['brier_score']:.4f}
ECE (calibration): {eval_metrics['ece']:.4f}

BACKTEST RESULTS
----------------
Total bets placed: {summary['n_bets']}
Total staked: {summary['total_staked']:.2%} of bankroll
Total profit/loss: {summary['total_profit']:.2%} of bankroll
ROI: {summary['roi']:.2%}
Hit rate: {summary['hit_rate']:.2%}
Average odds: {summary['avg_odds']:.2f}
Average edge: {summary['avg_edge']:.2%}
Maximum drawdown: {summary['max_drawdown']:.2%}
Final bankroll: {summary['final_bankroll']:.2%}
Sharpe ratio (annualized): {summary['sharpe_ratio']:.2f}

CLOSING LINE VALUE (CLV) ANALYSIS
---------------------------------
CLV measures whether bets beat Pinnacle's closing line (sharp odds).
Positive CLV indicates genuine predictive edge, not just luck.

Bets with CLV data: {summary.get('bets_with_clv', 0)}
Average CLV: {summary.get('avg_clv', 0):.2%}
CLV+ rate: {summary.get('clv_positive_pct', 0):.1%} (bets beating closing line)

TOP 10 TEAMS BY OVERALL RATING
------------------------------
"""

    for i, (_, row) in enumerate(model_components["team_ratings"].head(10).iterrows()):
        summary_text += f"{i+1:2}. {row['team']:25} (attack={row['attack']:+.3f}, defense={row['defense']:+.3f})\n"

    summary_text += "\n" + "=" * 80

    with open(output_path / "summary.txt", "w") as f:
        f.write(summary_text)

    logger.info(f"  Summary saved to {output_path / 'summary.txt'}")

    return output_path


def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("NOSTRUS BETTING MODEL PIPELINE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()

    # Prepare data
    train_df, val_df, feature_columns = prepare_data(config)

    # Train model
    model_components = train_model(train_df, feature_columns, config)

    # Generate predictions on validation set
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION PHASE")
    logger.info("=" * 60)

    predictions = generate_predictions(model_components, val_df, feature_columns)
    logger.info(f"Generated predictions for {len(predictions)} matches")

    # Evaluate prediction quality
    eval_metrics = evaluate_model(predictions)

    logger.info(f"\nModel Evaluation:")
    logger.info(f"  Accuracy: {eval_metrics['accuracy']:.2%}")
    logger.info(f"  Log-loss: {eval_metrics['log_loss']:.4f}")
    logger.info(f"  Brier score: {eval_metrics['brier_score']:.4f}")
    logger.info(f"  ECE: {eval_metrics['ece']:.4f}")

    # Run backtest
    bets_df, summary = run_backtest(predictions, config)

    # Save results
    output_path = save_results(
        model_components,
        predictions,
        bets_df,
        summary,
        eval_metrics,
        config
    )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    main()
