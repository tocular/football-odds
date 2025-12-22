"""
Kelly criterion and bet sizing module for Nostrus.

Implements fractional Kelly staking for bankroll management.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KellyCalculator:
    """
    Kelly criterion calculator for optimal bet sizing.

    Uses fractional Kelly to reduce variance while maintaining
    positive expected growth.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.02,
        min_edge: float = 0.03,
        min_odds: float = 1.30
    ):
        """
        Initialize Kelly calculator.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_bet_fraction: Maximum bet as fraction of bankroll
            min_edge: Minimum edge required to bet
            min_odds: Minimum decimal odds to consider
        """
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.min_odds = min_odds

    def calculate_edge(
        self,
        model_prob: float,
        odds: float
    ) -> float:
        """
        Calculate edge (model probability - implied probability).

        Args:
            model_prob: Model's predicted probability
            odds: Decimal odds

        Returns:
            Edge as decimal (e.g., 0.05 for 5%)
        """
        implied_prob = 1 / odds
        return model_prob - implied_prob

    def calculate_kelly_fraction(
        self,
        model_prob: float,
        odds: float
    ) -> float:
        """
        Calculate full Kelly stake as fraction of bankroll.

        Formula: f* = (bp - q) / b
        where:
            b = odds - 1 (net payout per unit)
            p = probability of winning
            q = probability of losing = 1 - p

        Args:
            model_prob: Model's predicted probability
            odds: Decimal odds

        Returns:
            Kelly fraction (can be negative if no edge)
        """
        b = odds - 1  # Net odds
        p = model_prob
        q = 1 - p

        kelly = (b * p - q) / b

        return kelly

    def calculate_stake(
        self,
        model_prob: float,
        odds: float,
        bankroll: float = 1.0
    ) -> Tuple[float, dict]:
        """
        Calculate recommended stake amount.

        Args:
            model_prob: Model's predicted probability
            odds: Decimal odds
            bankroll: Current bankroll (default 1.0 for fraction)

        Returns:
            Tuple of (stake amount, metadata dict)
        """
        edge = self.calculate_edge(model_prob, odds)

        metadata = {
            "model_prob": model_prob,
            "implied_prob": 1 / odds,
            "odds": odds,
            "edge": edge,
            "full_kelly": 0,
            "fractional_kelly": 0,
            "capped_stake": 0,
            "should_bet": False,
            "skip_reason": None
        }

        # Check minimum odds
        if odds < self.min_odds:
            metadata["skip_reason"] = f"Odds {odds:.2f} below minimum {self.min_odds}"
            return 0, metadata

        # Check minimum edge
        if edge < self.min_edge:
            metadata["skip_reason"] = f"Edge {edge:.1%} below minimum {self.min_edge:.1%}"
            return 0, metadata

        # Calculate Kelly
        full_kelly = self.calculate_kelly_fraction(model_prob, odds)
        metadata["full_kelly"] = full_kelly

        if full_kelly <= 0:
            metadata["skip_reason"] = "Negative Kelly (no edge)"
            return 0, metadata

        # Apply fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction
        metadata["fractional_kelly"] = fractional_kelly

        # Apply max bet cap
        capped_stake = min(fractional_kelly, self.max_bet_fraction) * bankroll
        metadata["capped_stake"] = capped_stake
        metadata["should_bet"] = True

        return capped_stake, metadata

    def find_value_bets(
        self,
        predictions_df,
        prob_home_col: str = "prob_home",
        prob_draw_col: str = "prob_draw",
        prob_away_col: str = "prob_away",
        home_odds_col: str = "home_odds",
        draw_odds_col: str = "draw_odds",
        away_odds_col: str = "away_odds",
        bankroll: float = 1.0
    ):
        """
        Find value bets in a DataFrame of predictions.

        Args:
            predictions_df: DataFrame with predictions and odds
            prob_*_col: Column names for probabilities
            *_odds_col: Column names for odds
            bankroll: Current bankroll

        Returns:
            DataFrame with value bets and recommended stakes
        """
        import pandas as pd

        value_bets = []

        for idx, row in predictions_df.iterrows():
            for outcome, prob_col, odds_col in [
                ("H", prob_home_col, home_odds_col),
                ("D", prob_draw_col, draw_odds_col),
                ("A", prob_away_col, away_odds_col)
            ]:
                if pd.isna(row.get(odds_col)):
                    continue

                model_prob = row[prob_col]
                odds = row[odds_col]

                stake, metadata = self.calculate_stake(model_prob, odds, bankroll)

                if metadata["should_bet"]:
                    value_bets.append({
                        "match_id": row.get("match_id"),
                        "date": row.get("date"),
                        "home_team": row.get("home_team"),
                        "away_team": row.get("away_team"),
                        "outcome": outcome,
                        "model_prob": model_prob,
                        "odds": odds,
                        "implied_prob": metadata["implied_prob"],
                        "edge": metadata["edge"],
                        "full_kelly": metadata["full_kelly"],
                        "stake": stake,
                        "stake_pct": stake / bankroll if bankroll > 0 else 0
                    })

        return pd.DataFrame(value_bets)


def calculate_expected_value(
    model_prob: float,
    odds: float,
    stake: float = 1.0
) -> float:
    """
    Calculate expected value of a bet.

    Args:
        model_prob: Probability of winning
        odds: Decimal odds
        stake: Stake amount

    Returns:
        Expected value (positive = profitable)
    """
    return stake * (model_prob * (odds - 1) - (1 - model_prob))


def calculate_expected_growth(
    model_prob: float,
    odds: float,
    kelly_fraction: float
) -> float:
    """
    Calculate expected log growth rate from Kelly betting.

    Args:
        model_prob: Probability of winning
        odds: Decimal odds
        kelly_fraction: Stake as fraction of bankroll

    Returns:
        Expected log growth rate
    """
    p = model_prob
    q = 1 - p
    b = odds - 1
    f = kelly_fraction

    # E[log(wealth)] = p * log(1 + f*b) + q * log(1 - f)
    growth = p * np.log(1 + f * b) + q * np.log(1 - f)

    return growth
