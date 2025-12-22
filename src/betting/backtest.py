"""
Backtesting framework for Nostrus.

Simulates betting strategy performance using historical data
with walk-forward validation.
"""

import logging
from typing import Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from .kelly import KellyCalculator

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine for betting strategies.

    Simulates historical performance using walk-forward methodology
    to avoid lookahead bias.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.02,
        min_edge: float = 0.03,
        min_odds: float = 1.30,
        max_odds: float = 10.0,
        exclude_draws: bool = False,
        initial_bankroll: float = 1.0
    ):
        """
        Initialize backtester.

        Args:
            kelly_fraction: Fraction of Kelly to use
            max_bet_fraction: Maximum bet size
            min_edge: Minimum required edge
            min_odds: Minimum acceptable odds
            max_odds: Maximum acceptable odds (avoid longshots)
            exclude_draws: If True, skip all draw bets
            initial_bankroll: Starting bankroll (1.0 = 100%)
        """
        self.kelly = KellyCalculator(
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
            min_edge=min_edge,
            min_odds=min_odds
        )
        self.max_odds = max_odds
        self.exclude_draws = exclude_draws
        self.initial_bankroll = initial_bankroll

    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        prob_home_col: str = "prob_home",
        prob_draw_col: str = "prob_draw",
        prob_away_col: str = "prob_away",
        home_odds_col: str = "home_odds",
        draw_odds_col: str = "draw_odds",
        away_odds_col: str = "away_odds",
        pinnacle_home_col: str = "pinnacle_home_odds",
        pinnacle_draw_col: str = "pinnacle_draw_odds",
        pinnacle_away_col: str = "pinnacle_away_odds"
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Run backtest on historical predictions with CLV tracking.

        Args:
            predictions_df: DataFrame with predictions, odds, and actual results
            *_col: Column name mappings
            pinnacle_*_col: Pinnacle closing odds columns for CLV calculation

        Returns:
            Tuple of (bets DataFrame, summary metrics)
        """
        df = predictions_df.sort_values("date").reset_index(drop=True)

        bets = []
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll

        # Map outcome to Pinnacle odds columns for CLV calculation
        pinnacle_cols = {"H": pinnacle_home_col, "D": pinnacle_draw_col, "A": pinnacle_away_col}

        for idx, row in df.iterrows():
            # Determine actual outcome
            home_goals = row[home_goals_col]
            away_goals = row[away_goals_col]

            if pd.isna(home_goals) or pd.isna(away_goals):
                continue

            if home_goals > away_goals:
                actual_outcome = "H"
            elif home_goals == away_goals:
                actual_outcome = "D"
            else:
                actual_outcome = "A"

            # Check each possible bet
            for outcome, prob_col, odds_col in [
                ("H", prob_home_col, home_odds_col),
                ("D", prob_draw_col, draw_odds_col),
                ("A", prob_away_col, away_odds_col)
            ]:
                # Skip draw bets if configured
                if self.exclude_draws and outcome == "D":
                    continue

                if pd.isna(row.get(odds_col)):
                    continue

                model_prob = row[prob_col]
                odds = row[odds_col]

                # Skip if odds exceed maximum (avoid longshots)
                if odds > self.max_odds:
                    continue

                stake, metadata = self.kelly.calculate_stake(model_prob, odds, bankroll)

                if not metadata["should_bet"]:
                    continue

                # Record bet
                won = (outcome == actual_outcome)
                profit = stake * (odds - 1) if won else -stake
                bankroll += profit

                # Track drawdown
                peak_bankroll = max(peak_bankroll, bankroll)
                drawdown = (peak_bankroll - bankroll) / peak_bankroll

                # Calculate CLV (Closing Line Value)
                # CLV = (bet_odds / pinnacle_closing_odds) - 1
                # Positive CLV means we got better odds than Pinnacle closing
                pinnacle_odds = row.get(pinnacle_cols[outcome])
                if pd.notna(pinnacle_odds) and pinnacle_odds > 0:
                    clv = (odds / pinnacle_odds) - 1
                else:
                    clv = None

                bets.append({
                    "date": row.get("date"),
                    "match_id": row.get("match_id"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "bet_outcome": outcome,
                    "actual_outcome": actual_outcome,
                    "model_prob": model_prob,
                    "implied_prob": 1 / odds,
                    "odds": odds,
                    "pinnacle_closing_odds": pinnacle_odds,
                    "clv": clv,
                    "edge": metadata["edge"],
                    "stake": stake,
                    "won": won,
                    "profit": profit,
                    "bankroll_after": bankroll,
                    "drawdown": drawdown
                })

        bets_df = pd.DataFrame(bets)

        # Calculate summary metrics
        summary = self._calculate_summary(bets_df)

        return bets_df, summary

    def _calculate_summary(self, bets_df: pd.DataFrame) -> dict:
        """
        Calculate summary statistics from bet history.

        Args:
            bets_df: DataFrame with bet history

        Returns:
            Dictionary with summary metrics
        """
        if len(bets_df) == 0:
            return {
                "n_bets": 0,
                "total_staked": 0,
                "total_profit": 0,
                "roi": 0,
                "hit_rate": 0,
                "final_bankroll": self.initial_bankroll,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "avg_odds": 0,
                "avg_edge": 0,
                "avg_clv": 0,
                "clv_positive_pct": 0,
                "bets_with_clv": 0
            }

        n_bets = len(bets_df)
        total_staked = bets_df["stake"].sum()
        total_profit = bets_df["profit"].sum()
        hit_rate = bets_df["won"].mean()
        final_bankroll = bets_df["bankroll_after"].iloc[-1]
        max_drawdown = bets_df["drawdown"].max()

        # ROI
        roi = total_profit / total_staked if total_staked > 0 else 0

        # Sharpe ratio (annualized, assuming ~1000 bets/year)
        returns = bets_df["profit"] / bets_df["stake"]
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(1000)
        else:
            sharpe = 0

        # Average odds and edge
        avg_odds = bets_df["odds"].mean()
        avg_edge = bets_df["edge"].mean()

        # CLV (Closing Line Value) analysis
        # Positive CLV = got better odds than Pinnacle closing (good sign)
        clv_data = bets_df["clv"].dropna()
        if len(clv_data) > 0:
            avg_clv = clv_data.mean()
            clv_positive_pct = (clv_data > 0).mean()
            bets_with_clv = len(clv_data)
        else:
            avg_clv = 0
            clv_positive_pct = 0
            bets_with_clv = 0

        return {
            "n_bets": n_bets,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": roi,
            "hit_rate": hit_rate,
            "final_bankroll": final_bankroll,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "avg_odds": avg_odds,
            "avg_edge": avg_edge,
            "avg_clv": avg_clv,
            "clv_positive_pct": clv_positive_pct,
            "bets_with_clv": bets_with_clv
        }

    def analyze_by_period(
        self,
        bets_df: pd.DataFrame,
        period: str = "M"
    ) -> pd.DataFrame:
        """
        Analyze performance by time period.

        Args:
            bets_df: DataFrame with bet history
            period: Pandas period string (D, W, M, Q, Y)

        Returns:
            DataFrame with period-level summary
        """
        if len(bets_df) == 0:
            return pd.DataFrame()

        bets_df = bets_df.copy()
        bets_df["period"] = pd.to_datetime(bets_df["date"]).dt.to_period(period)

        summary = bets_df.groupby("period").agg({
            "stake": "sum",
            "profit": "sum",
            "won": ["sum", "count"],
            "edge": "mean",
            "odds": "mean"
        })

        summary.columns = ["total_staked", "profit", "wins", "n_bets", "avg_edge", "avg_odds"]
        summary["roi"] = summary["profit"] / summary["total_staked"]
        summary["hit_rate"] = summary["wins"] / summary["n_bets"]

        return summary.reset_index()

    def analyze_by_league(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze performance by league/competition.

        Args:
            bets_df: DataFrame with bet history

        Returns:
            DataFrame with league-level summary
        """
        if len(bets_df) == 0 or "competition" not in bets_df.columns:
            return pd.DataFrame()

        summary = bets_df.groupby("competition").agg({
            "stake": "sum",
            "profit": "sum",
            "won": ["sum", "count"],
            "edge": "mean"
        })

        summary.columns = ["total_staked", "profit", "wins", "n_bets", "avg_edge"]
        summary["roi"] = summary["profit"] / summary["total_staked"]
        summary["hit_rate"] = summary["wins"] / summary["n_bets"]

        return summary.reset_index().sort_values("profit", ascending=False)

    def analyze_by_bet_type(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze performance by bet type (H/D/A).

        Args:
            bets_df: DataFrame with bet history

        Returns:
            DataFrame with bet-type summary
        """
        if len(bets_df) == 0:
            return pd.DataFrame()

        summary = bets_df.groupby("bet_outcome").agg({
            "stake": "sum",
            "profit": "sum",
            "won": ["sum", "count"],
            "edge": "mean",
            "odds": "mean"
        })

        summary.columns = ["total_staked", "profit", "wins", "n_bets", "avg_edge", "avg_odds"]
        summary["roi"] = summary["profit"] / summary["total_staked"]
        summary["hit_rate"] = summary["wins"] / summary["n_bets"]

        return summary.reset_index()

    def plot_equity_curve(self, bets_df: pd.DataFrame):
        """
        Generate equity curve plot data.

        Args:
            bets_df: DataFrame with bet history

        Returns:
            DataFrame with date and cumulative bankroll
        """
        if len(bets_df) == 0:
            return pd.DataFrame()

        equity = bets_df[["date", "bankroll_after"]].copy()
        equity = equity.rename(columns={"bankroll_after": "bankroll"})

        # Add starting point
        start = pd.DataFrame([{
            "date": bets_df["date"].min(),
            "bankroll": self.initial_bankroll
        }])

        equity = pd.concat([start, equity]).reset_index(drop=True)

        return equity

    def calculate_confidence_interval(
        self,
        bets_df: pd.DataFrame,
        n_simulations: int = 1000,
        confidence: float = 0.95
    ) -> dict:
        """
        Calculate confidence intervals via bootstrap.

        Args:
            bets_df: DataFrame with bet history
            n_simulations: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Dictionary with confidence intervals
        """
        if len(bets_df) == 0:
            return {}

        rois = []

        for _ in range(n_simulations):
            sample = bets_df.sample(n=len(bets_df), replace=True)
            roi = sample["profit"].sum() / sample["stake"].sum()
            rois.append(roi)

        rois = np.array(rois)
        alpha = (1 - confidence) / 2

        return {
            "roi_mean": np.mean(rois),
            "roi_median": np.median(rois),
            "roi_lower": np.percentile(rois, alpha * 100),
            "roi_upper": np.percentile(rois, (1 - alpha) * 100),
            "prob_profitable": np.mean(rois > 0)
        }
