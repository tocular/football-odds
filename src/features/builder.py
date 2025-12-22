"""
Feature engineering module for Nostrus.

Builds features for match prediction including team strength,
form, situational factors, and injury adjustments.
"""

import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import func

from ..data.db import Database, Match, Team, EloRating, TeamStats, MatchOdds

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds feature matrices for match prediction.

    Features are computed using only data available before each match
    to prevent lookahead bias.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the feature builder.

        Args:
            config_path: Path to configuration file
        """
        self.db = Database(config_path)

    def get_matches_df(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        leagues: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get matches DataFrame with basic info.

        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            leagues: List of leagues to include

        Returns:
            DataFrame with match data
        """
        session = self.db.get_session()

        try:
            # Join matches with teams
            query = session.query(
                Match.match_id,
                Match.date,
                Match.season,
                Match.competition,
                Match.matchweek,
                Match.home_team_id,
                Match.away_team_id,
                Match.home_goals,
                Match.away_goals,
                Match.home_xg,
                Match.away_xg,
            ).filter(Match.status == "completed")

            if start_date:
                query = query.filter(Match.date >= start_date)
            if end_date:
                query = query.filter(Match.date <= end_date)
            if leagues:
                query = query.filter(Match.competition.in_(leagues))

            df = pd.read_sql(query.statement, session.bind)

            # Get team names
            teams = pd.read_sql(
                session.query(Team.team_id, Team.canonical_name).statement,
                session.bind
            )

            df = df.merge(
                teams.rename(columns={"team_id": "home_team_id", "canonical_name": "home_team"}),
                on="home_team_id"
            )
            df = df.merge(
                teams.rename(columns={"team_id": "away_team_id", "canonical_name": "away_team"}),
                on="away_team_id"
            )

            return df.sort_values("date").reset_index(drop=True)

        finally:
            session.close()

    def compute_elo_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Elo-based features for each match.

        Features:
        - home_elo: Home team Elo rating before match
        - away_elo: Away team Elo rating before match
        - elo_diff: home_elo - away_elo
        - elo_home_prob: Implied home win probability from Elo

        Args:
            matches_df: DataFrame with match data

        Returns:
            DataFrame with Elo features added
        """
        session = self.db.get_session()
        df = matches_df.copy()

        try:
            # Get all Elo ratings
            elo_df = pd.read_sql(
                session.query(
                    EloRating.team_id,
                    EloRating.date,
                    EloRating.elo_rating
                ).statement,
                session.bind
            )
            elo_df["date"] = pd.to_datetime(elo_df["date"])

            # For each match, get the most recent Elo rating before match date
            home_elos = []
            away_elos = []

            for _, row in df.iterrows():
                match_date = pd.to_datetime(row["date"])

                # Home team Elo
                home_elo_data = elo_df[
                    (elo_df["team_id"] == row["home_team_id"]) &
                    (elo_df["date"] < match_date)
                ].sort_values("date", ascending=False)

                home_elo = home_elo_data["elo_rating"].iloc[0] if len(home_elo_data) > 0 else 1500

                # Away team Elo
                away_elo_data = elo_df[
                    (elo_df["team_id"] == row["away_team_id"]) &
                    (elo_df["date"] < match_date)
                ].sort_values("date", ascending=False)

                away_elo = away_elo_data["elo_rating"].iloc[0] if len(away_elo_data) > 0 else 1500

                home_elos.append(home_elo)
                away_elos.append(away_elo)

            df["home_elo"] = home_elos
            df["away_elo"] = away_elos
            df["elo_diff"] = df["home_elo"] - df["away_elo"]

            # Elo-implied probability (with ~65 point home advantage)
            home_advantage = 65
            df["elo_home_prob"] = 1 / (
                1 + 10 ** ((df["away_elo"] - df["home_elo"] - home_advantage) / 400)
            )

            return df

        finally:
            session.close()

    def compute_form_features(
        self,
        matches_df: pd.DataFrame,
        windows: list = [5, 10]
    ) -> pd.DataFrame:
        """
        Compute rolling form features.

        Features (for each window):
        - xg_for_l{n}: Average xG scored in last n matches
        - xg_against_l{n}: Average xG conceded in last n matches
        - goals_for_l{n}: Average goals scored
        - goals_against_l{n}: Average goals conceded
        - points_l{n}: Average points per game

        Args:
            matches_df: DataFrame with match data
            windows: Rolling window sizes

        Returns:
            DataFrame with form features added
        """
        df = matches_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Create team-level match history
        # Each match appears twice: once for home team, once for away team
        home_matches = df[["date", "home_team_id", "home_goals", "away_goals", "home_xg", "away_xg"]].copy()
        home_matches.columns = ["date", "team_id", "goals_for", "goals_against", "xg_for", "xg_against"]
        home_matches["is_home"] = 1
        home_matches["points"] = np.where(
            home_matches["goals_for"] > home_matches["goals_against"], 3,
            np.where(home_matches["goals_for"] == home_matches["goals_against"], 1, 0)
        )

        away_matches = df[["date", "away_team_id", "away_goals", "home_goals", "away_xg", "home_xg"]].copy()
        away_matches.columns = ["date", "team_id", "goals_for", "goals_against", "xg_for", "xg_against"]
        away_matches["is_home"] = 0
        away_matches["points"] = np.where(
            away_matches["goals_for"] > away_matches["goals_against"], 3,
            np.where(away_matches["goals_for"] == away_matches["goals_against"], 1, 0)
        )

        all_matches = pd.concat([home_matches, away_matches]).sort_values(["team_id", "date"])

        # Compute rolling stats for each team
        for window in windows:
            # Group by team and compute rolling means (shifted to avoid lookahead)
            for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points"]:
                all_matches[f"{col}_l{window}"] = (
                    all_matches.groupby("team_id")[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

        # Merge back to original matches
        for prefix, team_col in [("home", "home_team_id"), ("away", "away_team_id")]:
            team_form = all_matches[all_matches["is_home"] == (1 if prefix == "home" else 0)].copy()

            # Select relevant columns
            form_cols = [c for c in team_form.columns if c.endswith(tuple(f"_l{w}" for w in windows))]
            team_form = team_form[["date", "team_id"] + form_cols]

            # Rename columns
            rename_map = {c: f"{prefix}_{c}" for c in form_cols}
            rename_map["team_id"] = team_col
            team_form = team_form.rename(columns=rename_map)

            df = df.merge(team_form, on=["date", team_col], how="left")

        return df

    def compute_head_to_head_features(self, matches_df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Compute head-to-head features.

        Features:
        - h2h_home_wins: Home team wins in recent H2H matches
        - h2h_draws: Draws in recent H2H matches
        - h2h_away_wins: Away team wins in recent H2H matches
        - h2h_total_goals: Average total goals in H2H matches

        Args:
            matches_df: DataFrame with match data
            lookback: Number of past H2H matches to consider

        Returns:
            DataFrame with H2H features added
        """
        df = matches_df.copy()
        # Convert date column to datetime for consistent comparisons
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        h2h_features = []

        for idx, row in df.iterrows():
            match_date = row["date"]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            # Find past H2H matches (either team as home or away)
            past_h2h = df[
                (df["date"] < match_date) &
                (
                    ((df["home_team_id"] == home_id) & (df["away_team_id"] == away_id)) |
                    ((df["home_team_id"] == away_id) & (df["away_team_id"] == home_id))
                )
            ].tail(lookback)

            if len(past_h2h) == 0:
                h2h_features.append({
                    "h2h_home_wins": 0.33,
                    "h2h_draws": 0.33,
                    "h2h_away_wins": 0.33,
                    "h2h_total_goals": 2.5,
                    "h2h_matches": 0
                })
            else:
                # Count outcomes (from perspective of current home team)
                home_wins = 0
                away_wins = 0
                draws = 0
                total_goals = []

                for _, h2h_row in past_h2h.iterrows():
                    if h2h_row["home_goals"] == h2h_row["away_goals"]:
                        draws += 1
                    elif h2h_row["home_team_id"] == home_id:
                        if h2h_row["home_goals"] > h2h_row["away_goals"]:
                            home_wins += 1
                        else:
                            away_wins += 1
                    else:
                        if h2h_row["away_goals"] > h2h_row["home_goals"]:
                            home_wins += 1
                        else:
                            away_wins += 1

                    total_goals.append(h2h_row["home_goals"] + h2h_row["away_goals"])

                n = len(past_h2h)
                h2h_features.append({
                    "h2h_home_wins": home_wins / n,
                    "h2h_draws": draws / n,
                    "h2h_away_wins": away_wins / n,
                    "h2h_total_goals": np.mean(total_goals),
                    "h2h_matches": n
                })

        h2h_df = pd.DataFrame(h2h_features)
        return pd.concat([df.reset_index(drop=True), h2h_df], axis=1)

    def compute_rest_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rest and fixture congestion features.

        Features:
        - home_rest_days: Days since home team's last match
        - away_rest_days: Days since away team's last match
        - rest_diff: home_rest_days - away_rest_days
        - home_matches_14d: Home team matches in last 14 days
        - away_matches_14d: Away team matches in last 14 days

        Args:
            matches_df: DataFrame with match data

        Returns:
            DataFrame with rest features added
        """
        df = matches_df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        # Build team match history
        team_dates = {}

        home_rest = []
        away_rest = []
        home_14d = []
        away_14d = []

        for idx, row in df.iterrows():
            match_date = row["date"]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            # Get past matches for each team
            home_history = team_dates.get(home_id, [])
            away_history = team_dates.get(away_id, [])

            # Rest days
            if len(home_history) > 0:
                home_rest.append((match_date - home_history[-1]).days)
            else:
                home_rest.append(7)  # Default

            if len(away_history) > 0:
                away_rest.append((match_date - away_history[-1]).days)
            else:
                away_rest.append(7)

            # Matches in last 14 days
            cutoff = match_date - timedelta(days=14)
            home_14d.append(sum(1 for d in home_history if d >= cutoff))
            away_14d.append(sum(1 for d in away_history if d >= cutoff))

            # Update history
            if home_id not in team_dates:
                team_dates[home_id] = []
            if away_id not in team_dates:
                team_dates[away_id] = []

            team_dates[home_id].append(match_date)
            team_dates[away_id].append(match_date)

        df["home_rest_days"] = home_rest
        df["away_rest_days"] = away_rest
        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
        df["home_matches_14d"] = home_14d
        df["away_matches_14d"] = away_14d
        df["congestion_diff"] = df["away_matches_14d"] - df["home_matches_14d"]

        return df

    def compute_odds_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market-implied probability features from Pinnacle closing odds.

        Features:
        - market_home_prob: Implied probability from Pinnacle odds (vig-adjusted)
        - market_draw_prob: Implied probability from Pinnacle odds (vig-adjusted)
        - market_away_prob: Implied probability from Pinnacle odds (vig-adjusted)
        - home_odds: Pinnacle closing home odds (for backtest)
        - draw_odds: Pinnacle closing draw odds (for backtest)
        - away_odds: Pinnacle closing away odds (for backtest)
        - pinnacle_home_odds: Pinnacle closing odds for CLV calculation
        - pinnacle_draw_odds: Pinnacle closing odds for CLV calculation
        - pinnacle_away_odds: Pinnacle closing odds for CLV calculation

        Args:
            matches_df: DataFrame with match data

        Returns:
            DataFrame with market features added
        """
        session = self.db.get_session()
        df = matches_df.copy()

        try:
            # Get all odds data (prioritize Pinnacle closing odds)
            odds_df = pd.read_sql(
                session.query(
                    MatchOdds.match_id,
                    MatchOdds.bookmaker,
                    MatchOdds.home_odds,
                    MatchOdds.draw_odds,
                    MatchOdds.away_odds,
                    MatchOdds.is_closing
                ).filter(MatchOdds.bookmaker.in_(["pinnacle", "bet365", "avg"])).statement,
                session.bind
            )

            def get_odds_for_match(match_id):
                """Get Pinnacle closing odds, falling back to Bet365/avg if needed."""
                match_odds = odds_df[odds_df["match_id"] == match_id]

                # First priority: Pinnacle closing odds
                pinnacle_closing = match_odds[
                    (match_odds["bookmaker"] == "pinnacle") &
                    (match_odds["is_closing"] == True)
                ]
                if len(pinnacle_closing) > 0:
                    return pinnacle_closing.iloc[0], "pinnacle_closing"

                # Second priority: Pinnacle (any)
                pinnacle = match_odds[match_odds["bookmaker"] == "pinnacle"]
                if len(pinnacle) > 0:
                    return pinnacle.iloc[0], "pinnacle"

                # Third priority: Bet365
                bet365 = match_odds[match_odds["bookmaker"] == "bet365"]
                if len(bet365) > 0:
                    return bet365.iloc[0], "bet365"

                # Last resort: average
                avg = match_odds[match_odds["bookmaker"] == "avg"]
                if len(avg) > 0:
                    return avg.iloc[0], "avg"

                return None, None

            market_probs = []
            for _, row in df.iterrows():
                odds, source = get_odds_for_match(row["match_id"])
                if odds is not None:
                    # Remove vig and compute implied probabilities
                    raw_probs = [
                        1 / odds["home_odds"],
                        1 / odds["draw_odds"],
                        1 / odds["away_odds"]
                    ]
                    total = sum(raw_probs)
                    market_probs.append({
                        "market_home_prob": raw_probs[0] / total,
                        "market_draw_prob": raw_probs[1] / total,
                        "market_away_prob": raw_probs[2] / total,
                        "home_odds": odds["home_odds"],
                        "draw_odds": odds["draw_odds"],
                        "away_odds": odds["away_odds"],
                        "pinnacle_home_odds": odds["home_odds"] if "pinnacle" in (source or "") else None,
                        "pinnacle_draw_odds": odds["draw_odds"] if "pinnacle" in (source or "") else None,
                        "pinnacle_away_odds": odds["away_odds"] if "pinnacle" in (source or "") else None,
                        "odds_source": source
                    })
                else:
                    market_probs.append({
                        "market_home_prob": None,
                        "market_draw_prob": None,
                        "market_away_prob": None,
                        "home_odds": None,
                        "draw_odds": None,
                        "away_odds": None,
                        "pinnacle_home_odds": None,
                        "pinnacle_draw_odds": None,
                        "pinnacle_away_odds": None,
                        "odds_source": None
                    })

            market_df = pd.DataFrame(market_probs)
            return pd.concat([df.reset_index(drop=True), market_df], axis=1)

        finally:
            session.close()

    def build_feature_matrix(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        leagues: Optional[list] = None,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for modeling.

        Args:
            start_date: Start date filter
            end_date: End date filter
            leagues: List of leagues to include
            include_target: Whether to include target variables

        Returns:
            DataFrame with all features
        """
        logger.info("Building feature matrix...")

        # Get base matches
        df = self.get_matches_df(start_date, end_date, leagues)
        logger.info(f"Loaded {len(df)} matches")

        if df.empty:
            return df

        # Add features
        df = self.compute_elo_features(df)
        logger.info("Added Elo features")

        df = self.compute_form_features(df)
        logger.info("Added form features")

        df = self.compute_head_to_head_features(df)
        logger.info("Added H2H features")

        df = self.compute_rest_features(df)
        logger.info("Added rest features")

        df = self.compute_odds_features(df)
        logger.info("Added market features")

        # Add target variables
        if include_target:
            df["target_home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
            df["target_draw"] = (df["home_goals"] == df["away_goals"]).astype(int)
            df["target_away_win"] = (df["home_goals"] < df["away_goals"]).astype(int)
            df["target_total_goals"] = df["home_goals"] + df["away_goals"]

        logger.info(f"Feature matrix complete: {df.shape}")
        return df

    def get_feature_columns(self) -> list:
        """Get list of feature column names for modeling."""
        return [
            # Elo features
            "home_elo", "away_elo", "elo_diff", "elo_home_prob",
            # Form features (5-match window)
            "home_xg_for_l5", "home_xg_against_l5", "home_points_l5",
            "away_xg_for_l5", "away_xg_against_l5", "away_points_l5",
            # Form features (10-match window)
            "home_xg_for_l10", "home_xg_against_l10", "home_points_l10",
            "away_xg_for_l10", "away_xg_against_l10", "away_points_l10",
            # H2H features
            "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_total_goals",
            # Rest features
            "home_rest_days", "away_rest_days", "rest_diff", "congestion_diff",
            # Market features
            "market_home_prob", "market_draw_prob", "market_away_prob",
        ]
