"""
Dixon-Coles Poisson Model for football match prediction.

Implements the Dixon-Coles (1997) bivariate Poisson model with
adjustments for low-scoring outcomes.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model for football prediction.

    The model estimates:
    - Attack strength for each team
    - Defense strength for each team
    - Home advantage parameter
    - Rho parameter for low-score correlation adjustment
    """

    def __init__(
        self,
        time_decay: float = 0.0018,  # Half-life ~385 days
        max_goals: int = 10
    ):
        """
        Initialize the Dixon-Coles model.

        Args:
            time_decay: Exponential decay rate for match weights
            max_goals: Maximum goals to consider in probability calculations
        """
        self.time_decay = time_decay
        self.max_goals = max_goals

        # Model parameters (set after fitting)
        self.attack_params = {}
        self.defense_params = {}
        self.home_advantage = None
        self.rho = None
        self.teams = []

    def _tau(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float
    ) -> float:
        """
        Dixon-Coles adjustment factor for low-scoring outcomes.

        Args:
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            rho: Correlation parameter

        Returns:
            Adjustment factor tau
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - lambda_home * lambda_away * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + lambda_home * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + lambda_away * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0

    def _compute_weights(self, dates: pd.Series, reference_date: pd.Timestamp) -> np.ndarray:
        """
        Compute time-decay weights for matches.

        Args:
            dates: Series of match dates
            reference_date: Reference date for weight calculation

        Returns:
            Array of weights
        """
        days_diff = (reference_date - pd.to_datetime(dates)).dt.days
        return np.exp(-self.time_decay * days_diff)

    def _tau_vectorized(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        lambda_home: np.ndarray,
        lambda_away: np.ndarray,
        rho: float
    ) -> np.ndarray:
        """
        Vectorized Dixon-Coles adjustment factor for low-scoring outcomes.

        Args:
            home_goals: Array of home goals
            away_goals: Array of away goals
            lambda_home: Array of expected home goals
            lambda_away: Array of expected away goals
            rho: Correlation parameter

        Returns:
            Array of tau adjustment factors
        """
        tau = np.ones(len(home_goals))

        # 0-0: tau = 1 - lambda_home * lambda_away * rho
        mask_00 = (home_goals == 0) & (away_goals == 0)
        tau[mask_00] = 1 - lambda_home[mask_00] * lambda_away[mask_00] * rho

        # 0-1: tau = 1 + lambda_home * rho
        mask_01 = (home_goals == 0) & (away_goals == 1)
        tau[mask_01] = 1 + lambda_home[mask_01] * rho

        # 1-0: tau = 1 + lambda_away * rho
        mask_10 = (home_goals == 1) & (away_goals == 0)
        tau[mask_10] = 1 + lambda_away[mask_10] * rho

        # 1-1: tau = 1 - rho
        mask_11 = (home_goals == 1) & (away_goals == 1)
        tau[mask_11] = 1 - rho

        return tau

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        team_to_idx: dict
    ) -> float:
        """
        Vectorized negative log-likelihood for optimization.

        Args:
            params: Model parameters (attack, defense, home_adv, rho)
            home_teams: Array of home team indices
            away_teams: Array of away team indices
            home_goals: Array of home goals
            away_goals: Array of away goals
            weights: Array of match weights
            team_to_idx: Mapping from team to index

        Returns:
            Negative log-likelihood
        """
        n_teams = len(team_to_idx)

        # Extract parameters
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = np.clip(params[2*n_teams + 1], -0.99, 0.99)

        # Vectorized expected goals calculation
        # Clip the exponent to prevent overflow (exp(700) is already near float max)
        exp_arg_home = attack[home_teams] + defense[away_teams] + home_adv
        exp_arg_away = attack[away_teams] + defense[home_teams]
        exp_arg_home = np.clip(exp_arg_home, -20, 20)
        exp_arg_away = np.clip(exp_arg_away, -20, 20)

        lambda_home = np.exp(exp_arg_home)
        lambda_away = np.exp(exp_arg_away)

        # Ensure positive lambdas
        lambda_home = np.maximum(lambda_home, 0.001)
        lambda_away = np.maximum(lambda_away, 0.001)

        # Vectorized tau calculation
        tau = self._tau_vectorized(home_goals, away_goals, lambda_home, lambda_away, rho)
        tau = np.maximum(tau, 0.001)

        # Vectorized log-likelihood: log(tau) + logpmf(home_goals) + logpmf(away_goals)
        # poisson.logpmf is already vectorized
        log_lik = weights * (
            np.log(tau) +
            poisson.logpmf(home_goals, lambda_home) +
            poisson.logpmf(away_goals, lambda_away)
        )

        return -np.sum(log_lik)

    def fit(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        date_col: str = "date"
    ) -> "DixonColesModel":
        """
        Fit the Dixon-Coles model to match data.

        Args:
            df: DataFrame with match data
            home_team_col: Column name for home team
            away_team_col: Column name for away team
            home_goals_col: Column name for home goals
            away_goals_col: Column name for away goals
            date_col: Column name for match date

        Returns:
            Fitted model (self)
        """
        # Get unique teams
        self.teams = sorted(set(df[home_team_col].unique()) | set(df[away_team_col].unique()))
        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        logger.info(f"Fitting Dixon-Coles model with {n_teams} teams and {len(df)} matches")

        # Convert to arrays
        home_teams = df[home_team_col].map(team_to_idx).values
        away_teams = df[away_team_col].map(team_to_idx).values
        home_goals = df[home_goals_col].values
        away_goals = df[away_goals_col].values

        # Compute weights
        reference_date = pd.to_datetime(df[date_col]).max()
        weights = self._compute_weights(df[date_col], reference_date)

        # Initial parameters
        # attack and defense start at 0 (average team)
        # home advantage around 0.25 (typical ~1.3-1.4 goals home vs ~1.1 away)
        # rho around -0.1 (slight negative correlation for low scores)
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = 0.25  # home advantage
        x0[2 * n_teams + 1] = -0.1  # rho

        # Constraint: sum of attack params = 0 (identifiability)
        def constraint_attack(params):
            return np.sum(params[:n_teams])

        def constraint_defense(params):
            return np.sum(params[n_teams:2*n_teams])

        constraints = [
            {"type": "eq", "fun": constraint_attack},
            {"type": "eq", "fun": constraint_defense}
        ]

        # Bounds for rho
        bounds = [(None, None)] * (2 * n_teams) + [(None, None), (-0.99, 0.99)]

        # Optimize
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_teams, away_teams, home_goals, away_goals, weights, team_to_idx),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-6}
        )

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        # Extract parameters
        self.attack_params = {team: result.x[i] for i, team in enumerate(self.teams)}
        self.defense_params = {team: result.x[n_teams + i] for i, team in enumerate(self.teams)}
        self.home_advantage = result.x[2 * n_teams]
        self.rho = result.x[2 * n_teams + 1]

        logger.info(f"Model fitted: home_adv={self.home_advantage:.3f}, rho={self.rho:.3f}")

        return self

    def predict_lambda(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float]:
        """
        Predict expected goals for a match.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Tuple of (lambda_home, lambda_away)
        """
        attack_home = self.attack_params.get(home_team, 0)
        defense_home = self.defense_params.get(home_team, 0)
        attack_away = self.attack_params.get(away_team, 0)
        defense_away = self.defense_params.get(away_team, 0)

        lambda_home = np.exp(attack_home + defense_away + self.home_advantage)
        lambda_away = np.exp(attack_away + defense_home)

        return lambda_home, lambda_away

    def predict_proba(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float, float]:
        """
        Predict 1X2 probabilities for a match.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Tuple of (prob_home, prob_draw, prob_away)
        """
        lambda_home, lambda_away = self.predict_lambda(home_team, away_team)

        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = (
                    poisson.pmf(i, lambda_home) *
                    poisson.pmf(j, lambda_away) *
                    self._tau(i, j, lambda_home, lambda_away, self.rho)
                )

                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p

        # Normalize
        total = prob_home + prob_draw + prob_away
        return prob_home / total, prob_draw / total, prob_away / total

    def predict_score_proba(
        self,
        home_team: str,
        away_team: str
    ) -> pd.DataFrame:
        """
        Predict probability matrix for all score outcomes.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            DataFrame with score probabilities
        """
        lambda_home, lambda_away = self.predict_lambda(home_team, away_team)

        probs = np.zeros((self.max_goals + 1, self.max_goals + 1))

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                probs[i, j] = (
                    poisson.pmf(i, lambda_home) *
                    poisson.pmf(j, lambda_away) *
                    self._tau(i, j, lambda_home, lambda_away, self.rho)
                )

        # Normalize
        probs = probs / probs.sum()

        return pd.DataFrame(
            probs,
            index=[f"H{i}" for i in range(self.max_goals + 1)],
            columns=[f"A{j}" for j in range(self.max_goals + 1)]
        )

    def predict_over_under(
        self,
        home_team: str,
        away_team: str,
        line: float = 2.5
    ) -> Tuple[float, float]:
        """
        Predict over/under probabilities.

        Args:
            home_team: Home team name
            away_team: Away team name
            line: Goals line (default 2.5)

        Returns:
            Tuple of (prob_over, prob_under)
        """
        score_probs = self.predict_score_proba(home_team, away_team)

        prob_over = 0.0
        prob_under = 0.0

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                total_goals = i + j
                if total_goals > line:
                    prob_over += score_probs.iloc[i, j]
                elif total_goals < line:
                    prob_under += score_probs.iloc[i, j]

        return prob_over, prob_under

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for multiple matches.

        Args:
            df: DataFrame with home_team and away_team columns

        Returns:
            DataFrame with predictions added
        """
        results = []

        for _, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            lambda_home, lambda_away = self.predict_lambda(home_team, away_team)
            prob_home, prob_draw, prob_away = self.predict_proba(home_team, away_team)
            prob_over, prob_under = self.predict_over_under(home_team, away_team)

            results.append({
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "prob_over_25": prob_over,
                "prob_under_25": prob_under
            })

        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings DataFrame.

        Returns:
            DataFrame with attack and defense ratings per team
        """
        data = []
        for team in self.teams:
            data.append({
                "team": team,
                "attack": self.attack_params[team],
                "defense": self.defense_params[team],
                "overall": self.attack_params[team] - self.defense_params[team]
            })

        return pd.DataFrame(data).sort_values("overall", ascending=False)
