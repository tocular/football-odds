"""
Odds scraper for football-data.co.uk.

Downloads historical and current betting odds data directly from
football-data.co.uk CSV files.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
import yaml
from io import StringIO

logger = logging.getLogger(__name__)

# Column mapping for football-data.co.uk CSVs
ODDS_COLUMNS = {
    # Match info
    "Div": "division",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    # Results
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H, D, A
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    # Match stats
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    # Bet365 odds
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    # Pinnacle odds
    "PSH": "pinnacle_home",
    "PSD": "pinnacle_draw",
    "PSA": "pinnacle_away",
    # Max odds
    "MaxH": "max_home",
    "MaxD": "max_draw",
    "MaxA": "max_away",
    # Average odds
    "AvgH": "avg_home",
    "AvgD": "avg_draw",
    "AvgA": "avg_away",
    # Over/under 2.5
    "BbOU": "ou_line",
    "BbMx>2.5": "max_over_25",
    "BbAv>2.5": "avg_over_25",
    "BbMx<2.5": "max_under_25",
    "BbAv<2.5": "avg_under_25",
    # Asian handicap
    "BbAH": "ah_line",
    "BbAHh": "ah_size",
    "BbMxAHH": "max_ah_home",
    "BbAvAHH": "avg_ah_home",
    "BbMxAHA": "max_ah_away",
    "BbAvAHA": "avg_ah_away",
}


class OddsScraper:
    """Scraper for football-data.co.uk betting odds."""

    BASE_URL = "https://www.football-data.co.uk"

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the odds scraper.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.leagues = self.config["leagues"]
        self.league_codes = self.config["odds_league_codes"]
        self.season_codes = self.config["season_codes"]
        self.raw_data_path = Path("data/raw/odds")
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.timeout = self.config["scraping"]["timeout"]
        self.retry_attempts = self.config["scraping"]["retry_attempts"]

    def _build_url(self, league: str, season: str) -> str:
        """Build the download URL for a specific league and season."""
        code = self.league_codes[league]
        season_code = self.season_codes[season]
        return f"{self.BASE_URL}/mmz4281/{season_code}/{code}.csv"

    def _fetch_csv(self, url: str) -> Optional[pd.DataFrame]:
        """
        Fetch and parse a CSV from the given URL.

        Args:
            url: URL to fetch

        Returns:
            DataFrame or None if fetch failed
        """
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Handle encoding issues
                content = response.content.decode("utf-8", errors="replace")
                df = pd.read_csv(StringIO(content))

                return df

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                continue

        logger.error(f"Failed to fetch {url} after {self.retry_attempts} attempts")
        return None

    def _clean_dataframe(self, df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
        """
        Clean and standardize the odds DataFrame.

        Args:
            df: Raw DataFrame
            league: League identifier
            season: Season identifier

        Returns:
            Cleaned DataFrame
        """
        # Drop rows with no data
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])

        # Rename columns
        rename_map = {k: v for k, v in ODDS_COLUMNS.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Parse date - try multiple formats
        if "date" in df.columns:
            # Preserve original date strings for fallback parsing
            original_dates = df["date"].copy()
            # Football-data.co.uk uses DD/MM/YYYY or DD/MM/YY format
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            # Fallback for 2-digit year format (used in older seasons)
            mask = df["date"].isna()
            if mask.any():
                df.loc[mask, "date"] = pd.to_datetime(
                    original_dates.loc[mask], format="%d/%m/%y", errors="coerce"
                )

        # Add metadata
        df["league"] = league
        df["season"] = season

        # Select only mapped columns plus metadata
        available_cols = [v for v in ODDS_COLUMNS.values() if v in df.columns]
        available_cols.extend(["league", "season"])
        df = df[available_cols]

        return df

    def fetch_league_season(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """
        Fetch odds data for a specific league and season.

        Args:
            league: League identifier (e.g., 'premier_league')
            season: Season identifier (e.g., '2023-2024')

        Returns:
            DataFrame with odds data or None
        """
        if league not in self.league_codes:
            logger.error(f"Unknown league: {league}")
            return None

        if season not in self.season_codes:
            logger.error(f"Unknown season: {season}")
            return None

        url = self._build_url(league, season)
        logger.info(f"Fetching odds: {league} {season} from {url}")

        df = self._fetch_csv(url)
        if df is None:
            return None

        return self._clean_dataframe(df, league, season)

    def fetch_all_leagues(self, season: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch odds for all configured leagues.

        Args:
            season: Specific season to fetch (default: all configured seasons)

        Returns:
            Combined DataFrame with all odds data
        """
        seasons_to_fetch = [season] if season else list(self.season_codes.keys())
        all_data = []

        for lg in self.leagues:
            for ssn in seasons_to_fetch:
                df = self.fetch_league_season(lg, ssn)
                if df is not None and not df.empty:
                    all_data.append(df)

        if not all_data:
            logger.warning("No odds data retrieved")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        return combined

    def save_raw_data(self, df: pd.DataFrame, filename: str = "all_odds.csv") -> Path:
        """
        Save odds data to CSV.

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.raw_data_path / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path

    def fetch_and_save_all(self) -> pd.DataFrame:
        """
        Fetch all odds data and save to disk.

        Returns:
            Combined DataFrame
        """
        df = self.fetch_all_leagues()
        if not df.empty:
            self.save_raw_data(df)
        return df
