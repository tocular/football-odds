"""
ETL Pipeline for Nostrus.

Loads raw data from CSV files into the SQLite database,
handling team name standardization and data validation.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .db import Database, Team, Match, MatchOdds, EloRating, TeamStats

logger = logging.getLogger(__name__)


class ETLPipeline:
    """ETL pipeline for loading football data into the database."""

    # Country codes for each league
    LEAGUE_COUNTRIES = {
        "premier_league": "ENG",
        "la_liga": "ESP",
        "bundesliga": "GER",
        "serie_a": "ITA",
        "ligue_1": "FRA",
    }

    # Mapping from soccerdata league names to internal league names
    SOCCERDATA_LEAGUE_MAP = {
        "ENG-Premier League": "premier_league",
        "ESP-La Liga": "la_liga",
        "GER-Bundesliga": "bundesliga",
        "ITA-Serie A": "serie_a",
        "FRA-Ligue 1": "ligue_1",
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the ETL pipeline.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = Database(config_path)
        self.raw_data_path = Path("data/raw")

        # Load team name mappings
        mappings_path = Path("config/team_mappings.json")
        if mappings_path.exists():
            with open(mappings_path) as f:
                self.team_mappings = json.load(f)
        else:
            self.team_mappings = {}

        # Build reverse mapping for quick lookup
        self._build_reverse_mappings()

    def _build_reverse_mappings(self):
        """Build reverse mapping from alternate names to canonical names."""
        self.reverse_mappings = {}
        for league, teams in self.team_mappings.items():
            for canonical, aliases in teams.items():
                self.reverse_mappings[canonical.lower()] = (canonical, league)
                for alias in aliases:
                    self.reverse_mappings[alias.lower()] = (canonical, league)

    def standardize_team_name(self, name: str, league: Optional[str] = None) -> str:
        """
        Standardize a team name to its canonical form.

        Args:
            name: Raw team name
            league: Optional league hint

        Returns:
            Canonical team name
        """
        if not name:
            return name

        name_lower = name.lower().strip()

        # Check reverse mappings
        if name_lower in self.reverse_mappings:
            return self.reverse_mappings[name_lower][0]

        # If no mapping found, return cleaned original
        return name.strip()

    def initialize_database(self):
        """Create database tables if they don't exist."""
        self.db.create_tables()
        logger.info("Database initialized")

    def load_odds_data(self, filepath: Optional[Path] = None) -> int:
        """
        Load odds data from football-data.co.uk CSV.

        Args:
            filepath: Path to odds CSV (default: data/raw/odds/all_odds.csv)

        Returns:
            Number of matches loaded
        """
        if filepath is None:
            filepath = self.raw_data_path / "odds" / "all_odds.csv"

        if not filepath.exists():
            logger.warning(f"Odds file not found: {filepath}")
            return 0

        logger.info(f"Loading odds data from {filepath}")
        df = pd.read_csv(filepath)

        session = self.db.get_session()
        matches_loaded = 0

        try:
            for _, row in df.iterrows():
                # Skip rows without essential data
                if pd.isna(row.get("home_team")) or pd.isna(row.get("away_team")):
                    continue

                league = row.get("league", "unknown")
                country = self.LEAGUE_COUNTRIES.get(league, "UNK")

                # Get or create teams
                home_name = self.standardize_team_name(row["home_team"], league)
                away_name = self.standardize_team_name(row["away_team"], league)

                home_team = self.db.get_or_create_team(
                    session,
                    canonical_name=home_name,
                    country=country,
                    league=league,
                    footballdata_name=row["home_team"]
                )
                away_team = self.db.get_or_create_team(
                    session,
                    canonical_name=away_name,
                    country=country,
                    league=league,
                    footballdata_name=row["away_team"]
                )

                # Parse date
                match_date = pd.to_datetime(row.get("date"))
                if pd.isna(match_date):
                    continue

                # Check if match already exists
                existing = session.query(Match).filter(
                    Match.date == match_date.date(),
                    Match.home_team_id == home_team.team_id,
                    Match.away_team_id == away_team.team_id
                ).first()

                if existing:
                    match = existing
                else:
                    # Create match
                    match = Match(
                        date=match_date.date(),
                        season=row.get("season", "unknown"),
                        competition=league,
                        home_team_id=home_team.team_id,
                        away_team_id=away_team.team_id,
                        home_goals=int(row["home_goals"]) if pd.notna(row.get("home_goals")) else None,
                        away_goals=int(row["away_goals"]) if pd.notna(row.get("away_goals")) else None,
                        status="completed" if pd.notna(row.get("home_goals")) else "scheduled"
                    )
                    session.add(match)
                    session.flush()
                    matches_loaded += 1

                # Add odds for multiple bookmakers
                odds_data = [
                    ("pinnacle", row.get("pinnacle_home"), row.get("pinnacle_draw"), row.get("pinnacle_away")),
                    ("bet365", row.get("b365_home"), row.get("b365_draw"), row.get("b365_away")),
                    ("max", row.get("max_home"), row.get("max_draw"), row.get("max_away")),
                    ("avg", row.get("avg_home"), row.get("avg_draw"), row.get("avg_away")),
                ]

                for bookmaker, home_odds, draw_odds, away_odds in odds_data:
                    if pd.notna(home_odds) and pd.notna(draw_odds) and pd.notna(away_odds):
                        # Check for existing odds
                        existing_odds = session.query(MatchOdds).filter(
                            MatchOdds.match_id == match.match_id,
                            MatchOdds.bookmaker == bookmaker
                        ).first()

                        if not existing_odds:
                            odds = MatchOdds(
                                match_id=match.match_id,
                                bookmaker=bookmaker,
                                captured_at=datetime.utcnow(),
                                home_odds=float(home_odds),
                                draw_odds=float(draw_odds),
                                away_odds=float(away_odds),
                                over_25_odds=float(row["avg_over_25"]) if pd.notna(row.get("avg_over_25")) else None,
                                under_25_odds=float(row["avg_under_25"]) if pd.notna(row.get("avg_under_25")) else None,
                                is_closing=True
                            )
                            session.add(odds)

            session.commit()
            logger.info(f"Loaded {matches_loaded} matches from odds data")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading odds data: {e}")
            raise
        finally:
            session.close()

        return matches_loaded

    def load_fbref_matches(self, filepath: Optional[Path] = None) -> int:
        """
        Load FBref match data with xG statistics.

        Args:
            filepath: Path to FBref matches CSV

        Returns:
            Number of matches updated
        """
        if filepath is None:
            filepath = self.raw_data_path / "fbref" / "matches.csv"

        if not filepath.exists():
            logger.warning(f"FBref matches file not found: {filepath}")
            return 0

        logger.info(f"Loading FBref match data from {filepath}")
        df = pd.read_csv(filepath)

        session = self.db.get_session()
        matches_updated = 0

        try:
            for _, row in df.iterrows():
                # Get league from source
                source_league = row.get("source_league", "")
                league = self.SOCCERDATA_LEAGUE_MAP.get(source_league, "unknown")

                # Standardize team names
                home_name = self.standardize_team_name(str(row.get("home_team", "")), league)
                away_name = self.standardize_team_name(str(row.get("away_team", "")), league)

                if not home_name or not away_name:
                    continue

                # Find matching match by date and teams
                match_date = pd.to_datetime(row.get("date"))
                if pd.isna(match_date):
                    continue

                # Look up teams
                home_team = self.db.get_team_by_name(session, home_name, league)
                away_team = self.db.get_team_by_name(session, away_name, league)

                if not home_team or not away_team:
                    continue

                # Find existing match
                match = session.query(Match).filter(
                    Match.date == match_date.date(),
                    Match.home_team_id == home_team.team_id,
                    Match.away_team_id == away_team.team_id
                ).first()

                if match:
                    # Update xG data
                    if pd.notna(row.get("home_xg")):
                        match.home_xg = float(row["home_xg"])
                    if pd.notna(row.get("away_xg")):
                        match.away_xg = float(row["away_xg"])

                    match.updated_at = datetime.utcnow()
                    matches_updated += 1

            session.commit()
            logger.info(f"Updated xG data for {matches_updated} matches")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading FBref data: {e}")
            raise
        finally:
            session.close()

        return matches_updated

    def load_elo_ratings(self, filepath: Optional[Path] = None) -> int:
        """
        Load Elo ratings from Club Elo historical data.

        Args:
            filepath: Path to Elo ratings CSV (defaults to historical_ratings.csv)

        Returns:
            Number of ratings loaded
        """
        if filepath is None:
            filepath = self.raw_data_path / "elo" / "historical_ratings.csv"

        if not filepath.exists():
            logger.warning(f"Elo ratings file not found: {filepath}")
            return 0

        logger.info(f"Loading Elo ratings from {filepath}")
        df = pd.read_csv(filepath)

        session = self.db.get_session()
        ratings_loaded = 0
        teams_not_found = set()

        try:
            # Check if this is historical format (has db_team_name column)
            is_historical = "db_team_name" in df.columns

            for _, row in df.iterrows():
                # Use db_team_name if available (historical format), otherwise fall back to team/club
                if is_historical:
                    team_name = str(row.get("db_team_name", ""))
                else:
                    team_name = str(row.get("team", row.get("club", "")))

                if not team_name:
                    continue

                # Try to find team in database
                team = self.db.get_team_by_name(session, team_name)
                if not team:
                    canonical = self.standardize_team_name(team_name)
                    team = self.db.get_team_by_name(session, canonical)

                if not team:
                    teams_not_found.add(team_name)
                    continue

                # Update team's elo_name if not set (use ClubElo name from 'team' column)
                if not team.elo_name and "team" in row:
                    team.elo_name = str(row["team"])

                # Parse date - historical format uses 'from' column
                rating_date = pd.to_datetime(row.get("from", row.get("date", "")))
                if pd.isna(rating_date):
                    continue

                # Check for existing rating
                existing = session.query(EloRating).filter(
                    EloRating.team_id == team.team_id,
                    EloRating.date == rating_date.date()
                ).first()

                if not existing:
                    rating = EloRating(
                        team_id=team.team_id,
                        date=rating_date.date(),
                        elo_rating=float(row.get("elo", 0))
                    )
                    session.add(rating)
                    ratings_loaded += 1

            session.commit()
            logger.info(f"Loaded {ratings_loaded} Elo ratings")
            if teams_not_found:
                logger.warning(f"Teams not found in database: {teams_not_found}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading Elo ratings: {e}")
            raise
        finally:
            session.close()

        return ratings_loaded

    def load_fotmob_xg(self, filepath: Optional[Path] = None) -> int:
        """
        Load xG data from FotMob.

        Args:
            filepath: Path to FotMob xG CSV

        Returns:
            Number of matches updated with xG
        """
        if filepath is None:
            filepath = self.raw_data_path / "fotmob" / "match_xg.csv"

        if not filepath.exists():
            logger.warning(f"FotMob xG file not found: {filepath}")
            return 0

        logger.info(f"Loading FotMob xG data from {filepath}")
        df = pd.read_csv(filepath)

        session = self.db.get_session()
        matches_updated = 0

        try:
            for _, row in df.iterrows():
                # Get league from source
                source_league = row.get("source_league", "")
                league = self.SOCCERDATA_LEAGUE_MAP.get(source_league, "unknown")

                # Try to find team names in the data
                # FotMob format may vary - check for common column names
                home_name = None
                away_name = None

                for col in ["home_team", "home", "team_home"]:
                    if col in row and pd.notna(row.get(col)):
                        home_name = self.standardize_team_name(str(row[col]), league)
                        break

                for col in ["away_team", "away", "team_away"]:
                    if col in row and pd.notna(row.get(col)):
                        away_name = self.standardize_team_name(str(row[col]), league)
                        break

                if not home_name or not away_name:
                    continue

                # Parse date
                match_date = None
                for col in ["date", "match_date", "game_date"]:
                    if col in row and pd.notna(row.get(col)):
                        match_date = pd.to_datetime(row[col])
                        break

                if match_date is None or pd.isna(match_date):
                    continue

                # Look up teams
                home_team = self.db.get_team_by_name(session, home_name, league)
                away_team = self.db.get_team_by_name(session, away_name, league)

                if not home_team or not away_team:
                    continue

                # Find existing match
                match = session.query(Match).filter(
                    Match.date == match_date.date(),
                    Match.home_team_id == home_team.team_id,
                    Match.away_team_id == away_team.team_id
                ).first()

                if match:
                    # Update xG data - check various possible column names
                    for col in ["home_xg", "xg_home", "home_expected_goals", "xG_home"]:
                        if col in row and pd.notna(row.get(col)):
                            match.home_xg = float(row[col])
                            break

                    for col in ["away_xg", "xg_away", "away_expected_goals", "xG_away"]:
                        if col in row and pd.notna(row.get(col)):
                            match.away_xg = float(row[col])
                            break

                    match.updated_at = datetime.utcnow()
                    matches_updated += 1

            session.commit()
            logger.info(f"Updated xG data for {matches_updated} matches from FotMob")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading FotMob xG data: {e}")
            raise
        finally:
            session.close()

        return matches_updated

    def load_fbref_scraped_xg(self, filepath: Optional[Path] = None) -> int:
        """
        Load xG data from scraped FBref schedule pages.

        Args:
            filepath: Path to FBref xG CSV (default: data/raw/fbref/xg_all.csv)

        Returns:
            Number of matches updated with xG
        """
        if filepath is None:
            filepath = self.raw_data_path / "fbref" / "xg_all.csv"

        if not filepath.exists():
            logger.warning(f"FBref scraped xG file not found: {filepath}")
            return 0

        logger.info(f"Loading FBref scraped xG data from {filepath}")
        df = pd.read_csv(filepath)

        session = self.db.get_session()
        matches_updated = 0
        teams_not_found = set()

        # League name mapping (FBref uses internal league names)
        fbref_league_map = {
            "premier_league": "premier_league",
            "la_liga": "la_liga",
            "bundesliga": "bundesliga",
            "serie_a": "serie_a",
            "ligue_1": "ligue_1",
        }

        try:
            for _, row in df.iterrows():
                league = fbref_league_map.get(row.get("league"), row.get("league"))

                # Get team names
                home_name = self.standardize_team_name(str(row.get("home", "")), league)
                away_name = self.standardize_team_name(str(row.get("away", "")), league)

                if not home_name or not away_name:
                    continue

                # Parse date
                match_date = pd.to_datetime(row.get("date"))
                if pd.isna(match_date):
                    continue

                # Look up teams
                home_team = self.db.get_team_by_name(session, home_name, league)
                away_team = self.db.get_team_by_name(session, away_name, league)

                if not home_team:
                    teams_not_found.add(f"{home_name} ({league})")
                    continue
                if not away_team:
                    teams_not_found.add(f"{away_name} ({league})")
                    continue

                # Find existing match
                match = session.query(Match).filter(
                    Match.date == match_date.date(),
                    Match.home_team_id == home_team.team_id,
                    Match.away_team_id == away_team.team_id
                ).first()

                if match:
                    # Update xG data if available
                    if pd.notna(row.get("home_xg")):
                        match.home_xg = float(row["home_xg"])
                    if pd.notna(row.get("away_xg")):
                        match.away_xg = float(row["away_xg"])

                    match.updated_at = datetime.utcnow()
                    matches_updated += 1

            session.commit()
            logger.info(f"Updated xG data for {matches_updated} matches from FBref")
            if teams_not_found:
                logger.warning(f"Teams not found ({len(teams_not_found)}): {list(teams_not_found)[:10]}...")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading FBref scraped xG data: {e}")
            raise
        finally:
            session.close()

        return matches_updated

    def load_understat_xg(self, filepath: Optional[Path] = None) -> int:
        """
        Load xG data from Understat.

        Args:
            filepath: Path to Understat xG CSV (default: data/raw/understat/xg_all.csv)

        Returns:
            Number of matches updated with xG
        """
        if filepath is None:
            filepath = self.raw_data_path / "understat" / "xg_all.csv"

        if not filepath.exists():
            logger.warning(f"Understat xG file not found: {filepath}")
            return 0

        logger.info(f"Loading Understat xG data from {filepath}")
        df = pd.read_csv(filepath)

        session = self.db.get_session()
        matches_updated = 0
        teams_not_found = set()

        try:
            for _, row in df.iterrows():
                league = row.get("league", "unknown")

                # Get team names
                home_name = self.standardize_team_name(str(row.get("home", "")), league)
                away_name = self.standardize_team_name(str(row.get("away", "")), league)

                if not home_name or not away_name:
                    continue

                # Parse date
                match_date = pd.to_datetime(row.get("date"))
                if pd.isna(match_date):
                    continue

                # Look up teams
                home_team = self.db.get_team_by_name(session, home_name, league)
                away_team = self.db.get_team_by_name(session, away_name, league)

                if not home_team:
                    teams_not_found.add(f"{home_name} ({league})")
                    continue
                if not away_team:
                    teams_not_found.add(f"{away_name} ({league})")
                    continue

                # Find existing match - try exact date first, then day before for timezone discrepancies
                # (Understat dates are sometimes 1 day ahead due to timezone differences)
                match = session.query(Match).filter(
                    Match.date == match_date.date(),
                    Match.home_team_id == home_team.team_id,
                    Match.away_team_id == away_team.team_id
                ).first()

                if not match:
                    # Try day before (Understat often shows next-day dates for late matches)
                    match = session.query(Match).filter(
                        Match.date == (match_date - timedelta(days=1)).date(),
                        Match.home_team_id == home_team.team_id,
                        Match.away_team_id == away_team.team_id
                    ).first()

                if match:
                    # Update xG data if available
                    if pd.notna(row.get("home_xg")):
                        match.home_xg = float(row["home_xg"])
                    if pd.notna(row.get("away_xg")):
                        match.away_xg = float(row["away_xg"])

                    match.updated_at = datetime.utcnow()
                    matches_updated += 1

            session.commit()
            logger.info(f"Updated xG data for {matches_updated} matches from Understat")
            if teams_not_found:
                logger.warning(f"Teams not found ({len(teams_not_found)}): {list(teams_not_found)[:10]}...")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading Understat xG data: {e}")
            raise
        finally:
            session.close()

        return matches_updated

    def run_full_etl(self) -> dict:
        """
        Run the complete ETL pipeline.

        Returns:
            Dictionary with counts of loaded records
        """
        self.initialize_database()

        results = {
            "odds_matches": self.load_odds_data(),
            "elo_ratings": self.load_elo_ratings(),
            "understat_xg_updates": self.load_understat_xg(),  # Primary xG source
        }

        logger.info(f"ETL complete: {results}")
        return results

    def get_match_dataframe(
        self,
        season: Optional[str] = None,
        league: Optional[str] = None,
        include_odds: bool = True
    ) -> pd.DataFrame:
        """
        Export matches to a DataFrame for modeling.

        Args:
            season: Filter by season
            league: Filter by league
            include_odds: Include odds data

        Returns:
            DataFrame with match data
        """
        session = self.db.get_session()

        try:
            query = session.query(Match).join(
                Team, Match.home_team_id == Team.team_id
            )

            if season:
                query = query.filter(Match.season == season)
            if league:
                query = query.filter(Match.competition == league)

            matches = query.all()

            data = []
            for match in matches:
                row = {
                    "match_id": match.match_id,
                    "date": match.date,
                    "season": match.season,
                    "competition": match.competition,
                    "home_team": match.home_team.canonical_name if match.home_team else None,
                    "away_team": match.away_team.canonical_name if match.away_team else None,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                    "home_xg": match.home_xg,
                    "away_xg": match.away_xg,
                }

                if include_odds:
                    # Get Pinnacle odds (preferred) or Bet365 as fallback
                    for bookmaker in ["pinnacle", "bet365", "avg"]:
                        odds = next(
                            (o for o in match.odds if o.bookmaker == bookmaker),
                            None
                        )
                        if odds:
                            row["home_odds"] = odds.home_odds
                            row["draw_odds"] = odds.draw_odds
                            row["away_odds"] = odds.away_odds
                            row["odds_source"] = bookmaker
                            break

                data.append(row)

            return pd.DataFrame(data)

        finally:
            session.close()
