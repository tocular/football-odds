"""
Database module for Nostrus.

Provides SQLite database connection and schema management using SQLAlchemy.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.engine import Engine
import yaml

logger = logging.getLogger(__name__)

Base = declarative_base()


class Team(Base):
    """Teams table - master reference for all teams."""

    __tablename__ = "teams"

    team_id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_name = Column(String, unique=True, nullable=False)
    fbref_name = Column(String)
    elo_name = Column(String)
    transfermarkt_name = Column(String)
    footballdata_name = Column(String)
    country = Column(String, nullable=False)
    league = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    home_matches = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    stats = relationship("TeamStats", back_populates="team")
    elo_ratings = relationship("EloRating", back_populates="team")
    injuries = relationship("Injury", back_populates="team")


class Match(Base):
    """Matches table - central fact table for all matches."""

    __tablename__ = "matches"

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    season = Column(String, nullable=False)
    competition = Column(String, nullable=False)
    matchweek = Column(Integer)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"))
    away_team_id = Column(Integer, ForeignKey("teams.team_id"))
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_xg_open_play = Column(Float)
    away_xg_open_play = Column(Float)
    home_xg_set_piece = Column(Float)
    away_xg_set_piece = Column(Float)
    home_xg_penalty = Column(Float)
    away_xg_penalty = Column(Float)
    status = Column(String, default="scheduled")  # scheduled, completed, postponed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    odds = relationship("MatchOdds", back_populates="match")
    predictions = relationship("Prediction", back_populates="match")

    __table_args__ = (
        Index("idx_matches_date", "date"),
        Index("idx_matches_teams", "home_team_id", "away_team_id"),
        Index("idx_matches_season_comp", "season", "competition"),
    )


class TeamStats(Base):
    """Team statistics table - season-to-date and rolling stats."""

    __tablename__ = "team_stats"

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    as_of_date = Column(Date, nullable=False)
    season = Column(String, nullable=False)
    matches_played = Column(Integer)
    # Offensive stats
    npxg_per90 = Column(Float)
    npxg_per90_home = Column(Float)
    npxg_per90_away = Column(Float)
    npxg_per90_l5 = Column(Float)
    npxg_per90_l10 = Column(Float)
    # Defensive stats
    xga_per90 = Column(Float)
    xga_per90_home = Column(Float)
    xga_per90_away = Column(Float)
    xga_per90_l5 = Column(Float)
    xga_per90_l10 = Column(Float)
    # Performance
    points_per_game = Column(Float)
    points_per_game_l5 = Column(Float)
    goals_minus_xg = Column(Float)
    # Shot stats
    shots_per90 = Column(Float)
    shots_on_target_pct = Column(Float)
    # Possession and passing
    possession_avg = Column(Float)
    progressive_passes_per90 = Column(Float)
    # Defensive actions
    tackles_won_per90 = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    team = relationship("Team", back_populates="stats")

    __table_args__ = (
        UniqueConstraint("team_id", "as_of_date", name="uq_team_stats"),
        Index("idx_team_stats_team_date", "team_id", "as_of_date"),
        Index("idx_team_stats_season", "season"),
    )


class EloRating(Base):
    """Elo ratings table - daily ratings from Club Elo."""

    __tablename__ = "elo_ratings"

    elo_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    date = Column(Date, nullable=False)
    elo_rating = Column(Float, nullable=False)
    elo_change = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    team = relationship("Team", back_populates="elo_ratings")

    __table_args__ = (
        UniqueConstraint("team_id", "date", name="uq_elo_rating"),
        Index("idx_elo_team_date", "team_id", "date"),
    )


class MatchOdds(Base):
    """Match odds table - betting odds from various bookmakers."""

    __tablename__ = "match_odds"

    odds_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    bookmaker = Column(String, nullable=False)
    captured_at = Column(DateTime, nullable=False)
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    over_25_odds = Column(Float)
    under_25_odds = Column(Float)
    asian_handicap_line = Column(Float)
    asian_handicap_home = Column(Float)
    asian_handicap_away = Column(Float)
    is_closing = Column(Boolean, default=False)

    # Relationships
    match = relationship("Match", back_populates="odds")

    __table_args__ = (
        Index("idx_odds_match", "match_id"),
        Index("idx_odds_bookmaker", "bookmaker"),
    )


class Injury(Base):
    """Injuries table - player availability data."""

    __tablename__ = "injuries"

    injury_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    player_name = Column(String, nullable=False)
    position = Column(String)
    injury_type = Column(String)
    injury_date = Column(Date)
    expected_return = Column(Date)
    status = Column(String, nullable=False)  # out, doubtful, unknown
    market_value_eur = Column(Integer)
    minutes_played_season = Column(Integer)
    player_importance_index = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    team = relationship("Team", back_populates="injuries")

    __table_args__ = (
        Index("idx_injuries_team", "team_id"),
        Index("idx_injuries_status", "status"),
    )


class Prediction(Base):
    """Predictions table - model predictions for matches."""

    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    model_version = Column(String, nullable=False)
    predicted_at = Column(DateTime, nullable=False)
    lambda_home = Column(Float)
    lambda_away = Column(Float)
    prob_home = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_away = Column(Float, nullable=False)
    prob_over_25 = Column(Float)
    recommended_bet = Column(String)  # H, D, A, or NULL
    edge = Column(Float)
    kelly_stake = Column(Float)

    # Relationships
    match = relationship("Match", back_populates="predictions")
    bets = relationship("Bet", back_populates="prediction")

    __table_args__ = (
        Index("idx_predictions_match", "match_id"),
        Index("idx_predictions_model", "model_version"),
    )


class Bet(Base):
    """Bets table - tracking actual and paper bets."""

    __tablename__ = "bets"

    bet_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.prediction_id"))
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    bet_type = Column(String, nullable=False)  # paper, live
    outcome = Column(String, nullable=False)  # H, D, A
    odds_at_bet = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)
    placed_at = Column(DateTime, nullable=False)
    result = Column(String)  # won, lost, or NULL if pending
    profit = Column(Float)
    settled_at = Column(DateTime)

    # Relationships
    prediction = relationship("Prediction", back_populates="bets")

    __table_args__ = (
        Index("idx_bets_match", "match_id"),
        Index("idx_bets_type", "bet_type"),
    )


class Database:
    """Database connection and management class."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize database connection.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.db_path = Path(config["database"]["path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Created database tables at {self.db_path}")

    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)
        logger.info("Dropped all database tables")

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    def get_engine(self) -> Engine:
        """Get the database engine."""
        return self.engine

    def get_or_create_team(
        self,
        session,
        canonical_name: str,
        country: str,
        league: str,
        **kwargs
    ) -> Team:
        """
        Get existing team or create new one.

        Args:
            session: Database session
            canonical_name: Standardized team name
            country: Country code
            league: League name
            **kwargs: Additional team attributes

        Returns:
            Team object
        """
        team = session.query(Team).filter_by(canonical_name=canonical_name).first()

        if team is None:
            team = Team(
                canonical_name=canonical_name,
                country=country,
                league=league,
                **kwargs
            )
            session.add(team)
            session.flush()
            logger.debug(f"Created new team: {canonical_name}")

        return team

    def get_team_by_name(self, session, name: str, league: Optional[str] = None) -> Optional[Team]:
        """
        Find a team by any of its known names.

        Args:
            session: Database session
            name: Team name to search for
            league: Optional league filter

        Returns:
            Team object or None
        """
        query = session.query(Team).filter(
            (Team.canonical_name == name) |
            (Team.fbref_name == name) |
            (Team.elo_name == name) |
            (Team.transfermarkt_name == name) |
            (Team.footballdata_name == name)
        )

        if league:
            query = query.filter(Team.league == league)

        return query.first()
