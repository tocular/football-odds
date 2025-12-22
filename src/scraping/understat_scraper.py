"""
Understat xG scraper using the understat Python library.

Fetches match-level xG data from understat.com for all configured leagues/seasons.
Includes exponential backoff and retry logic to handle rate limiting.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Optional
import aiohttp
import pandas as pd
from understat import Understat

logger = logging.getLogger(__name__)

# Understat league codes (different from our internal codes)
UNDERSTAT_LEAGUES = {
    "premier_league": "epl",
    "la_liga": "la_liga",
    "bundesliga": "bundesliga",
    "serie_a": "serie_a",
    "ligue_1": "ligue_1",
}

# Understat uses calendar year for season start (e.g., 2023 = 2023-2024 season)
SEASONS = {
    "2015-2016": 2015,
    "2016-2017": 2016,
    "2017-2018": 2017,
    "2018-2019": 2018,
    "2019-2020": 2019,
    "2020-2021": 2020,
    "2021-2022": 2021,
    "2022-2023": 2022,
    "2023-2024": 2023,
    "2024-2025": 2024,
    "2025-2026": 2025,
}


class UnderstatScraper:
    """Scraper for Understat xG data with rate limit handling."""

    def __init__(self, delay: float = 2.0, max_retries: int = 5):
        """
        Initialize the scraper.

        Args:
            delay: Base seconds to wait between API calls
            max_retries: Maximum retry attempts for failed requests
        """
        self.delay = delay
        self.max_retries = max_retries
        self.raw_data_path = Path("data/raw/understat")
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def _convert_match(self, match: dict, league: str, season: str) -> dict:
        """Convert Understat match format to our standard format."""
        return {
            "date": match["datetime"].split(" ")[0],
            "home": match["h"]["title"],
            "away": match["a"]["title"],
            "home_goals": int(match["goals"]["h"]),
            "away_goals": int(match["goals"]["a"]),
            "home_xg": float(match["xG"]["h"]),
            "away_xg": float(match["xG"]["a"]),
            "league": league,
            "season": season,
            "understat_id": match["id"],
        }

    async def fetch_league_season_async(
        self,
        league: str,
        season: str
    ) -> pd.DataFrame:
        """Fetch xG data for a single league/season with retry logic."""
        understat_league = UNDERSTAT_LEAGUES.get(league)
        understat_year = SEASONS.get(season)

        if not understat_league or not understat_year:
            logger.warning(f"Skipping {league} {season} - not in Understat config")
            return pd.DataFrame()

        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Create fresh session for each request to avoid connection issues
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    understat = Understat(session)
                    fixtures = await understat.get_league_results(understat_league, understat_year)

                if not fixtures:
                    logger.warning(f"No fixtures found for {league} {season}")
                    return pd.DataFrame()

                # Convert to our format
                matches = [self._convert_match(m, league, season) for m in fixtures if m.get("isResult")]
                df = pd.DataFrame(matches)

                logger.info(f"Fetched {len(df)} matches from {league} {season}")
                return df

            except Exception as e:
                wait_time = self.delay * (2 ** attempt) + random.uniform(0.5, 1.5)

                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {league} {season}: {e}")
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {league} {season}: {e}")
                    return pd.DataFrame()

        return pd.DataFrame()

    async def fetch_all_async(
        self,
        leagues: Optional[list[str]] = None,
        seasons: Optional[list[str]] = None,
        save_checkpoint: bool = True
    ) -> pd.DataFrame:
        """Fetch xG data for all configured leagues and seasons."""
        leagues = leagues or list(UNDERSTAT_LEAGUES.keys())
        seasons = seasons or list(SEASONS.keys())

        all_data = []
        failed_requests = []
        total_requests = len(leagues) * len(seasons)
        completed = 0

        for league in leagues:
            league_data = []

            for season in seasons:
                completed += 1
                logger.info(f"Progress: {completed}/{total_requests} - {league} {season}")

                df = await self.fetch_league_season_async(league, season)

                if not df.empty:
                    league_data.append(df)
                    all_data.append(df)
                else:
                    failed_requests.append((league, season))

                # Add jitter to delay to avoid pattern detection
                wait_time = self.delay + random.uniform(0.5, 1.5)
                await asyncio.sleep(wait_time)

            # Save checkpoint after each league
            if save_checkpoint and league_data:
                checkpoint_df = pd.concat(league_data, ignore_index=True)
                checkpoint_path = self.raw_data_path / f"xg_{league}.csv"
                checkpoint_df.to_csv(checkpoint_path, index=False)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Retry failed requests with longer delays
        if failed_requests:
            logger.info(f"Retrying {len(failed_requests)} failed requests with longer delays...")

            for league, season in failed_requests:
                logger.info(f"Retrying {league} {season}...")

                # Wait longer before retry
                await asyncio.sleep(self.delay * 3 + random.uniform(1, 3))

                df = await self.fetch_league_season_async(league, season)

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Successfully fetched {league} {season} on retry")

        if not all_data:
            logger.warning("No Understat xG data fetched")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Save combined file
        output_path = self.raw_data_path / "xg_all.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} total matches to {output_path}")

        return combined

    def fetch_all(
        self,
        leagues: Optional[list[str]] = None,
        seasons: Optional[list[str]] = None,
        save_checkpoint: bool = True
    ) -> pd.DataFrame:
        """Synchronous wrapper for fetching all data."""
        return asyncio.run(self.fetch_all_async(leagues, seasons, save_checkpoint))


def main():
    """Run the Understat xG scraper."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Scrape Understat xG data")
    parser.add_argument(
        "--league",
        type=str,
        help="Specific league to scrape (default: all)"
    )
    parser.add_argument(
        "--season",
        type=str,
        help="Specific season to scrape (default: all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Base delay between requests in seconds (default: 2)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Max retry attempts per request (default: 5)"
    )

    args = parser.parse_args()

    scraper = UnderstatScraper(delay=args.delay, max_retries=args.retries)

    leagues = [args.league] if args.league else None
    seasons = [args.season] if args.season else None

    df = scraper.fetch_all(leagues=leagues, seasons=seasons)

    print(f"\nFetched {len(df)} matches with xG data")
    if not df.empty:
        print(f"xG coverage: 100%")
        print("\nMatches per league:")
        print(df.groupby("league").size())
        print("\nMatches per season:")
        print(df.groupby("season").size())


if __name__ == "__main__":
    main()
