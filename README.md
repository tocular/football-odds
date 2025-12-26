# Football betting model using Dixon-Coles + GBM ensemble for match prediction.

DISCLAIMER: This model will absolutely NOT make you money.

## Model Architecture

1. **Dixon-Coles Poisson Model** - Estimates team attack/defense strengths with time-decay weighting
2. **GBM Correction Layer** - Adjusts expected goals using 27 features (form, Elo, H2H, rest)
3. **Temperature Calibration** - Ensures probabilities are well-calibrated
4. **Kelly Criterion** - Optimal bet sizing with configurable risk parameters

## Data

- Matches from 5 leagues **(Premier League, La Liga, Bundesliga, Serie A, Ligue 1)**
- **Seasons**: 2015-2016 to 2024-2025
- **xG coverage**: 99.96% (Understat)
- **Odds**: Pinnacle closing lines

## Features

### Odds & CLV Analysis
- Uses **Pinnacle closing odds** for backtesting
- **Closing Line Value (CLV)** tracking to measure bet quality
- Positive CLV indicates genuine predictive edge, not luck

### Betting Strategy (Tuned)
- **8% minimum edge** threshold
- **10% Kelly fraction** (conservative sizing)
- **5.0 max odds**
- **Excludes draws**

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline (data loading, training, backtest)
python run_pipeline.py

# Collect fresh data
python -m src.automation.collect_data
```

## Structure

```
src/
  scraping/       # Data collection (Understat, odds)
  data/           # ETL pipeline, database models
  features/       # Feature engineering (Elo, form, H2H, rest)
  models/         # Dixon-Coles, GBM corrector, calibration
  betting/        # Kelly criterion, backtesting with CLV
  automation/     # Automated data collection
config/           # Configuration
data/
  nostrus.db      # SQLite database
  raw/            # Raw CSV files
models/           # Trained models and reports
```

## Config

Edit `config/config.yaml`:

```yaml
model:
  min_edge_threshold: 0.08   # Minimum edge to place bet (8%)
  kelly_fraction: 0.10       # Conservative Kelly (10%)
  max_bet_pct: 0.02          # Max 2% of bankroll per bet
  max_odds: 5.0              # Avoid longshots
  exclude_draws: true        # Skip draw bets
```

## Key Files Modified

- `src/features/builder.py` - Pinnacle closing odds integration
- `src/betting/backtest.py` - CLV calculation and tracking
- `run_pipeline.py` - CLV reporting in output
