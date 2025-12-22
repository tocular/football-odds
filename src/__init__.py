"""
Nostrus: Football betting model for Europe's top 5 leagues.

Core modules:
- scraping: Data collection from FBref, Club Elo, FotMob, and football-data.co.uk
- data: Database utilities and ETL pipeline
- features: Feature engineering for model training
- models: Dixon-Coles Poisson + GBM ensemble
- betting: Kelly criterion and backtesting
- automation: Scheduled job scripts
"""

__version__ = "0.1.0"
