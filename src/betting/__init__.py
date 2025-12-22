"""Betting utilities for Nostrus."""

from .kelly import KellyCalculator
from .backtest import Backtester

__all__ = ["KellyCalculator", "Backtester"]
