"""
Base adapter interface for sports.
"""

from abc import ABC, abstractmethod
from models import Game, Market


class SportAdapter(ABC):
    """Abstract base for sport-specific parsing logic."""

    @property
    @abstractmethod
    def sport(self) -> str:
        """Sport identifier (e.g., 'NBA')."""
        pass

    @property
    @abstractmethod
    def market_types(self) -> list[str]:
        """Supported market types for this sport."""
        pass

    @abstractmethod
    def normalize_team(self, name: str) -> str:
        """Normalize team name for matching across sources."""
        pass

    @abstractmethod
    def parse_polymarket(self, raw_games: dict) -> list[Game]:
        """Parse Polymarket API response into Game objects."""
        pass

    @abstractmethod
    def parse_bovada(self, raw_data: dict, sport_key: str) -> list['Game']:
        """Parse Bovada odds-api.io output into Game objects."""
        pass
