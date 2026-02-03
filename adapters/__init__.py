"""Sport adapters."""

from .nba import NBAAdapter

# NBA, NHL, and NFL use same 2-way market logic
ADAPTERS = {
    "NBA": NBAAdapter(),
    "NHL": NBAAdapter(),
    "NFL": NBAAdapter(),
}


def get_adapter(sport: str):
    """Get adapter for a sport."""
    if sport not in ADAPTERS:
        raise ValueError(f"No adapter for sport: {sport}")
    return ADAPTERS[sport]
