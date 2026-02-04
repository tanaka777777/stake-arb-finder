"""
Data models for Arb Monitor.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Market:
    """A betting market (ML, spread, or total)."""
    type: str  # "moneyline" | "spread" | "total"
    line: float | None  # None for moneyline, number for spread/total
    outcomes: dict[str, float]  # {"yes": 2.10, "no": 1.85} or {"home": 1.95, "away": 2.05}
    side: str | None = None  # For spreads: "home" or "away" - which team the line is for
    sizes: dict[str, float] | None = None  # {"yes": 100, "no": 200} - shares at best ask


@dataclass
class Game:
    """A game with markets from one source."""
    id: str
    sport: str
    away_team: str
    home_team: str
    game_date: str
    source: str
    url: str = ""  # Link to the game page
    markets: dict[str, list[Market]] = field(default_factory=dict)  # {"moneyline": [...], "spreads": [...]}


@dataclass
class Arb:
    """An arbitrage opportunity."""
    id: str  # Hash for tracking
    game_id: str
    game_display: str  # "Magic @ Wizards"
    sport: str
    market_type: str  # "moneyline" | "spread" | "total"
    line: float | None
    profit_pct: float

    # Poly side
    poly_side: str  # "yes" | "no"
    poly_odds: float
    poly_stake_pct: float  # Optimal stake %

    # TP side (kept as tp_ for compatibility - represents Bovada in this repo)
    tp_side: str  # "home" | "away" | "over" | "under"
    tp_odds: float
    tp_stake_pct: float

    # Optional fields (must come after required fields)
    poly_size: float | None = None  # Shares available at best ask
    poly_url: str = ""
    tp_url: str = ""
    tp_line: float | None = None  # Bovada line for spreads (may differ from poly line)

    # Tracking
    found_at: datetime = field(default_factory=datetime.now)
    closed_at: datetime | None = None

    def to_log_dict(self) -> dict:
        """Convert to JSON-serializable dict for logging."""
        return {
            "id": self.id,
            "game": self.game_display,
            "sport": self.sport,
            "market_type": self.market_type,
            "line": self.line,
            "profit_pct": round(self.profit_pct, 2),
            "poly_side": self.poly_side,
            "poly_odds": self.poly_odds,
            "poly_size": self.poly_size,
            "poly_url": self.poly_url,
            "tp_side": self.tp_side,
            "tp_odds": self.tp_odds,
            "tp_url": self.tp_url,
            "tp_line": self.tp_line,
            "found_at": self.found_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }
