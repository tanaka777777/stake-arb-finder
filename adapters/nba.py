"""
NBA adapter - handles 2-way markets (moneyline, spreads, totals).
"""

from adapters.base import SportAdapter
from models import Game, Market


class NBAAdapter(SportAdapter):
    """NBA-specific parsing logic."""

    @property
    def sport(self) -> str:
        return "NBA"

    @property
    def market_types(self) -> list[str]:
        return ["moneyline", "spreads", "totals"]

    def normalize_team(self, name: str) -> str:
        """Normalize NBA team name to last word (e.g., 'Los Angeles Lakers' -> 'lakers')."""
        if not name:
            return ""
        words = name.lower().split()
        # Special cases
        if "trail" in words and "blazers" in words:
            return "blazers"
        if "76ers" in name.lower() or "sixers" in name.lower():
            return "76ers"
        return words[-1] if words else ""

    def parse_polymarket(self, raw_games: dict) -> list[Game]:
        """Parse Polymarket games dict into Game objects."""
        games = []
        for gid, data in raw_games.items():
            # gid is already the clean game ID (e.g., nba-lal-nop-2026-01-06)
            poly_url = f"https://polymarket.com/event/{gid}" if gid else ""

            game = Game(
                id=gid,
                sport=self.sport,
                away_team=data.get("away_team", ""),
                home_team=data.get("home_team", ""),
                game_date=data.get("game_date", ""),
                source="polymarket",
                url=poly_url,
                markets={"moneyline": [], "spreads": [], "totals": []},
            )

            # Moneyline: yes=away, no=home
            if data.get("moneyline"):
                ml = data["moneyline"]
                yes_ask = ml["yes"].get("ask")
                no_ask = ml["no"].get("ask")
                if yes_ask and no_ask:
                    game.markets["moneyline"].append(Market(
                        type="moneyline",
                        line=None,
                        outcomes={"yes": 1/yes_ask, "no": 1/no_ask}  # Convert prob to decimal
                    ))

            # Spreads: side indicates which team the line is for
            # YES = that team covers, NO = other team covers
            for spread in data.get("spreads", []):
                yes_ask = spread["yes"].get("ask")
                no_ask = spread["no"].get("ask")
                if yes_ask and no_ask and spread.get("line") is not None:
                    game.markets["spreads"].append(Market(
                        type="spread",
                        line=spread["line"],
                        outcomes={"yes": 1/yes_ask, "no": 1/no_ask},
                        side=spread.get("side"),  # "home" or "away"
                    ))

            # Totals: yes=over, no=under
            for total in data.get("totals", []):
                yes_ask = total["yes"].get("ask")
                no_ask = total["no"].get("ask")
                if yes_ask and no_ask and total.get("line") is not None:
                    game.markets["totals"].append(Market(
                        type="total",
                        line=total["line"],
                        outcomes={"yes": 1/yes_ask, "no": 1/no_ask}  # yes=over, no=under
                    ))

            games.append(game)
        return games

    def parse_stake(self, raw_data: dict, sport_key: str = "NBA") -> list[Game]:
        """Parse Stake odds-api.io output into Game objects."""
        games = []
        stake_games = raw_data.get(sport_key, {}).get("games", [])

        for data in stake_games:
            if not data.get("markets"):
                continue

            game = Game(
                id=data.get("id", ""),
                sport=sport_key,
                away_team=data.get("awayTeam", ""),
                home_team=data.get("homeTeam", ""),
                game_date=data.get("startDate", "")[:10] if data.get("startDate") else "",
                source="stake",
                url=data.get("url", ""),
                markets={"moneyline": [], "spreads": [], "totals": []},
            )

            markets = data["markets"]

            # Moneyline
            if markets.get("moneyline"):
                ml = markets["moneyline"]
                game.markets["moneyline"].append(Market(
                    type="moneyline",
                    line=None,
                    outcomes={"home": ml["home"]["odds"], "away": ml["away"]["odds"]}
                ))

            # Spreads: line is from home team perspective
            for spread in markets.get("spreads", []):
                game.markets["spreads"].append(Market(
                    type="spread",
                    line=spread["line"],  # Home team's line
                    outcomes={"home": spread["home"]["odds"], "away": spread["away"]["odds"]}
                ))

            # Totals
            for total in markets.get("totals", []):
                game.markets["totals"].append(Market(
                    type="total",
                    line=total["line"],
                    outcomes={"over": total["over"]["odds"], "under": total["under"]["odds"]}
                ))

            games.append(game)
        return games
