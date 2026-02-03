"""
Stake odds fetcher via odds-api.io.

Fetches Stake odds for NBA, NHL, NFL and returns in the same format
as thunderpick.fetch_odds() so adapters work unchanged.
"""

import os
import logging
import requests
import time
from datetime import datetime, timezone, timedelta

log = logging.getLogger("arb-monitor")

# API Configuration
API_KEYS = [
    os.environ.get("ODDS_API_KEY_1", "c13742b92619a901ee790a91db9f4d529478f05b014ab75d7d8f831132d1eedc"),
    os.environ.get("ODDS_API_KEY_2", "33e219b22b94361eadb6f7a96989c17aa8a52720cbbf51f7dd30558247560193"),
    os.environ.get("ODDS_API_KEY_3", "c194a85186dfb29ac25431afb10709a66f0b62cf6d31be36f5754735fe7d7432"),
    os.environ.get("ODDS_API_KEY_4", "b897ea1890213e77f3dd93a7e9805419133ec9af539ef458f2f2865ab426196a"),
    os.environ.get("ODDS_API_KEY_5", "f6a828279f15233822fd35084b42e43c7056f0b960eb0eb89ca44437fe5a996d"),
    os.environ.get("ODDS_API_KEY_6", "d5f225920b0ca565ba553cc04a4b25331d74cc099918ecc706a51ccea12d43b4"),
]

EVENTS_URL = "https://api2.odds-api.io/v3/events"
ODDS_URL = "https://api2.odds-api.io/v3/odds/multi"

# League mappings for odds-api.io
LEAGUE_MAPPING = {
    "NBA": {"sport": "basketball", "league": "usa-nba"},
    "NHL": {"sport": "ice-hockey", "league": "usa-nhl"},
    "NFL": {"sport": "american-football", "league": "usa-nfl"},
}

# Current API key index (for rotation)
_current_key_index = 0


def _get_api_key() -> str:
    """Get current API key."""
    global _current_key_index
    return API_KEYS[_current_key_index]


def _rotate_api_key() -> bool:
    """Rotate to next API key. Returns False if all keys exhausted."""
    global _current_key_index
    _current_key_index += 1
    if _current_key_index >= len(API_KEYS):
        _current_key_index = 0
        return False
    return True


def _api_request(url: str, params: dict, retries: int = 3) -> dict | list | None:
    """
    Make API request with key rotation on rate limit.
    """
    for attempt in range(len(API_KEYS)):
        params["apiKey"] = _get_api_key()
        try:
            response = requests.get(url, params=params, timeout=30)

            # Check for rate limit
            if response.status_code == 429:
                log.warning(f"Rate limit hit, rotating API key...")
                if not _rotate_api_key():
                    log.error("All API keys exhausted")
                    return None
                continue

            response.raise_for_status()
            data = response.json()

            # Check for rate limit error in response body
            if isinstance(data, dict) and "error" in data:
                if "rate limit" in data.get("error", "").lower():
                    log.warning(f"Rate limit in response, rotating API key...")
                    if not _rotate_api_key():
                        log.error("All API keys exhausted")
                        return None
                    continue

            return data

        except requests.RequestException as e:
            log.error(f"API request failed: {e}")
            if attempt < len(API_KEYS) - 1:
                _rotate_api_key()
            continue

    return None


def _fetch_events(sport: str, league: str) -> list:
    """Fetch all events for a specific sport/league."""
    params = {
        "sport": sport,
        "league": league,
        "bookmaker": "Stake",
    }

    events = _api_request(EVENTS_URL, params)
    if events is None or not isinstance(events, list):
        return []

    # Filter to next 48 hours, exclude settled
    cutoff_time = datetime.now(timezone.utc) + timedelta(hours=48)
    filtered = []

    for event in events:
        if event.get("status") == "settled":
            continue

        event_date_str = event.get("date")
        if event_date_str:
            try:
                event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
                if event_date <= cutoff_time:
                    filtered.append(event)
            except Exception:
                continue

    return filtered


def _fetch_odds_batch(event_ids: list[str]) -> list:
    """Fetch odds for a batch of event IDs."""
    if not event_ids:
        return []

    params = {
        "eventIds": ",".join(map(str, event_ids)),
        "bookmakers": "Stake",
    }

    odds = _api_request(ODDS_URL, params)
    if odds is None or not isinstance(odds, list):
        return []

    return odds


def _convert_nhl_3way_to_2way(home_3way: float, draw_3way: float, away_3way: float) -> tuple[float, float]:
    """
    Convert NHL 3-way odds to 2-way moneyline by removing draw probability.

    Returns (home_decimal, away_decimal) with 2% vig applied.
    """
    if home_3way <= 0 or away_3way <= 0:
        return 0, 0

    # Convert to probabilities
    home_prob = 1 / home_3way
    away_prob = 1 / away_3way

    # Redistribute draw probability proportionally
    total_prob = home_prob + away_prob
    home_prob_2way = home_prob / total_prob
    away_prob_2way = away_prob / total_prob

    # Add 2% vig (bookmaker margin)
    vig_factor = 1.02
    home_prob_2way_vig = home_prob_2way * vig_factor
    away_prob_2way_vig = away_prob_2way * vig_factor

    # Convert back to decimal odds
    home_decimal = 1 / home_prob_2way_vig
    away_decimal = 1 / away_prob_2way_vig

    return home_decimal, away_decimal


def _extract_markets(event_odds: dict, league: str) -> dict | None:
    """
    Extract markets from Stake odds data.

    Returns dict in thunderpick format:
    {
        "moneyline": {"home": {"odds": X}, "away": {"odds": X}},
        "spreads": [{"line": X, "home": {"odds": X}, "away": {"odds": X}}],
        "totals": [{"line": X, "over": {"odds": X}, "under": {"odds": X}}],
    }
    """
    if "bookmakers" not in event_odds or "Stake" not in event_odds["bookmakers"]:
        return None

    stake_markets = event_odds["bookmakers"]["Stake"]
    result = {"moneyline": None, "spreads": [], "totals": []}

    # For all sports (including NHL), use ML market directly if available
    # NHL has both "3-Way Result" and "ML" markets - use ML for accurate 2-way odds
    ml_market = None
    for market in stake_markets:
        if market["name"] == "ML":
            ml_market = market
            break

    if ml_market and ml_market.get("odds"):
        odds_data = ml_market["odds"][0]
        home = float(odds_data.get("home", 0))
        away = float(odds_data.get("away", 0))
        if home > 0 and away > 0:
            result["moneyline"] = {
                "home": {"odds": home},
                "away": {"odds": away},
            }
    elif league == "usa-nhl":
        # Fallback: convert 3-way to 2-way if ML not available
        threeway_market = None
        for market in stake_markets:
            if market["name"] == "3-Way Result":
                threeway_market = market
                break

        if threeway_market and threeway_market.get("odds"):
            odds_data = threeway_market["odds"][0]
            home_3way = float(odds_data.get("home", 0))
            draw_3way = float(odds_data.get("draw", 0))
            away_3way = float(odds_data.get("away", 0))

            if home_3way > 0 and away_3way > 0:
                home_2way, away_2way = _convert_nhl_3way_to_2way(home_3way, draw_3way, away_3way)
                if home_2way > 0 and away_2way > 0:
                    result["moneyline"] = {
                        "home": {"odds": home_2way},
                        "away": {"odds": away_2way},
                    }
    else:
        # For other sports, use ML market
        for market in stake_markets:
            if market["name"] == "ML":
                if market.get("odds"):
                    odds_data = market["odds"][0]
                    home = float(odds_data.get("home", 0))
                    away = float(odds_data.get("away", 0))
                    if home > 0 and away > 0:
                        result["moneyline"] = {
                            "home": {"odds": home},
                            "away": {"odds": away},
                        }
                break

    # Extract spreads (Handicap market)
    for market in stake_markets:
        if market["name"] == "Handicap":
            for odds_entry in market.get("odds", []):
                line = odds_entry.get("hdp")
                home = float(odds_entry.get("home", 0))
                away = float(odds_entry.get("away", 0))
                if line is not None and home > 0 and away > 0:
                    result["spreads"].append({
                        "line": float(line),
                        "home": {"odds": home},
                        "away": {"odds": away},
                    })
            break

    # Extract totals
    for market in stake_markets:
        if market["name"] == "Totals":
            for odds_entry in market.get("odds", []):
                line = odds_entry.get("hdp")
                over = float(odds_entry.get("over", 0))
                under = float(odds_entry.get("under", 0))
                if line is not None and over > 0 and under > 0:
                    result["totals"].append({
                        "line": float(line),
                        "over": {"odds": over},
                        "under": {"odds": under},
                    })
            break

    # Only return if we have at least moneyline
    if not result["moneyline"]:
        return None

    return result


def fetch_odds(leagues: list[str] = None) -> dict | None:
    """
    Fetch Stake odds via odds-api.io.

    Args:
        leagues: List of league keys (e.g., ["NBA", "NHL"]). If None, fetches all.

    Returns same format as thunderpick.fetch_odds() so adapters work unchanged:
    {
        "NBA": {
            "gamesCount": 5,
            "games": [
                {
                    "id": "event123",
                    "homeTeam": "Celtics",
                    "awayTeam": "Lakers",
                    "startDate": "2026-02-05T00:00:00Z",
                    "url": "https://stake.com/...",
                    "markets": {
                        "moneyline": {"home": {"odds": 1.85}, "away": {"odds": 2.05}},
                        "spreads": [{"line": -4.5, "home": {"odds": 1.91}, "away": {"odds": 1.91}}],
                        "totals": [{"line": 220.5, "over": {"odds": 1.90}, "under": {"odds": 1.90}}]
                    }
                }
            ]
        }
    }
    """
    if leagues is None:
        leagues = list(LEAGUE_MAPPING.keys())

    result = {}

    for league_key in leagues:
        if league_key not in LEAGUE_MAPPING:
            log.warning(f"Unknown league: {league_key}")
            continue

        config = LEAGUE_MAPPING[league_key]
        log.info(f"[{league_key}] Fetching events from odds-api.io...")

        # Step 1: Fetch events
        events = _fetch_events(config["sport"], config["league"])
        if not events:
            log.warning(f"[{league_key}] No events found")
            result[league_key] = {"gamesCount": 0, "games": []}
            continue

        log.info(f"[{league_key}] Found {len(events)} events")

        # Step 2: Fetch odds in batches of 10
        event_ids = [e["id"] for e in events]
        all_odds = []

        for i in range(0, len(event_ids), 10):
            batch = event_ids[i:i + 10]
            log.debug(f"[{league_key}] Fetching odds batch {i // 10 + 1}...")
            batch_odds = _fetch_odds_batch(batch)
            all_odds.extend(batch_odds)

            # Small delay between batches to avoid rate limits
            if i + 10 < len(event_ids):
                time.sleep(0.3)

        log.info(f"[{league_key}] Got odds for {len(all_odds)} events")

        # Step 3: Process into games
        games = []
        for event_odds in all_odds:
            markets = _extract_markets(event_odds, config["league"])
            if not markets:
                continue

            # Build Stake URL
            stake_url = event_odds.get("urls", {}).get("Stake", "")
            if not stake_url:
                # Construct URL if not provided
                stake_url = f"https://stake.com/sports/{config['sport']}"

            game = {
                "id": str(event_odds.get("id", "")),
                "homeTeam": event_odds.get("home", ""),
                "awayTeam": event_odds.get("away", ""),
                "startDate": event_odds.get("date", ""),
                "url": stake_url,
                "markets": markets,
            }
            games.append(game)

        result[league_key] = {
            "gamesCount": len(games),
            "games": games,
        }

        log.info(f"[{league_key}] Processed {len(games)} games with Stake odds")

    return result


if __name__ == "__main__":
    # Test standalone
    logging.basicConfig(level=logging.INFO)
    odds = fetch_odds(["NBA"])
    if odds:
        import json
        print(json.dumps(odds, indent=2))
