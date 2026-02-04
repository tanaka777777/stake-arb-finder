"""
Cache enrichment script for Polymarket asset IDs.

Fetches Polymarket markets and matches them to Bovada games in the cache,
storing asset_ids for WebSocket subscriptions.

Run daily or when match_cache.json is updated:
    python -m scripts.enrich_cache
"""

import json
import re
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("cache-enricher")

CACHE_FILE = Path(__file__).parent.parent / "match_cache.json"

SPORT_TAG_IDS = {
    "NBA": 745,
    "NHL": 899,
    "NFL": 450,
}


def fetch_polymarket_games(tag_id: int) -> dict:
    """
    Fetch and group markets by game, returning game_id -> game_data.
    Uses the existing polymarket.py logic.
    """
    from sources.polymarket import fetch_games
    return fetch_games(tag_id)


def parse_polymarket_market(market: dict) -> dict | None:
    """
    Parse a Polymarket market into structured format with asset IDs.

    Returns None for invalid/skipped markets.
    """
    q = market.get("question", "")
    slug = market.get("slug", "")

    # Skip half-time markets
    ql = q.lower()
    if "1h " in ql or "2h " in ql or "1st half" in ql or "2nd half" in ql:
        return None

    # Determine market type (traditional sports only)
    mtype = None
    line = None
    spread_side = None

    if "spread" in ql:
        mtype = "spread"
    elif "o/u" in ql:
        mtype = "total"
    elif "vs" in ql:
        mtype = "moneyline"

    if not mtype:
        return None

    # Extract game ID from slug
    match = re.match(r'([a-z]+-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2})', slug)
    if not match:
        return None
    game_id = match.group(1)

    # Parse line and spread side
    if mtype == "total":
        m = re.search(r'O/U\s*([\d.]+)', q)
        line = float(m.group(1)) if m else None
    elif mtype == "spread":
        m = re.search(r'\(([-+]?[\d.]+)\)', q)
        line = float(m.group(1)) if m else None
        side_match = re.search(r'spread-(home|away)', slug)
        spread_side = side_match.group(1) if side_match else None

    # Extract CLOB token IDs (asset IDs for WebSocket)
    clob_token_ids = market.get("clobTokenIds", "")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids) if clob_token_ids else []
        except:
            clob_token_ids = []

    # Token IDs correspond to outcomes: [YES, NO]
    yes_token = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    no_token = clob_token_ids[1] if len(clob_token_ids) > 1 else None

    # Parse date from slug
    parts = game_id.split("-")
    game_date = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"

    # Extract team slugs from game_id
    team_parts = game_id.split("-")[1:-3]
    away_slug = team_parts[0] if len(team_parts) > 0 else None
    home_slug = team_parts[1] if len(team_parts) > 1 else None

    return {
        "game_id": game_id,
        "game_date": game_date,
        "away_slug": away_slug,
        "home_slug": home_slug,
        "type": mtype,
        "line": line,
        "spread_side": spread_side,
        "slug": slug,
        "condition_id": market.get("conditionId"),
        "yes_token": yes_token,
        "no_token": no_token,
    }


def match_bovada_to_poly(bovada_match: dict, poly_games: dict, sport: str = "") -> tuple[str, bool] | None:
    """
    Find matching Polymarket game_id for a Bovada match.

    Returns tuple of (game_id, teams_swapped) or None if no match found.
    teams_swapped is True when Poly's team order is opposite of Bovada's.
    """
    # Extract date from Bovada match - handle timezone offset
    start_date = bovada_match.get("startDate", "")
    possible_dates = set()
    if start_date:
        try:
            dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            # Add both the UTC date and the day before (for timezone offset)
            possible_dates.add(dt.strftime("%Y-%m-%d"))
            possible_dates.add((dt - timedelta(days=1)).strftime("%Y-%m-%d"))
        except:
            pass

    # Get team names
    home_team = bovada_match.get("homeTeam", "")
    away_team = bovada_match.get("awayTeam", "")

    log.debug(f"Looking for: {away_team} @ {home_team} on {possible_dates}")

    # Try to find matching game in Polymarket data
    for game_id, game_data in poly_games.items():
        poly_date = game_data.get("game_date")
        if poly_date not in possible_dates:
            continue

        # Game ID format: "nba-hou-por-2026-01-07"
        parts = game_id.split("-")
        if len(parts) < 6:
            continue

        poly_team1_abbr = parts[1]
        poly_team2_abbr = parts[2]

        # Check normal order: Bovada away = Poly team1, Bovada home = Poly team2
        away_match_t1 = _abbr_match(away_team, poly_team1_abbr)
        home_match_t2 = _abbr_match(home_team, poly_team2_abbr)

        if away_match_t1 and home_match_t2:
            log.debug(f"  Matched: {game_id} (normal order)")
            return (game_id, False)

        # Check swapped order: Bovada away = Poly team2, Bovada home = Poly team1
        away_match_t2 = _abbr_match(away_team, poly_team2_abbr)
        home_match_t1 = _abbr_match(home_team, poly_team1_abbr)

        if away_match_t2 and home_match_t1:
            log.debug(f"  Matched: {game_id} (swapped order)")
            return (game_id, True)

    log.debug(f"  No match found")
    return None


# Team abbreviation mappings
TEAM_ABBRS = {
    # NBA
    "Atlanta Hawks": "atl",
    "Boston Celtics": "bos",
    "Brooklyn Nets": "bkn",
    "Charlotte Hornets": "cha",
    "Chicago Bulls": "chi",
    "Cleveland Cavaliers": "cle",
    "Dallas Mavericks": "dal",
    "Denver Nuggets": "den",
    "Detroit Pistons": "det",
    "Golden State Warriors": "gsw",
    "Houston Rockets": "hou",
    "Indiana Pacers": "ind",
    "Los Angeles Clippers": "lac",
    "Los Angeles Lakers": "lal",
    "Memphis Grizzlies": "mem",
    "Miami Heat": "mia",
    "Milwaukee Bucks": "mil",
    "Minnesota Timberwolves": "min",
    "New Orleans Pelicans": "nop",
    "New York Knicks": "nyk",
    "Oklahoma City Thunder": "okc",
    "Orlando Magic": "orl",
    "Philadelphia 76ers": "phi",
    "Phoenix Suns": "phx",
    "Portland Trail Blazers": "por",
    "Sacramento Kings": "sac",
    "San Antonio Spurs": "sas",
    "Toronto Raptors": "tor",
    "Utah Jazz": "uta",
    "Washington Wizards": "was",
    # NHL
    "Anaheim Ducks": "ana",
    "Arizona Coyotes": "ari",
    "Boston Bruins": "bos",
    "Buffalo Sabres": "buf",
    "Calgary Flames": "cal",
    "Carolina Hurricanes": "car",
    "Chicago Blackhawks": "chi",
    "Colorado Avalanche": "col",
    "Columbus Blue Jackets": "cbj",
    "Dallas Stars": "dal",
    "Detroit Red Wings": "det",
    "Edmonton Oilers": "edm",
    "Florida Panthers": "fla",
    "Los Angeles Kings": "lak",
    "Minnesota Wild": "min",
    "Montreal Canadiens": "mon",
    "Nashville Predators": "nsh",
    "New Jersey Devils": "nj",
    "New York Islanders": "nyi",
    "New York Rangers": "nyr",
    "Ottawa Senators": "ott",
    "Philadelphia Flyers": "phi",
    "Pittsburgh Penguins": "pit",
    "San Jose Sharks": "sj",
    "Seattle Kraken": "sea",
    "St. Louis Blues": "stl",
    "Tampa Bay Lightning": "tb",
    "Toronto Maple Leafs": "tor",
    "Vancouver Canucks": "van",
    "Vegas Golden Knights": "las",
    "Washington Capitals": "wsh",
    "Winnipeg Jets": "wpg",
    "Utah Hockey Club": "utah",
    # NFL
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "las",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "wsh",
}


def _abbr_match(team_name: str, abbr: str) -> bool:
    """Check if a team name matches an abbreviation."""
    # Direct mapping lookup
    if team_name in TEAM_ABBRS:
        return TEAM_ABBRS[team_name] == abbr

    # Fuzzy: check if abbr is a prefix of any word in team name
    words = team_name.lower().split()
    for word in words:
        if word.startswith(abbr) or abbr.startswith(word[:3]):
            return True
    return False


def enrich_cache() -> dict:
    """
    Enrich match_cache.json with Polymarket asset IDs.

    Returns the enriched cache data.
    """
    # Load existing cache
    if not CACHE_FILE.exists():
        log.error(f"Cache file not found: {CACHE_FILE}")
        return {}

    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    # Process each sport
    for sport, sport_data in cache.items():
        if sport not in SPORT_TAG_IDS:
            continue

        tag_id = SPORT_TAG_IDS[sport]

        log.info(f"[{sport}] Fetching Polymarket games (tag_id={tag_id})...")

        # Fetch games (already grouped by game_id)
        poly_games = fetch_polymarket_games(tag_id)
        log.info(f"[{sport}] Got {len(poly_games)} Polymarket games")

        # Also fetch raw markets for asset IDs
        from sources.polymarket import fetch_markets
        raw_markets = fetch_markets(tag_id)
        log.info(f"[{sport}] Got {len(raw_markets)} raw markets for asset IDs")

        # Build game_id -> asset IDs mapping
        assets_by_game: dict[str, list] = {}
        for market in raw_markets:
            parsed = parse_polymarket_market(market)
            if not parsed:
                continue

            game_id = parsed["game_id"]
            if game_id not in assets_by_game:
                assets_by_game[game_id] = []

            # Add YES token
            if parsed.get("yes_token"):
                assets_by_game[game_id].append({
                    "asset_id": parsed["yes_token"],
                    "market_type": parsed["type"],
                    "line": parsed.get("line"),
                    "side": parsed.get("spread_side"),
                    "outcome": "yes",
                    "slug": parsed["slug"],
                    "condition_id": parsed.get("condition_id"),
                })

            # Add NO token
            if parsed.get("no_token"):
                assets_by_game[game_id].append({
                    "asset_id": parsed["no_token"],
                    "market_type": parsed["type"],
                    "line": parsed.get("line"),
                    "side": parsed.get("spread_side"),
                    "outcome": "no",
                    "slug": parsed["slug"],
                    "condition_id": parsed.get("condition_id"),
                })

        # Match each Bovada game to Polymarket
        matches = sport_data.get("matches", [])
        matched_count = 0

        for bovada_match in matches:
            match_result = match_bovada_to_poly(bovada_match, poly_games, sport)

            if match_result:
                poly_game_id, teams_swapped = match_result
                assets = assets_by_game.get(poly_game_id, [])
                bovada_match["polymarket"] = {
                    "game_id": poly_game_id,
                    "assets": assets,
                    "teams_swapped": teams_swapped,
                }
                matched_count += 1
                swap_note = " (swapped)" if teams_swapped else ""
                log.info(f"  ✓ {bovada_match['name']} -> {poly_game_id}{swap_note} ({len(assets)} assets)")
            else:
                bovada_match["polymarket"] = None
                log.debug(f"  ✗ {bovada_match['name']} - no match")

        log.info(f"[{sport}] Matched {matched_count}/{len(matches)} games to Polymarket")

    # Save enriched cache
    cache["_enriched_at"] = datetime.now(timezone.utc).isoformat()
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    log.info(f"Saved enriched cache to {CACHE_FILE}")

    return cache


def load_asset_mapping_from_cache() -> dict[str, dict]:
    """
    Load asset mapping from enriched cache file.

    Returns dict mapping asset_id to market info for WebSocket subscription.
    """
    if not CACHE_FILE.exists():
        return {}

    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    mapping = {}

    for sport, sport_data in cache.items():
        if not isinstance(sport_data, dict) or "matches" not in sport_data:
            continue

        for match in sport_data["matches"]:
            poly_data = match.get("polymarket")
            if not poly_data:
                continue

            game_id = poly_data.get("game_id")

            for asset in poly_data.get("assets", []):
                asset_id = asset.get("asset_id")
                if asset_id:
                    mapping[asset_id] = {
                        "game_id": game_id,
                        "market_type": asset["market_type"],
                        "line": asset.get("line"),
                        "side": asset.get("side"),
                        "outcome": asset["outcome"],
                        "slug": asset.get("slug"),
                    }

    return mapping


if __name__ == "__main__":
    enrich_cache()
