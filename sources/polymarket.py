"""
Polymarket data source.
"""

import requests
import re
from datetime import datetime, timezone, timedelta
from collections import defaultdict

API_BASE = "https://gamma-api.polymarket.com"


def fetch_markets(tag_id: int, hours: int = 48) -> list:
    """
    Fetch markets from Polymarket by tag with date filtering and pagination.

    Uses end_date_min/max to filter server-side, then paginates to get all results.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)

    all_markets = []
    offset = 0
    limit = 500

    while True:
        r = requests.get(f"{API_BASE}/markets", params={
            "tag_id": tag_id,
            "limit": limit,
            "offset": offset,
            "order": "endDate",
            "ascending": True,
            "closed": False,
            "end_date_min": now.isoformat(),
            "end_date_max": cutoff.isoformat(),
        })
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        all_markets.extend(batch)

        if len(batch) < limit:
            break

        offset += limit

    return all_markets


def parse_market(market: dict) -> dict | None:
    """Parse a single market into standardized format."""
    q = market.get("question", "")
    slug = market.get("slug", "")

    # Skip half-time markets (1H, 2H, 1st half, etc.)
    ql = q.lower()
    if "1h " in ql or "2h " in ql or "1st half" in ql or "2nd half" in ql:
        return None

    # Determine market type (traditional sports only - no LoL)
    mtype = None
    if "spread" in ql:
        mtype = "spread"
    elif "o/u" in ql:
        mtype = "total"
    elif "vs." in q:
        mtype = "moneyline"

    if not mtype:
        return None

    # Extract game ID from slug
    match = re.match(r'([a-z]+-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2})', slug)
    if not match:
        return None
    game_id = match.group(1)

    # Parse line and spread side (home/away) from slug
    line = None
    spread_side = None  # 'home' or 'away'
    if mtype == "total":
        m = re.search(r'O/U\s*([\d.]+)', q)
        line = float(m.group(1)) if m else None
    elif mtype == "spread":
        # Try parentheses first: "Spread (-4.5)"
        m = re.search(r'\(([-+]?[\d.]+)\)', q)
        if not m:
            # Try without parentheses
            m = re.search(r'spread\s*([-+]?[\d.]+)', ql)
        line = float(m.group(1)) if m else None
        # Extract side from slug: "spread-home-4pt5" or "spread-away-5pt5"
        side_match = re.search(r'spread-(home|away)', slug)
        spread_side = side_match.group(1) if side_match else None

    # Parse odds
    best_bid = float(market["bestBid"]) if market.get("bestBid") else None
    best_ask = float(market["bestAsk"]) if market.get("bestAsk") else None

    # Parse date from slug
    parts = game_id.split("-")
    game_date = f"{parts[3]}-{parts[4]}-{parts[5]}"

    # Extract team names from outcomes
    away_team, home_team, teams = None, None, None
    try:
        outcomes = eval(market.get("outcomes", "[]"))
        if len(outcomes) >= 2:
            away_team = outcomes[0]
            home_team = outcomes[1]
            teams = f"{away_team} vs {home_team}"
    except:
        pass

    if not teams:
        away_team = parts[1].upper()
        home_team = parts[2].upper()
        teams = f"{away_team} vs {home_team}"

    return {
        "game_id": game_id,
        "game_date": game_date,
        "teams": teams,
        "away_team": away_team,
        "home_team": home_team,
        "type": mtype,
        "line": line,
        "spread_side": spread_side,  # 'home' or 'away' - which team the spread is for
        "slug": slug,
        "yes": {"bid": best_bid, "ask": best_ask},
        "no": {"bid": 1 - best_ask if best_ask else None, "ask": 1 - best_bid if best_bid else None},
    }


def fetch_games(tag_id: int, hours: int = 48) -> dict:
    """
    Fetch and group markets by game.

    Note: Use market endDate (UTC) to decide the window instead of slug date.
    This avoids dropping late-night games whose slug date is "yesterday" in UTC.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)

    markets = fetch_markets(tag_id, hours)
    games = defaultdict(lambda: {
        "moneyline": None, "spreads": [], "totals": [],
        "away_team": None, "home_team": None, "game_date": None, "slug": None
    })

    for market in markets:
        parsed = parse_market(market)
        if not parsed:
            continue

        # Prefer the market endDate (UTC). Fallback to slug date if missing.
        end_date = market.get("endDate")
        game_dt = None
        if end_date:
            try:
                game_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except Exception:
                game_dt = None

        if game_dt is None:
            try:
                game_dt = datetime.strptime(parsed["game_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue

        # Keep only games within the lookahead window
        if game_dt < now or game_dt > cutoff:
            continue

        gid = parsed["game_id"]
        # Only set team names if not already set or if current names are invalid (Over/Under from totals)
        if not games[gid]["away_team"] or games[gid]["away_team"] in ("Over", "Under"):
            games[gid]["away_team"] = parsed["away_team"]
        if not games[gid]["home_team"] or games[gid]["home_team"] in ("Over", "Under"):
            games[gid]["home_team"] = parsed["home_team"]
        games[gid]["game_date"] = parsed["game_date"]
        if parsed["slug"] and not games[gid]["slug"]:
            games[gid]["slug"] = parsed["slug"]

        entry = {"line": parsed["line"], "yes": parsed["yes"], "no": parsed["no"]}
        mtype = parsed["type"]

        if mtype == "moneyline":
            games[gid]["moneyline"] = entry
        elif mtype == "spread" and parsed["line"]:
            entry["side"] = parsed.get("spread_side")  # 'home' or 'away'
            games[gid]["spreads"].append(entry)
        elif mtype == "total" and parsed["line"]:
            games[gid]["totals"].append(entry)

    for g in games.values():
        g["spreads"].sort(key=lambda x: x["line"])
        g["totals"].sort(key=lambda x: x["line"])

    return dict(games)
