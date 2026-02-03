"""
Arb Monitor - Real-time arbitrage detection between Polymarket and Stake.

Uses WebSocket for real-time Polymarket prices.
Stake odds fetched via odds-api.io (no Puppeteer needed).
"""

import json
import logging
import threading
import time
import traceback

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from flask import Flask, Response, render_template, jsonify

import config
from models import Arb
from adapters import get_adapter
from sources import fetch_polymarket, fetch_stake, get_price_provider
from core import match_games, find_arbs
from scripts.enrich_cache import load_asset_mapping_from_cache, enrich_cache as run_enrich_cache

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("arb-monitor")


# =============================================================================
# STATUS TRACKING
# =============================================================================

class Status(Enum):
    IDLE = "idle"
    FETCHING_POLY = "fetching_polymarket"
    FETCHING_STAKE = "fetching_stake"
    PROCESSING = "processing"
    ERROR = "error"
    WS_CONNECTING = "ws_connecting"
    WS_CONNECTED = "ws_connected"


class DataMode(Enum):
    REST = "rest"
    WEBSOCKET = "websocket"


@dataclass
class PipelineState:
    """Tracks the current state of the data pipeline."""
    status: Status = Status.IDLE
    current_sport: str = ""
    last_error: str = ""
    last_success: datetime | None = None
    data_mode: DataMode = DataMode.REST
    ws_connected: bool = False
    ws_assets_subscribed: int = 0

    # Counters for current cycle
    poly_games_fetched: int = 0
    stake_games_fetched: int = 0
    games_matched: int = 0
    arbs_found: int = 0

    # Fetch timestamps
    last_poly_fetch: datetime | None = None
    last_stake_fetch: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "current_sport": self.current_sport,
            "last_error": self.last_error,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "data_mode": self.data_mode.value,
            "ws_connected": self.ws_connected,
            "ws_assets": self.ws_assets_subscribed,
            "poly_games": self.poly_games_fetched,
            "tp_games": self.stake_games_fetched,  # Keep tp_games for UI compatibility
            "matched": self.games_matched,
            "arbs": self.arbs_found,
            "last_poly_fetch": self.last_poly_fetch.isoformat() if self.last_poly_fetch else None,
            "last_tp_fetch": self.last_stake_fetch.isoformat() if self.last_stake_fetch else None,
        }


# =============================================================================
# APP STATE
# =============================================================================

app = Flask(__name__)

# Thread-safe state
state_lock = threading.Lock()
current_arbs: list[Arb] = []
arb_history: dict[str, Arb] = {}
matched_no_arb: list[dict] = []
last_update: datetime | None = None
pipeline = PipelineState()
subscribers: list = []

# WebSocket price provider
price_provider = get_price_provider()
ws_game_mapping: dict[str, dict] = {}  # game_id -> {"sport": str, "poly_game": Game, "stake_game": Game}

# Cached Stake data for instant arb recalculation
cached_stake_games: dict[str, list] = {}  # sport -> list of Stake Game objects
cached_match_info: list[dict] = []  # List of match info from cache file
last_stake_fetch_time: datetime | None = None


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_name(home: str, away: str) -> str:
    """Normalize team names for matching."""
    def clean(s):
        return s.lower().replace(".", "").replace("-", " ").strip()
    return f"{clean(away)}|{clean(home)}"


def _game_has_started(start_date: str) -> bool:
    """Check if game has already started based on startDate."""
    if not start_date:
        return False
    try:
        game_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        now = datetime.now(game_time.tzinfo)
        return now > game_time
    except Exception:
        return False


def _build_poly_game_from_ws(game_id: str, sport: str, match: dict, prices: dict) -> 'Game | None':
    """
    Build a Polymarket Game object from WebSocket prices.

    Args:
        game_id: The polymarket game ID
        sport: Sport key (e.g., "NBA")
        match: Match data from cache containing team names
        prices: Prices dict from price_provider.get_all_prices_for_game()

    Returns:
        Game object or None if no prices available
    """
    from models import Game, Market

    # Check if we have any markets
    has_markets = (prices.get("moneyline") or prices.get("spreads") or prices.get("totals"))
    if not has_markets:
        return None

    markets = {}

    # Moneyline - use ASK price (what we PAY to buy)
    if prices.get("moneyline"):
        ml = prices["moneyline"]
        yes_ask = ml["yes"].get("ask", 0)
        no_ask = ml["no"].get("ask", 0)
        if yes_ask > 0 and no_ask > 0:
            markets["moneyline"] = [Market(
                type="moneyline",
                line=None,
                outcomes={"yes": 1 / yes_ask, "no": 1 / no_ask},
                sizes={"yes": ml["yes"].get("ask_size"), "no": ml["no"].get("ask_size")}
            )]

    # Spreads - use ASK price
    if prices.get("spreads"):
        spread_markets = []
        for sp in prices["spreads"]:
            yes_ask = sp["yes"].get("ask", 0)
            no_ask = sp["no"].get("ask", 0)
            if yes_ask > 0 and no_ask > 0:
                spread_markets.append(Market(
                    type="spread",
                    line=sp.get("line"),
                    side=sp.get("side"),
                    outcomes={"yes": 1 / yes_ask, "no": 1 / no_ask},
                    sizes={"yes": sp["yes"].get("ask_size"), "no": sp["no"].get("ask_size")}
                ))
        if spread_markets:
            markets["spreads"] = spread_markets

    # Totals - use ASK price
    if prices.get("totals"):
        total_markets = []
        for tot in prices["totals"]:
            yes_ask = tot["yes"].get("ask", 0)
            no_ask = tot["no"].get("ask", 0)
            if yes_ask > 0 and no_ask > 0:
                total_markets.append(Market(
                    type="total",
                    line=tot.get("line"),
                    outcomes={"yes": 1 / yes_ask, "no": 1 / no_ask},
                    sizes={"yes": tot["yes"].get("ask_size"), "no": tot["no"].get("ask_size")}
                ))
        if total_markets:
            markets["totals"] = total_markets

    if not markets:
        return None

    return Game(
        id=game_id,
        sport=sport,
        home_team=match.get("homeTeam", ""),
        away_team=match.get("awayTeam", ""),
        game_date=[],
        source="polymarket",
        url=f"https://polymarket.com/event/{game_id}" if game_id else "",
        markets=markets
    )


def _update_arb_state(all_arbs: list[Arb], all_matched_no_arb: list[dict],
                      total_matched: int, total_stake: int = 0,
                      total_poly: int = 0, is_ws_mode: bool = False) -> None:
    """
    Update global arb state and notify subscribers.
    Handles tracking of new/closed arbs and updates pipeline counters.
    """
    global current_arbs, matched_no_arb, last_update, pipeline

    all_arbs.sort(key=lambda a: a.profit_pct, reverse=True)

    with state_lock:
        current_ids = {a.id for a in all_arbs}
        prev_ids = {a.id for a in current_arbs}

        # Track closed arbs
        for arb in current_arbs:
            if arb.id not in current_ids and arb.id in arb_history:
                arb_history[arb.id].closed_at = datetime.now()
                if arb_history[arb.id].profit_pct >= config.LOG_THRESHOLD_PCT:
                    log_arb(arb_history[arb.id])

        # Track new arbs
        new_arbs = [a for a in all_arbs if a.id not in prev_ids]
        for arb in new_arbs:
            arb_history[arb.id] = arb
            log.info(f"NEW ARB: {arb.game_display} | {arb.market_type} | +{arb.profit_pct:.2f}%")

        # Check for changed profit on existing arbs (WS mode only)
        if is_ws_mode:
            for arb in all_arbs:
                if arb.id in prev_ids:
                    old_arb = next((a for a in current_arbs if a.id == arb.id), None)
                    if old_arb and abs(arb.profit_pct - old_arb.profit_pct) > 0.1:
                        log.info(f"ARB UPDATE: {arb.game_display} | {old_arb.profit_pct:.2f}% -> {arb.profit_pct:.2f}%")

        current_arbs = all_arbs
        matched_no_arb = all_matched_no_arb
        last_update = datetime.now()

        # Update pipeline counters
        if is_ws_mode:
            pipeline.status = Status.WS_CONNECTED
            pipeline.ws_connected = price_provider.connected
        else:
            pipeline.status = Status.IDLE

        pipeline.current_sport = ""
        pipeline.games_matched = total_matched
        pipeline.arbs_found = len(all_arbs)
        pipeline.last_success = datetime.now()
        pipeline.last_error = ""

        if total_stake > 0:
            pipeline.stake_games_fetched = total_stake
        if total_poly > 0:
            pipeline.poly_games_fetched = total_poly

    notify_subscribers()


# =============================================================================
# DATA PIPELINE
# =============================================================================

def refresh_data():
    """Fetch fresh data and detect arbs (REST mode)."""
    global current_arbs, matched_no_arb, last_update, pipeline

    cycle_start = datetime.now()
    log.info("=" * 50)
    log.info("Starting refresh cycle")

    all_arbs = []
    all_matched_no_arb = []
    total_poly = 0
    total_stake = 0
    total_matched = 0

    for sport in config.ACTIVE_SPORTS:
        log.info(f"[{sport}] Processing...")

        with state_lock:
            pipeline.status = Status.FETCHING_POLY
            pipeline.current_sport = sport
        notify_subscribers()

        sport_config = config.SPORTS[sport]
        adapter = get_adapter(sport)

        # Fetch Polymarket
        log.info(f"[{sport}] Fetching Polymarket (tag_id={sport_config['polymarket_tag_id']})...")
        poly_start = time.time()
        poly_raw = fetch_polymarket(sport_config["polymarket_tag_id"])
        poly_elapsed = time.time() - poly_start

        if not poly_raw:
            log.warning(f"[{sport}] Polymarket fetch failed")
            with state_lock:
                pipeline.last_error = f"{sport}: Polymarket fetch failed"
            continue

        poly_games = adapter.parse_polymarket(poly_raw)
        log.info(f"[{sport}] Polymarket: {len(poly_games)} games ({poly_elapsed:.1f}s)")
        total_poly += len(poly_games)

        with state_lock:
            pipeline.status = Status.FETCHING_STAKE
            pipeline.last_poly_fetch = datetime.now()
        notify_subscribers()

        # Fetch Stake
        log.info(f"[{sport}] Fetching Stake...")
        stake_start = time.time()
        stake_raw = fetch_stake([sport])
        stake_elapsed = time.time() - stake_start

        if not stake_raw:
            log.warning(f"[{sport}] Stake fetch failed")
            with state_lock:
                pipeline.last_error = f"{sport}: Stake fetch failed"
            continue

        stake_games = adapter.parse_stake(stake_raw, sport)
        log.info(f"[{sport}] Stake: {len(stake_games)} games ({stake_elapsed:.1f}s)")
        total_stake += len(stake_games)

        with state_lock:
            pipeline.status = Status.PROCESSING
            pipeline.last_stake_fetch = datetime.now()
        notify_subscribers()

        # Match games
        matched = match_games(poly_games, stake_games, adapter)
        log.info(f"[{sport}] Matched: {len(matched)} games")
        total_matched += len(matched)

        # Find arbs
        sport_arbs = 0
        for poly_game, stake_game, is_swapped in matched:
            game_arbs = find_arbs(poly_game, stake_game, is_swapped)
            profitable = [a for a in game_arbs if a.profit_pct >= config.MIN_PROFIT_PCT]
            all_arbs.extend(profitable)
            sport_arbs += len(profitable)

            if not profitable:
                all_matched_no_arb.append({
                    "sport": sport,
                    "game": f"{poly_game.away_team} @ {poly_game.home_team}",
                    "poly_url": poly_game.url,
                    "tp_url": stake_game.url,
                    "poly_markets": list(poly_game.markets.keys()),
                    "tp_markets": list(stake_game.markets.keys()),
                })

        log.info(f"[{sport}] Arbs found: {sport_arbs}")

    # Update state and notify subscribers
    _update_arb_state(all_arbs, all_matched_no_arb, total_matched,
                      total_stake=total_stake, total_poly=total_poly, is_ws_mode=False)

    cycle_elapsed = (datetime.now() - cycle_start).total_seconds()
    log.info(f"Cycle complete: {total_poly} poly, {total_stake} stake, {total_matched} matched, {len(all_arbs)} arbs ({cycle_elapsed:.1f}s)")
    log.info("=" * 50)


# =============================================================================
# WEBSOCKET MODE
# =============================================================================

def init_websocket_mode():
    """Initialize WebSocket mode - enrich cache and start price feed."""
    global ws_game_mapping, pipeline

    log.info("[WS_INIT] Starting WebSocket mode initialization...")

    with state_lock:
        pipeline.status = Status.WS_CONNECTING
        pipeline.data_mode = DataMode.WEBSOCKET
    notify_subscribers()

    # Check cache before enrichment
    cache_path = Path("match_cache.json")
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            for sport, data in cache.items():
                if isinstance(data, dict):
                    matches = data.get("matches", [])
                    poly_count = sum(1 for m in matches if m.get("polymarket"))
                    log.info(f"[WS_INIT] {sport}: {len(matches)} matches, {poly_count} with polymarket data")
        except Exception as e:
            log.error(f"[WS_INIT] Cache read error: {e}")

    # Enrich cache with Polymarket asset IDs
    log.info("[WS_INIT] Running cache enrichment...")
    try:
        run_enrich_cache()
        log.info("[WS_INIT] Cache enrichment completed")
    except Exception as e:
        log.error(f"[WS_INIT] Cache enrichment failed: {e}")
        with state_lock:
            pipeline.last_error = f"Cache enrichment failed: {e}"
        return False

    # Load asset mapping
    log.info("[WS_INIT] Loading asset mapping from cache...")
    asset_mapping = load_asset_mapping_from_cache()
    if not asset_mapping:
        log.error("[WS_INIT] No assets found in cache after enrichment!")
        with state_lock:
            pipeline.last_error = "No assets in cache"
        return False

    log.info(f"[WS_INIT] Loaded {len(asset_mapping)} assets from cache")

    # Subscribe to assets and fetch initial REST prices
    price_provider.load_asset_mapping(asset_mapping)

    # Fetch REST prices BEFORE starting WS (instant data)
    log.info("Fetching initial REST prices...")
    price_provider.fetch_initial_prices()

    # Now start WS for live updates
    price_provider.start()

    with state_lock:
        pipeline.ws_assets_subscribed = len(asset_mapping)

    # Wait for connection
    for _ in range(50):  # 5 second timeout
        if price_provider.connected:
            break
        time.sleep(0.1)

    if price_provider.connected:
        log.info("WebSocket connected")
        with state_lock:
            pipeline.status = Status.WS_CONNECTED
            pipeline.ws_connected = True
        notify_subscribers()
        return True
    else:
        log.error("WebSocket connection timeout")
        with state_lock:
            pipeline.last_error = "WebSocket connection timeout"
        return False


def recalculate_arbs_instant():
    """
    Instantly recalculate arbs using cached Stake odds + live WS Polymarket prices.
    Called on every Polymarket WebSocket price update.
    """
    if not cached_stake_games:
        return

    all_arbs = []
    all_matched_no_arb = []
    total_matched = 0

    cache_path = Path("match_cache.json")
    if not cache_path.exists():
        return

    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return

    for sport in config.ACTIVE_SPORTS:
        stake_games = cached_stake_games.get(sport, [])
        if not stake_games:
            continue

        stake_games_by_name = {_normalize_name(g.home_team, g.away_team): g for g in stake_games}
        sport_cache = cache.get(sport, {})
        matches = sport_cache.get("matches", [])

        for match in matches:
            poly_data = match.get("polymarket")
            if not poly_data:
                continue

            if _game_has_started(match.get("startDate")):
                continue

            game_id = poly_data["game_id"]
            teams_swapped = poly_data.get("teams_swapped", False)
            prices = price_provider.get_all_prices_for_game(game_id)
            poly_game = _build_poly_game_from_ws(game_id, sport, match, prices)
            if not poly_game:
                continue

            # Match with cached Stake game
            home_team = match.get("homeTeam", "")
            away_team = match.get("awayTeam", "")
            match_key = _normalize_name(home_team, away_team)
            stake_game = stake_games_by_name.get(match_key)

            if stake_game:
                total_matched += 1
                game_arbs = find_arbs(poly_game, stake_game, teams_swapped)
                profitable = [a for a in game_arbs if a.profit_pct >= config.MIN_PROFIT_PCT]
                all_arbs.extend(profitable)

                if not profitable:
                    all_matched_no_arb.append({
                        "sport": sport,
                        "game": f"{away_team} @ {home_team}",
                        "poly_url": poly_game.url,
                        "tp_url": stake_game.url,
                        "poly_markets": list(poly_game.markets.keys()),
                        "tp_markets": list(stake_game.markets.keys()),
                    })

    _update_arb_state(all_arbs, all_matched_no_arb, total_matched, is_ws_mode=True)


# Debounce state for price updates
_last_recalc_time = 0
_recalc_lock = threading.Lock()
_pending_recalc = False


def on_poly_price_update(asset_id: str, price):
    """
    Callback triggered by Polymarket WebSocket on every price update.
    Debounced to avoid recalculating on every single tick.
    """
    global _last_recalc_time, _pending_recalc

    if not cached_stake_games:
        return

    with _recalc_lock:
        now = time.time()
        # Debounce: max 1 recalc per 2 seconds
        if now - _last_recalc_time < 2:
            _pending_recalc = True
            return
        _last_recalc_time = now
        _pending_recalc = False

    try:
        recalculate_arbs_instant()
    except Exception as e:
        log.error(f"Instant recalc error: {e}")


def debounce_flush_worker():
    """Flush any pending recalculations every few seconds."""
    global _pending_recalc, _last_recalc_time

    while True:
        time.sleep(3)
        with _recalc_lock:
            if _pending_recalc and cached_stake_games:
                _pending_recalc = False
                _last_recalc_time = time.time()
                try:
                    recalculate_arbs_instant()
                except Exception as e:
                    log.error(f"Debounce flush error: {e}")


def build_stake_cache():
    """
    Build initial Stake game cache by fetching from odds-api.io.
    This replaces the Thunderpick Puppeteer scraping.
    """
    global cached_stake_games

    log.info("[STAKE] Building Stake game cache from odds-api.io...")

    # Fetch Stake odds for all active sports
    stake_raw = fetch_stake(config.ACTIVE_SPORTS)
    if not stake_raw:
        log.error("[STAKE] Failed to fetch Stake odds")
        return False

    # Parse into Game objects for each sport
    for sport in config.ACTIVE_SPORTS:
        adapter = get_adapter(sport)
        stake_games = adapter.parse_stake(stake_raw, sport)
        cached_stake_games[sport] = stake_games
        log.info(f"[STAKE] {sport}: {len(stake_games)} games cached")

    return True


def build_match_cache():
    """
    Build match_cache.json from Stake data.
    This creates the initial cache that will be enriched with Polymarket data.
    """
    cache_path = Path("match_cache.json")

    # Load existing cache or create new
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except:
            cache = {}
    else:
        cache = {}

    # Fetch Stake data
    stake_raw = fetch_stake(config.ACTIVE_SPORTS)
    if not stake_raw:
        log.error("Failed to fetch Stake data for cache building")
        return False

    # Build cache entries from Stake data
    for sport in config.ACTIVE_SPORTS:
        sport_data = stake_raw.get(sport, {})
        games = sport_data.get("games", [])

        matches = []
        for game in games:
            matches.append({
                "id": game.get("id", ""),
                "name": f"{game.get('awayTeam', '')} @ {game.get('homeTeam', '')}",
                "homeTeam": game.get("homeTeam", ""),
                "awayTeam": game.get("awayTeam", ""),
                "startDate": game.get("startDate", ""),
                "url": game.get("url", ""),
                "polymarket": None,  # Will be enriched later
            })

        cache[sport] = {
            "matches": matches,
            "lastUpdated": datetime.now().isoformat(),
        }

        log.info(f"[CACHE] {sport}: {len(matches)} matches")

    # Save cache
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    log.info(f"Saved match cache to {cache_path}")

    return True


def event_driven_worker():
    """
    Event-driven worker:
    - Fetches Stake every 60s to update cache
    - Poly prices update instantly via WebSocket callback
    """
    global cached_stake_games, last_stake_fetch_time, pipeline

    log.info("=" * 60)
    log.info("EVENT-DRIVEN WORKER STARTING")
    log.info("=" * 60)

    # Step 1: Build Stake match cache
    log.info("[STARTUP] Building Stake match cache...")
    if not build_match_cache():
        log.error("[STARTUP] Failed to build match cache")

    # Step 2: Initialize WebSocket mode (enrichment happens inside)
    log.info("[STARTUP] Initializing WebSocket mode...")
    if not init_websocket_mode():
        log.error("[STARTUP] WebSocket mode failed, falling back to REST")
        poll_worker()
        return

    log.info("[STARTUP] WebSocket mode initialized successfully!")

    # Step 3: Set up price update callback for instant arb detection
    price_provider._ws.set_on_price_update(on_poly_price_update)
    log.info("[STARTUP] Price update callback registered")

    # Start debounce flush thread
    threading.Thread(target=debounce_flush_worker, daemon=True).start()

    # Step 4: Initial Stake fetch + arb calculation
    log.info("[STARTUP] Fetching initial Stake odds...")
    if build_stake_cache():
        log.info("[STARTUP] Running initial arb calculation...")
        recalculate_arbs_instant()
        log.info(f"[STARTUP] Initial calculation complete - {len(current_arbs)} arbs, {pipeline.games_matched} matched")

    # Main loop: refresh Stake cache every 60s
    while True:
        try:
            log.info("=" * 50)
            log.info("Refreshing Stake cache...")

            with state_lock:
                pipeline.status = Status.FETCHING_STAKE
            notify_subscribers()

            # Fetch Stake for all sports
            stake_raw = fetch_stake(config.ACTIVE_SPORTS)
            if stake_raw:
                for sport in config.ACTIVE_SPORTS:
                    adapter = get_adapter(sport)
                    stake_games = adapter.parse_stake(stake_raw, sport)
                    cached_stake_games[sport] = stake_games
                    log.info(f"[{sport}] Stake cache updated: {len(stake_games)} games")
            else:
                log.warning("Stake fetch failed")

            last_stake_fetch_time = datetime.now()

            # Refresh stale Polymarket prices via REST
            stale_refreshed = price_provider.refresh_stale_prices(max_age_seconds=60.0)
            if stale_refreshed > 0:
                log.info(f"Refreshed {stale_refreshed} stale Polymarket prices via REST")

            with state_lock:
                pipeline.status = Status.WS_CONNECTED
                pipeline.last_stake_fetch = datetime.now()
                pipeline.stake_games_fetched = sum(len(g) for g in cached_stake_games.values())
                pipeline.ws_connected = price_provider.connected
            notify_subscribers()

            # Trigger immediate recalc with fresh Stake data
            recalculate_arbs_instant()

            log.info(f"Stake cache refresh complete. Next refresh in {config.POLL_INTERVAL}s")
            log.info("=" * 50)

        except Exception as e:
            log.error(f"Stake refresh error: {e}")
            log.error(traceback.format_exc())
            with state_lock:
                pipeline.status = Status.ERROR
                pipeline.last_error = str(e)
            notify_subscribers()

        time.sleep(config.POLL_INTERVAL)


def log_arb(arb: Arb):
    """Append arb to history log file."""
    log_path = Path(config.ARB_LOG_FILE)
    try:
        logs = json.loads(log_path.read_text()) if log_path.exists() else []
    except:
        logs = []
    logs.append(arb.to_log_dict())
    log_path.write_text(json.dumps(logs, indent=2))


def poll_worker():
    """Background thread that polls for data (REST mode fallback)."""
    log.info(f"Poll worker started (interval: {config.POLL_INTERVAL}s)")
    while True:
        try:
            refresh_data()
        except Exception as e:
            log.error(f"Refresh failed: {e}")
            log.error(traceback.format_exc())
            with state_lock:
                pipeline.status = Status.ERROR
                pipeline.last_error = str(e)
            notify_subscribers()
        time.sleep(config.POLL_INTERVAL)


# =============================================================================
# SSE / API
# =============================================================================

def notify_subscribers():
    """Send update to all SSE subscribers."""
    data = get_state_json()
    active_subs = len(subscribers)
    if active_subs > 0:
        log.debug(f"Notifying {active_subs} SSE subscribers")
    for q in subscribers[:]:
        try:
            q.append(data)
        except:
            subscribers.remove(q)


def get_state_json() -> str:
    """Get current state as JSON."""
    with state_lock:
        return json.dumps({
            "arbs": [a.to_log_dict() for a in current_arbs],
            "count": len(current_arbs),
            "matched_no_arb": matched_no_arb,
            "matched_count": len(matched_no_arb),
            "last_update": last_update.isoformat() if last_update else None,
            "pipeline": pipeline.to_dict(),
        })


@app.route('/events')
def events():
    """SSE endpoint for live updates."""
    def stream():
        q = []
        subscribers.append(q)
        log.info(f"SSE client connected. Total subscribers: {len(subscribers)}")
        last_heartbeat = time.time()
        try:
            yield f"data: {get_state_json()}\n\n"
            while True:
                if q:
                    yield f"data: {q.pop(0)}\n\n"
                    last_heartbeat = time.time()
                else:
                    # Send heartbeat every 15s to keep connection alive
                    if time.time() - last_heartbeat > 15:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.time()
                time.sleep(0.5)
        except GeneratorExit:
            log.info("SSE client disconnected (GeneratorExit)")
        finally:
            if q in subscribers:
                subscribers.remove(q)
            log.info(f"SSE client cleanup. Remaining subscribers: {len(subscribers)}")

    response = Response(stream(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Force refresh data."""
    log.info("Manual refresh triggered")
    threading.Thread(target=refresh_data).start()
    return jsonify({"status": "refreshing"})


@app.route('/health')
def health():
    """Simple healthcheck."""
    return "OK", 200


@app.route('/api/status')
def api_status():
    """Get current pipeline status."""
    with state_lock:
        return jsonify(pipeline.to_dict())


@app.route('/api/mode', methods=['GET'])
def api_get_mode():
    """Get current data mode."""
    with state_lock:
        return jsonify({
            "mode": pipeline.data_mode.value,
            "ws_connected": pipeline.ws_connected,
            "ws_assets": pipeline.ws_assets_subscribed,
        })


@app.route('/api/reconnect-ws', methods=['POST'])
def api_reconnect_ws():
    """Force reconnect WebSocket."""
    def do_reconnect():
        price_provider.stop()
        time.sleep(1)
        init_websocket_mode()

    threading.Thread(target=do_reconnect).start()
    return jsonify({"status": "reconnecting websocket"})


@app.route('/api/enrich-cache', methods=['POST'])
def api_enrich_cache():
    """Manually trigger cache enrichment."""
    def do_enrich():
        log.info("Enriching cache with Polymarket assets...")
        try:
            run_enrich_cache()
            log.info("Cache enrichment complete")

            with state_lock:
                if pipeline.data_mode == DataMode.WEBSOCKET:
                    mapping = load_asset_mapping_from_cache()
                    price_provider.load_asset_mapping(mapping)
                    pipeline.ws_assets_subscribed = len(mapping)
            notify_subscribers()
        except Exception as e:
            log.error(f"Cache enrichment failed: {e}")

    threading.Thread(target=do_enrich).start()
    return jsonify({"status": "enriching cache"})


@app.route('/api/debug/<sport>')
def api_debug(sport):
    """Get raw debug data for a sport."""
    sport = sport.upper()
    if sport not in config.SPORTS:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400

    sport_config = config.SPORTS[sport]
    adapter = get_adapter(sport)

    # Fetch raw data
    poly_raw = fetch_polymarket(sport_config["polymarket_tag_id"])
    stake_raw = fetch_stake([sport])

    # Parse into games
    poly_games = adapter.parse_polymarket(poly_raw) if poly_raw else []
    stake_games = adapter.parse_stake(stake_raw, sport) if stake_raw else []

    # Match games
    matched = match_games(poly_games, stake_games, adapter)

    # Build debug output
    debug_data = {
        "sport": sport,
        "polymarket_parsed": [
            {
                "id": g.id,
                "away": g.away_team,
                "home": g.home_team,
                "markets": {
                    k: [{"line": m.line, "outcomes": m.outcomes} for m in v]
                    for k, v in g.markets.items()
                }
            }
            for g in poly_games
        ],
        "stake_parsed": [
            {
                "id": g.id,
                "away": g.away_team,
                "home": g.home_team,
                "markets": {
                    k: [{"line": m.line, "outcomes": m.outcomes} for m in v]
                    for k, v in g.markets.items()
                }
            }
            for g in stake_games
        ],
        "matched_games": [
            {
                "poly_id": pg.id,
                "stake_id": sg.id,
                "display": f"{pg.away_team} @ {pg.home_team}",
            }
            for pg, sg, _ in matched
        ]
    }

    return jsonify(debug_data)


@app.route('/')
def index():
    return render_template('index.html', poll_interval=config.POLL_INTERVAL)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Stake/Polymarket Arb Monitor')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port to run on')
    parser.add_argument('--rest', action='store_true', help='Force REST-only mode (no WebSocket)')
    args = parser.parse_args()

    log.info("Stake/Polymarket Arb Monitor starting...")
    log.info(f"Active sports: {config.ACTIVE_SPORTS}")

    if args.rest:
        log.info("Starting in REST-only mode (forced)...")
        log.info(f"Poll interval: {config.POLL_INTERVAL}s")
        threading.Thread(target=poll_worker, daemon=True).start()
    else:
        # Default: Event-driven mode (instant Poly updates, Stake cached 60s)
        log.info("Starting in EVENT-DRIVEN mode...")
        log.info(f"  - Polymarket: Instant updates via WebSocket")
        log.info(f"  - Stake: Cached, refresh every {config.POLL_INTERVAL}s")
        threading.Thread(target=event_driven_worker, daemon=True).start()

    app.run(debug=False, host='0.0.0.0', port=args.port, threaded=True)
