"""
Polymarket WebSocket client for real-time price updates.

Subscribes to market channel for live best_bid/best_ask updates.
No authentication required for public market data.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import requests
import websocket

log = logging.getLogger("arb-monitor")

WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB_API = "https://clob.polymarket.com"
PING_INTERVAL = 10  # seconds


@dataclass
class AssetPrice:
    """Current price state for an asset."""
    asset_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_size: float | None = None  # Size in shares at best bid
    best_ask_size: float | None = None  # Size in shares at best ask
    last_trade: float | None = None
    updated_at: float = 0


@dataclass
class PolymarketWS:
    """
    WebSocket client for Polymarket real-time prices.

    Usage:
        ws = PolymarketWS()
        ws.subscribe(["asset_id_1", "asset_id_2"])
        ws.start()

        # Get current prices
        price = ws.get_price("asset_id_1")

        # Cleanup
        ws.stop()
    """

    _ws: websocket.WebSocketApp | None = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _ping_thread: threading.Thread | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    _connected: bool = field(default=False, repr=False)
    _prices: dict[str, AssetPrice] = field(default_factory=dict, repr=False)
    _subscribed_assets: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _on_price_update: Callable[[str, AssetPrice], None] | None = field(default=None, repr=False)
    _reconnect_delay: float = field(default=1.0, repr=False)
    _max_reconnect_delay: float = field(default=30.0, repr=False)
    # Stats for monitoring
    update_count: int = field(default=0, repr=False)
    last_update_time: float = field(default=0, repr=False)
    message_count: int = field(default=0, repr=False)

    def subscribe(self, asset_ids: list[str]) -> None:
        """Add assets to subscription list. Call before start() or to add more assets."""
        with self._lock:
            new_assets = set(asset_ids) - self._subscribed_assets
            self._subscribed_assets.update(asset_ids)

            # Initialize price entries
            for asset_id in asset_ids:
                if asset_id not in self._prices:
                    self._prices[asset_id] = AssetPrice(asset_id=asset_id)

        # If already connected, send subscription for new assets
        if self._connected and new_assets and self._ws:
            self._send_subscription(list(new_assets))

    def unsubscribe(self, asset_ids: list[str]) -> None:
        """Remove assets from subscription."""
        with self._lock:
            for asset_id in asset_ids:
                self._subscribed_assets.discard(asset_id)
                self._prices.pop(asset_id, None)

    def get_price(self, asset_id: str) -> AssetPrice | None:
        """Get current price for an asset."""
        with self._lock:
            return self._prices.get(asset_id)

    def get_all_prices(self) -> dict[str, AssetPrice]:
        """Get all current prices."""
        with self._lock:
            return dict(self._prices)

    def set_on_price_update(self, callback: Callable[[str, AssetPrice], None]) -> None:
        """Set callback for price updates."""
        self._on_price_update = callback

    def fetch_initial_prices(self) -> int:
        """
        Fetch current prices via REST API for all subscribed assets.
        Call after subscribe() to pre-populate prices before WS updates arrive.

        Returns: number of assets successfully fetched
        """
        with self._lock:
            asset_ids = list(self._subscribed_assets)
        return self._fetch_prices_for_assets(asset_ids, "initialization")

    def refresh_stale_prices(self, max_age_seconds: float = 60.0) -> int:
        """
        Refresh prices older than max_age_seconds via REST API.
        Call periodically to keep prices fresh for less liquid markets.

        Returns: number of assets refreshed
        """
        now = time.time()
        stale_assets = []

        with self._lock:
            for asset_id, price in self._prices.items():
                age = now - price.updated_at if price.updated_at else float('inf')
                if age > max_age_seconds:
                    stale_assets.append(asset_id)

        if not stale_assets:
            return 0

        log.info(f"Refreshing {len(stale_assets)} stale prices (>{max_age_seconds}s old)...")
        return self._fetch_prices_for_assets(stale_assets, "refresh")

    def _fetch_prices_for_assets(self, asset_ids: list[str], context: str) -> int:
        """Fetch REST prices for a list of assets."""
        fetched = 0

        for asset_id in asset_ids:
            try:
                r = requests.get(
                    f"{CLOB_API}/book",
                    params={"token_id": asset_id},
                    timeout=5
                )
                if r.status_code != 200:
                    continue

                data = r.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                with self._lock:
                    if asset_id not in self._prices:
                        self._prices[asset_id] = AssetPrice(asset_id=asset_id)

                    price = self._prices[asset_id]

                    # Get best (highest) bid and best (lowest) ask with sizes
                    if bids:
                        best_bid_entry = max(bids, key=lambda b: float(b["price"]))
                        price.best_bid = float(best_bid_entry["price"])
                        price.best_bid_size = float(best_bid_entry.get("size", 0))
                    if asks:
                        best_ask_entry = min(asks, key=lambda a: float(a["price"]))
                        price.best_ask = float(best_ask_entry["price"])
                        price.best_ask_size = float(best_ask_entry.get("size", 0))

                    if bids or asks:
                        price.updated_at = time.time()
                        fetched += 1

            except Exception as e:
                log.debug(f"REST fetch failed for {asset_id[:20]}...: {e}")
                continue

        log.info(f"REST {context}: {fetched}/{len(asset_ids)} assets fetched")
        return fetched

    def start(self) -> None:
        """Start WebSocket connection in background thread."""
        if self._running:
            log.warning("PolymarketWS already running")
            return

        self._running = True
        self._reconnect_delay = 1.0
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        log.info("PolymarketWS started")

    def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            self._ws.close()
        log.info("PolymarketWS stopped")

    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    def _run_forever(self) -> None:
        """Main loop - connects and reconnects as needed."""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                log.error(f"PolymarketWS connection error: {e}")

            if self._running:
                log.info(f"PolymarketWS reconnecting in {self._reconnect_delay:.1f}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    def _connect(self) -> None:
        """Establish WebSocket connection."""
        self._ws = websocket.WebSocketApp(
            WSS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever()

    def _on_open(self, ws) -> None:
        """Handle connection open."""
        log.info("PolymarketWS connected")
        self._connected = True
        self._reconnect_delay = 1.0

        # Send subscription for all assets
        with self._lock:
            if self._subscribed_assets:
                self._send_subscription(list(self._subscribed_assets))

        # Start ping thread
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def _on_message(self, ws, message: str) -> None:
        """Handle incoming message."""
        if message in ("PONG", "INVALID OPERATION"):
            return

        try:
            data = json.loads(message)
            self._handle_message(data)
        except json.JSONDecodeError:
            if len(message) < 50:  # Short non-JSON messages are likely status msgs
                return
            log.warning(f"PolymarketWS invalid JSON: {message[:100]}")

    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        log.error(f"PolymarketWS error: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle connection close."""
        self._connected = False
        log.info(f"PolymarketWS closed: {close_status_code} {close_msg}")

    def _send_subscription(self, asset_ids: list[str]) -> None:
        """Send subscription message for assets."""
        if not self._ws or not self._connected:
            return

        # Polymarket accepts subscriptions in batches
        batch_size = 100
        for i in range(0, len(asset_ids), batch_size):
            batch = asset_ids[i:i + batch_size]
            msg = json.dumps({"assets_ids": batch, "type": "market"})
            try:
                self._ws.send(msg)
                log.debug(f"PolymarketWS subscribed to {len(batch)} assets")
            except Exception as e:
                log.error(f"PolymarketWS subscription error: {e}")

    def _ping_loop(self) -> None:
        """Send PING every PING_INTERVAL seconds."""
        while self._running and self._connected:
            try:
                if self._ws:
                    self._ws.send("PING")
            except Exception:
                break
            time.sleep(PING_INTERVAL)

    def _handle_message(self, data: list | dict) -> None:
        """Process incoming market data."""
        self.message_count += 1

        # Messages come as arrays
        if isinstance(data, dict):
            data = [data]

        # Collect updates to fire callbacks AFTER releasing lock (avoid deadlock)
        callbacks_to_fire: list[tuple[str, AssetPrice]] = []

        for msg in data:
            event_type = msg.get("event_type")

            # Handle new price_change schema with price_changes array
            if event_type == "price_change" and "price_changes" in msg:
                for change in msg["price_changes"]:
                    asset_id = change.get("asset_id")
                    if not asset_id:
                        continue

                    with self._lock:
                        if asset_id not in self._prices:
                            continue

                        price = self._prices[asset_id]
                        updated = False

                        if "best_bid" in change:
                            bid_val = change["best_bid"]
                            if bid_val and bid_val != "0":
                                price.best_bid = float(bid_val)
                                updated = True
                        if "best_ask" in change:
                            ask_val = change["best_ask"]
                            if ask_val and ask_val != "0":
                                price.best_ask = float(ask_val)
                                updated = True

                        if updated:
                            price.updated_at = time.time()
                            self.update_count += 1
                            self.last_update_time = price.updated_at
                            if self._on_price_update:
                                callbacks_to_fire.append((asset_id, price))
                continue  # Done with this message

            # Legacy handling for other message types
            asset_id = msg.get("asset_id")

            if not asset_id:
                continue

            with self._lock:
                if asset_id not in self._prices:
                    continue

                price = self._prices[asset_id]
                updated = False

                if event_type == "price_change":
                    # Legacy price_change format (asset_id at root)
                    if "best_bid" in msg:
                        price.best_bid = float(msg["best_bid"])
                        updated = True
                    if "best_ask" in msg:
                        price.best_ask = float(msg["best_ask"])
                        updated = True

                elif event_type == "book":
                    # Full order book - extract best bid/ask with sizes
                    bids = msg.get("bids", [])
                    asks = msg.get("asks", [])
                    if bids:
                        # Best bid is HIGHEST price (don't assume sorted)
                        best_bid_entry = max(bids, key=lambda b: float(b.get("price", 0)))
                        price.best_bid = float(best_bid_entry.get("price", 0))
                        price.best_bid_size = float(best_bid_entry.get("size", 0))
                        updated = True
                    if asks:
                        # Best ask is LOWEST price (don't assume sorted)
                        best_ask_entry = min(asks, key=lambda a: float(a.get("price", 0)))
                        price.best_ask = float(best_ask_entry.get("price", 0))
                        price.best_ask_size = float(best_ask_entry.get("size", 0))
                        updated = True

                elif event_type == "last_trade_price":
                    price.last_trade = float(msg.get("price", 0))
                    updated = True

                if updated:
                    price.updated_at = time.time()
                    self.update_count += 1
                    self.last_update_time = price.updated_at
                    if self._on_price_update:
                        # Queue callback, don't call inside lock
                        callbacks_to_fire.append((asset_id, price))

        # Fire callbacks OUTSIDE the lock to prevent deadlock
        for asset_id, price in callbacks_to_fire:
            try:
                self._on_price_update(asset_id, price)
            except Exception as e:
                log.error(f"Price update callback error: {e}")


class PolymarketPriceProvider:
    """
    High-level interface for Polymarket prices.
    Manages WebSocket lifecycle and provides clean price access.
    """

    def __init__(self):
        self._ws = PolymarketWS()
        self._asset_to_market: dict[str, dict] = {}  # asset_id -> {game_id, market_type, line, outcome}
        self._lock = threading.Lock()

    def load_asset_mapping(self, mapping: dict[str, dict]) -> None:
        """
        Load mapping from asset_ids to market info.

        mapping format: {
            "asset_id": {
                "game_id": "...",
                "market_type": "moneyline|spread|total",
                "line": float | None,
                "outcome": "yes|no",
                "side": "home|away" | None  # for spreads
            }
        }
        """
        with self._lock:
            self._asset_to_market = mapping

        self._ws.subscribe(list(mapping.keys()))

    def start(self) -> None:
        """Start price feed."""
        self._ws.start()

    def stop(self) -> None:
        """Stop price feed."""
        self._ws.stop()

    def fetch_initial_prices(self) -> int:
        """
        Fetch current prices via REST API.
        Call after load_asset_mapping() and before or after start().
        Returns number of assets successfully fetched.
        """
        return self._ws.fetch_initial_prices()

    def refresh_stale_prices(self, max_age_seconds: float = 60.0) -> int:
        """
        Refresh prices older than max_age_seconds via REST API.
        Call periodically to keep prices fresh for less liquid markets.
        Returns number of assets refreshed.
        """
        return self._ws.refresh_stale_prices(max_age_seconds)

    @property
    def connected(self) -> bool:
        return self._ws.connected

    def get_all_prices_for_game(self, game_id: str) -> dict:
        """
        Get all market prices for a game.

        Returns: {
            "moneyline": {"yes": {...}, "no": {...}},
            "spreads": [{"line": float, "yes": {...}, "no": {...}, "side": str}, ...],
            "totals": [{"line": float, "yes": {...}, "no": {...}}, ...],
        }
        """
        result = {
            "moneyline": None, "spreads": [], "totals": [],
        }

        # Collect all markets for this game
        markets_by_type: dict[str, dict[float | None, dict]] = {
            "moneyline": {},
            "spread": {},
            "total": {},
        }

        with self._lock:
            for asset_id, info in self._asset_to_market.items():
                if info["game_id"] != game_id:
                    continue

                price = self._ws.get_price(asset_id)
                if not price or price.best_bid is None:
                    continue

                mtype = info["market_type"]
                line = info.get("line")
                outcome = info["outcome"]
                side = info.get("side")

                if mtype not in markets_by_type:
                    continue

                key = line
                if key not in markets_by_type[mtype]:
                    markets_by_type[mtype][key] = {"line": line, "side": side}

                if outcome == "yes":
                    markets_by_type[mtype][key]["yes"] = {
                        "bid": price.best_bid,
                        "ask": price.best_ask,
                        "bid_size": price.best_bid_size,  # Shares at bid
                        "ask_size": price.best_ask_size   # Shares at ask
                    }
                    if side:
                        markets_by_type[mtype][key]["side"] = side

        # Build result
        for mtype, lines in markets_by_type.items():
            for line, data in lines.items():
                if "yes" not in data:
                    continue

                # Calculate NO from YES (NO ask = 1 - YES bid, etc.)
                yes = data["yes"]
                data["no"] = {
                    "bid": round(1 - yes["ask"], 4) if yes["ask"] else None,
                    "ask": round(1 - yes["bid"], 4) if yes["bid"] else None,
                    "bid_size": yes.get("ask_size"),  # NO bid size = YES ask size
                    "ask_size": yes.get("bid_size")   # NO ask size = YES bid size
                }

                if mtype == "moneyline":
                    result["moneyline"] = data
                elif mtype == "spread":
                    result["spreads"].append(data)
                elif mtype == "total":
                    result["totals"].append(data)

        # Sort spreads and totals by line
        result["spreads"].sort(key=lambda x: x.get("line") or 0)
        result["totals"].sort(key=lambda x: x.get("line") or 0)

        return result


# Module-level singleton for easy access
_provider: PolymarketPriceProvider | None = None


def get_price_provider() -> PolymarketPriceProvider:
    """Get the global price provider instance."""
    global _provider
    if _provider is None:
        _provider = PolymarketPriceProvider()
    return _provider
