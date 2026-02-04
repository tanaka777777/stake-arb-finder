"""
Configuration for Bovada/Polymarket Arb Monitor.
"""

import os

# Polling interval in seconds
POLL_INTERVAL = 60

# WebSocket mode poll interval (faster since Poly prices are already cached)
WS_POLL_INTERVAL = 10

# Minimum profit % to display (filter noise)
MIN_PROFIT_PCT = 0.0

# Log arbs above this threshold for analysis
LOG_THRESHOLD_PCT = 5.0

# Arb log file
ARB_LOG_FILE = "arb_history.json"

# Sports configurations
# bovada_sport and bovada_league match odds-api.io naming
SPORTS = {
    "NBA": {
        "polymarket_tag_id": 745,
        "bovada_sport": "basketball",
        "bovada_league": "usa-nba",
        "market_types": ["moneyline", "spreads", "totals"],
    },
    "NHL": {
        "polymarket_tag_id": 899,
        "bovada_sport": "ice-hockey",
        "bovada_league": "usa-nhl",
        "market_types": ["moneyline", "totals"],  # No spreads for NHL
    },
    "NFL": {
        "polymarket_tag_id": 450,
        "bovada_sport": "american-football",
        "bovada_league": "usa-nfl",
        "market_types": ["moneyline", "spreads", "totals"],
    },
}

# Active sports to monitor
ACTIVE_SPORTS = ["NBA", "NHL", "NFL"]

# Discord Bot Configuration
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")
DISCORD_MIN_PROFIT_PCT = 1.0  # Minimum profit % to send Discord notifications
