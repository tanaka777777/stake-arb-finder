"""
Configuration for Stake/Polymarket Arb Monitor.
"""

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
# stake_sport and stake_league match odds-api.io naming
SPORTS = {
    "NBA": {
        "polymarket_tag_id": 745,
        "stake_sport": "basketball",
        "stake_league": "usa-nba",
        "market_types": ["moneyline", "spreads", "totals"],
    },
    "NHL": {
        "polymarket_tag_id": 899,
        "stake_sport": "ice-hockey",
        "stake_league": "usa-nhl",
        "market_types": ["moneyline", "totals"],  # No spreads for NHL
    },
    "NFL": {
        "polymarket_tag_id": 450,
        "stake_sport": "american-football",
        "stake_league": "usa-nfl",
        "market_types": ["moneyline", "spreads", "totals"],
    },
}

# Active sports to monitor
ACTIVE_SPORTS = ["NBA", "NHL", "NFL"]
