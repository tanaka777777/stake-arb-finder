"""
Discord bot for arbitrage alerts.

Sends notifications when arbs >1% are detected.
Users can reply with stake amount to get calculated bet sizes.
"""

import asyncio
import logging
import os
import re
import threading
from datetime import datetime
from typing import Optional

import discord
from discord import Embed, ButtonStyle
from discord.ui import View, Button

from models import Arb

log = logging.getLogger("discord-bot")


class ArbAlertView(View):
    """View with buttons for Polymarket and Bovada links."""

    def __init__(self, poly_url: str, bovada_url: str):
        super().__init__(timeout=None)
        if poly_url:
            self.add_item(Button(
                label="Polymarket",
                style=ButtonStyle.link,
                url=poly_url,
                emoji="\U0001F7E3"  # purple circle
            ))
        if bovada_url:
            self.add_item(Button(
                label="Bovada",
                style=ButtonStyle.link,
                url=bovada_url,
                emoji="\U0001F7E0"  # orange circle
            ))


class DiscordNotifier:
    """
    Discord bot for sending arbitrage alerts and handling stake calculations.
    """

    def __init__(self, token: str, channel_id: str, min_profit_pct: float = 1.0):
        self.token = token
        self.channel_id = int(channel_id) if channel_id else 0
        self.min_profit_pct = min_profit_pct
        self.client: Optional[discord.Client] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Track recent arbs by message ID for reply handling
        self.arb_by_message_id: dict[int, Arb] = {}
        # Track arbs by ID to avoid duplicate notifications
        self.notified_arb_ids: set[str] = set()

    def start(self):
        """Start the Discord bot in a background thread."""
        if not self.token or not self.channel_id:
            log.warning("Discord bot not configured (missing token or channel ID)")
            return

        self._thread = threading.Thread(target=self._run_bot, daemon=True)
        self._thread.start()

        # Wait for bot to be ready (max 10 seconds)
        if self._ready.wait(timeout=10):
            log.info("Discord bot started and ready")
        else:
            log.warning("Discord bot startup timed out")

    def _run_bot(self):
        """Run the Discord bot (called in background thread)."""
        intents = discord.Intents.default()
        intents.message_content = True

        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            log.info(f"Discord bot logged in as {self.client.user}")
            self._ready.set()

        @self.client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self.client.user:
                return

            # Check if this is a reply to one of our arb alerts
            if message.reference and message.reference.message_id:
                ref_id = message.reference.message_id
                if ref_id in self.arb_by_message_id:
                    await self._handle_stake_reply(message, self.arb_by_message_id[ref_id])

        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.client.start(self.token))
        except Exception as e:
            log.error(f"Discord bot error: {e}")

    async def _handle_stake_reply(self, message: discord.Message, arb: Arb):
        """Handle a reply with stake amount."""
        # Parse amount from message (e.g., "$500", "500", "100.50")
        match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', message.content)
        if not match:
            await message.reply(
                "Please specify an amount (e.g., `$500` or `500`)",
                mention_author=False
            )
            return

        amount_str = match.group(1).replace(",", "")
        try:
            total_stake = float(amount_str)
        except ValueError:
            await message.reply(
                "Invalid amount. Please use a number (e.g., `$500`)",
                mention_author=False
            )
            return

        if total_stake <= 0:
            await message.reply(
                "Amount must be positive",
                mention_author=False
            )
            return

        # Calculate stakes using the arb's stake percentages
        poly_stake = total_stake * (arb.poly_stake_pct / 100)
        bovada_stake = total_stake * (arb.tp_stake_pct / 100)

        # Calculate guaranteed return
        # If we bet poly_stake at poly_odds, payout = poly_stake * poly_odds
        guaranteed_return = poly_stake * arb.poly_odds
        profit = guaranteed_return - total_stake
        profit_pct = (profit / total_stake) * 100

        # Build response embed
        embed = Embed(
            title=f"\U0001F4CA Stake Calculation for ${total_stake:,.2f}",
            color=0x00FF00
        )

        embed.add_field(
            name=f"\U0001F7E3 Polymarket ({arb.poly_side.upper()})",
            value=f"**${poly_stake:,.2f}** @ {arb.poly_odds:.3f}",
            inline=True
        )
        embed.add_field(
            name=f"\U0001F7E0 Bovada ({arb.tp_side.upper()})",
            value=f"**${bovada_stake:,.2f}** @ {arb.tp_odds:.3f}",
            inline=True
        )
        embed.add_field(
            name="\U0001F4B0 Guaranteed Return",
            value=f"**${guaranteed_return:,.2f}** (+{profit_pct:.2f}%)",
            inline=False
        )

        await message.reply(embed=embed, mention_author=False)

    def send_arb_alert(self, arb: Arb) -> bool:
        """
        Send an arb alert to the Discord channel.

        Args:
            arb: The arbitrage opportunity to alert on

        Returns:
            True if alert was sent, False otherwise
        """
        if not self._ready.is_set() or not self.client or not self.loop:
            log.debug("Discord bot not ready, skipping alert")
            return False

        if arb.profit_pct < self.min_profit_pct:
            log.debug(f"Arb profit {arb.profit_pct:.2f}% below threshold {self.min_profit_pct}%")
            return False

        # Check if already notified
        if arb.id in self.notified_arb_ids:
            log.debug(f"Already notified for arb {arb.id}")
            return False

        # Mark as notified
        self.notified_arb_ids.add(arb.id)

        # Schedule the async send
        future = asyncio.run_coroutine_threadsafe(
            self._send_alert_async(arb),
            self.loop
        )

        try:
            # Wait up to 5 seconds for the message to send
            future.result(timeout=5)
            return True
        except Exception as e:
            log.error(f"Failed to send Discord alert: {e}")
            return False

    async def _send_alert_async(self, arb: Arb):
        """Send the alert message asynchronously."""
        channel = self.client.get_channel(self.channel_id)
        if not channel:
            log.error(f"Could not find Discord channel {self.channel_id}")
            return

        # Build embed
        embed = Embed(
            title=f"\U0001F6A8 ARB ALERT: +{arb.profit_pct:.2f}%",
            color=0xFF6600 if arb.profit_pct >= 2 else 0xFFAA00,
            timestamp=datetime.now()
        )

        # Game info
        embed.add_field(
            name=f"\U0001F3C0 {arb.game_display} ({arb.sport})",
            value=f"\U0001F4CA {arb.market_type.title()}" + (f" ({arb.line})" if arb.line else ""),
            inline=False
        )

        # Polymarket side
        poly_desc = f"**{arb.poly_side.upper()}** @ {arb.poly_odds:.3f}"
        if arb.poly_size:
            poly_desc += f"\nSize: ${arb.poly_size:,.0f}"
        embed.add_field(
            name="\U0001F7E3 Polymarket",
            value=poly_desc,
            inline=True
        )

        # Bovada side
        bovada_desc = f"**{arb.tp_side.upper()}** @ {arb.tp_odds:.3f}"
        if arb.tp_line is not None and arb.tp_line != arb.line:
            bovada_desc += f"\nLine: {arb.tp_line}"
        embed.add_field(
            name="\U0001F7E0 Bovada",
            value=bovada_desc,
            inline=True
        )

        # Instructions
        embed.add_field(
            name="\U0001F4AC How to use",
            value="Reply with amount (e.g. `$500`) for stake calculation",
            inline=False
        )

        # Create view with link buttons
        view = ArbAlertView(arb.poly_url, arb.tp_url)

        # Send message
        msg = await channel.send(embed=embed, view=view)

        # Store for reply handling
        self.arb_by_message_id[msg.id] = arb

        # Cleanup old messages (keep last 100)
        if len(self.arb_by_message_id) > 100:
            oldest_ids = sorted(self.arb_by_message_id.keys())[:-100]
            for old_id in oldest_ids:
                del self.arb_by_message_id[old_id]

        log.info(f"Discord alert sent for {arb.game_display} (+{arb.profit_pct:.2f}%)")

    def stop(self):
        """Stop the Discord bot."""
        if self.client and self.loop:
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)
            log.info("Discord bot stopped")


# Singleton instance
_notifier: Optional[DiscordNotifier] = None


def get_discord_notifier() -> Optional[DiscordNotifier]:
    """Get the Discord notifier singleton (creates if needed)."""
    global _notifier

    if _notifier is None:
        # Import config here to avoid circular imports
        import config

        token = getattr(config, 'DISCORD_BOT_TOKEN', '') or os.environ.get('DISCORD_BOT_TOKEN', '')
        channel_id = getattr(config, 'DISCORD_CHANNEL_ID', '') or os.environ.get('DISCORD_CHANNEL_ID', '')
        min_profit = getattr(config, 'DISCORD_MIN_PROFIT_PCT', 1.0)

        if token and channel_id:
            _notifier = DiscordNotifier(token, channel_id, min_profit)
        else:
            log.info("Discord bot not configured (set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID)")

    return _notifier


def start_discord_bot():
    """Start the Discord bot if configured."""
    notifier = get_discord_notifier()
    if notifier:
        notifier.start()


def send_arb_alert(arb: Arb) -> bool:
    """Send an arb alert if Discord is configured."""
    notifier = get_discord_notifier()
    if notifier:
        return notifier.send_arb_alert(arb)
    return False
