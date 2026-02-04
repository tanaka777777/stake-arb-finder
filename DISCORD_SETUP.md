# Discord Bot Setup for Arbitrage Alerts

This guide explains how to set up the Discord bot for receiving arbitrage alerts.

## Features

- Automatic notifications when arbs >1% are detected
- Includes both Polymarket and Bovada links
- Reply with stake amount (e.g., "$500") to get calculated bet sizes
- Shows guaranteed return and profit percentage

## Setup Instructions

### 1. Create a Discord Application

1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Name it "Arb Alert Bot" (or any name you prefer)
4. Click "Create"

### 2. Create the Bot

1. In your application, go to the "Bot" tab (left sidebar)
2. Click "Add Bot" then "Yes, do it!"
3. Under "Privileged Gateway Intents", enable:
   - **Message Content Intent** (required for reading reply messages)
4. Click "Reset Token" to generate a new token
5. Copy and save the token securely (you won't be able to see it again!)

### 3. Generate Invite URL

1. Go to "OAuth2" > "URL Generator" (left sidebar)
2. Under "Scopes", select:
   - `bot`
3. Under "Bot Permissions", select:
   - Send Messages
   - Read Message History
   - Embed Links
4. Copy the generated URL at the bottom
5. Open the URL in your browser and add the bot to your server

### 4. Get Your Channel ID

1. In Discord, go to User Settings > Advanced
2. Enable "Developer Mode"
3. Right-click the channel where you want alerts
4. Click "Copy ID"

### 5. Set Environment Variables

Set these environment variables before running the app:

**Windows (PowerShell):**
```powershell
$env:DISCORD_BOT_TOKEN = "your_bot_token_here"
$env:DISCORD_CHANNEL_ID = "your_channel_id_here"
```

**Windows (Command Prompt):**
```cmd
set DISCORD_BOT_TOKEN=your_bot_token_here
set DISCORD_CHANNEL_ID=your_channel_id_here
```

**Linux/macOS:**
```bash
export DISCORD_BOT_TOKEN="your_bot_token_here"
export DISCORD_CHANNEL_ID="your_channel_id_here"
```

Or create a `.env` file (requires python-dotenv):
```
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_CHANNEL_ID=your_channel_id_here
```

### 6. Run the App

```bash
pip install -r requirements.txt
python app.py
```

The bot should show as online in your Discord server.

## Usage

### Receiving Alerts

When an arb >1% is detected, you'll receive a message like:

```
ARB ALERT: +2.45%
Lakers @ Celtics (NBA)
Moneyline

Polymarket: YES @ 2.100
Bovada: HOME @ 1.950

Reply with amount (e.g. "$500") for stake calculation
[Polymarket] [Bovada]
```

### Calculating Bovadas

Reply to any alert message with a dollar amount:

```
$500
```

The bot will respond with:

```
Bovada Calculation for $500.00:
Polymarket (YES): $243.90 @ 2.100
Bovada (HOME): $256.10 @ 1.950
Guaranteed Return: $512.19 (+2.45%)
```

## Configuration

You can adjust the minimum profit threshold in `config.py`:

```python
DISCORD_MIN_PROFIT_PCT = 1.0  # Only notify for arbs >= 1%
```

## Troubleshooting

### Bot not coming online
- Verify the token is correct
- Check that "Message Content Intent" is enabled in the Discord Developer Portal
- Look for error messages in the console

### Not receiving alerts
- Verify the channel ID is correct
- Make sure the bot has permissions to send messages in that channel
- Check that arbs are above the minimum threshold (default 1%)

### Reply not working
- Make sure you're replying to the alert message (not just sending a new message)
- Use a valid amount format: `$500`, `500`, `$1,000.00`
