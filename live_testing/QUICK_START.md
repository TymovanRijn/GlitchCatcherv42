# Live Testing Quick Start

## Op Raspberry Pi

### 1. Setup
```bash
cd /path/to/GlitchCatcherv42
source glitch_env/bin/activate
pip install mysql-connector-python
```

### 2. Start Beide Modellen

**Terminal 1 - Single-Task:**
```bash
cd live_testing
python live_trade_single.py
```

**Terminal 2 - Multi-Task:**
```bash
cd live_testing
python live_trade_multi.py
```

## Output

- **Groen [BUY/SELL]**: Profitable trade
- **Rood [SELL]**: Loss trade
- **Statistics**: Elke 5 trades

## Database Config

- Host: localhost:3306
- Database: crypto
- User: dbuser
- Password: dbuser_01
- Tables: assets_history (monitored)

## Paper Money

- Start: $10,000
- Per trade: 10% of balance
- Real-time PnL tracking

## Stop

Press Ctrl+C in beide terminals om te stoppen.
