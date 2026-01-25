# Live Testing - Paper Trading

Live trading scripts die de MySQL database monitoren en trades uitvoeren met nep geld.

## Setup

### 1. Install Dependencies
```bash
cd /Users/tymo/Documents/GlitchCatcherv42
source glitch_env/bin/activate
pip install mysql-connector-python
```

### 2. Database Configuratie
De scripts verbinden automatisch met:
- Host: localhost
- Port: 3306
- Database: crypto
- User: dbuser
- Password: dbuser_01

## Gebruik

### Single-Task Model
```bash
cd live_testing
python live_trade_single.py
```

### Multi-Task Model
```bash
cd live_testing
python live_trade_multi.py
```

### Beide Modellen Tegelijk
Open twee terminals:

**Terminal 1 (Single-Task):**
```bash
cd live_testing
python live_trade_single.py
```

**Terminal 2 (Multi-Task):**
```bash
cd live_testing
python live_trade_multi.py
```

## Output

- **Groen [BUY/SELL]**: Profitable trade
- **Rood [SELL]**: Loss trade
- **Statistics**: Elke 10 trades worden statistieken getoond

## Trading Parameters

**Single-Task:**
- Entry Threshold: 85% probability
- Profit Target: 1.0%
- Stop Loss: 0.5%

**Multi-Task:**
- Detection Threshold: 70%
- Persistence Threshold: 50%
- Profit Target: 1.0%
- Stop Loss: 0.5%

## Paper Money

Beide scripts starten met $10,000 nep geld.
Elke trade gebruikt 10% van de huidige balance.

## Monitoring

Scripts monitoren de `assets_history` tabel en verwerken nieuwe data elke 5 seconden.
