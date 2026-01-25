#!/bin/bash
# Start Multi-Task Model Live Trading

cd "$(dirname "$0")/.."
source glitch_env/bin/activate
cd live_testing
python live_trade_multi.py
