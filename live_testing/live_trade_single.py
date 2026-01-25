#!/usr/bin/env python3
"""
Live Trading - Single-Task Model
Monitors database and executes trades with paper money
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from database import CryptoDatabase

# ANSI colors for terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def create_features(df):
    """Create features from market data."""
    features = df.copy()
    
    if 'asth_symbol' not in features.columns:
        features['asth_symbol'] = 'UNKNOWN'
    
    features['price_momentum_5'] = features.groupby('asth_symbol')['asth_bidPrice'].pct_change(5)
    features['price_momentum_10'] = features.groupby('asth_symbol')['asth_bidPrice'].pct_change(10)
    
    features['volume_mean'] = features.groupby('asth_symbol')['asth_ticketCount'].rolling(50, min_periods=1).mean().reset_index(0, drop=True)
    features['volume_std'] = features.groupby('asth_symbol')['asth_ticketCount'].rolling(50, min_periods=1).std().reset_index(0, drop=True)
    features['volume_zscore'] = (features['asth_ticketCount'] - features['volume_mean']) / (features['volume_std'] + 1e-8)
    
    features['spread_pct'] = (features['asth_askPrice'] - features['asth_bidPrice']) / (features['asth_bidPrice'] + 1e-8)
    features['spread_abs'] = features['asth_askPrice'] - features['asth_bidPrice']
    
    features['weight_change'] = features.groupby('asth_symbol')['asth_lastPriceWeight'].diff()
    features['weight_velocity'] = features.groupby('asth_symbol')['asth_lastPriceWeight'].diff(2)
    
    features['bid_ask_ratio'] = features['asth_bidSize'] / (features['asth_askSize'] + 1e-8)
    features['order_flow_imbalance'] = (features['asth_bidSize'] - features['asth_askSize']) / (features['asth_bidSize'] + features['asth_askSize'] + 1e-8)
    
    return features

class LiveTrader:
    """Live trading engine with paper money."""
    
    def __init__(self, model_path, entry_threshold=0.85, profit_target=0.01, stop_loss=0.005):
        """Initialize live trader."""
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        
        # Trading parameters
        self.entry_threshold = entry_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        
        # State
        self.positions = {}
        self.trades = []
        self.start_time = datetime.now()
        self.start_balance = 10000.0  # Paper money: $10,000
        self.current_balance = self.start_balance
        
        # Data buffer (keep last 200 rows per symbol for feature calculation)
        self.data_buffer = {}
        
    def process_market_data(self, df):
        """Process new market data and make trading decisions."""
        if df is None or len(df) == 0:
            return
        
        # Add to buffer (group by symbol)
        for _, row in df.iterrows():
            symbol = row['asth_symbol']
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = []
            self.data_buffer[symbol].append(row.to_dict())
            # Keep only last 200 rows per symbol
            if len(self.data_buffer[symbol]) > 200:
                self.data_buffer[symbol] = self.data_buffer[symbol][-200:]
        
        # Process each new row
        for _, row in df.iterrows():
            symbol = row['asth_symbol']
            
            # Get buffer for this symbol
            if symbol not in self.data_buffer or len(self.data_buffer[symbol]) < 50:
                continue  # Need at least 50 rows for features
            
            # Create DataFrame from buffer
            symbol_df = pd.DataFrame(self.data_buffer[symbol])
            symbol_df = symbol_df.sort_values('changed_time').reset_index(drop=True)
            
            # Create features
            features = create_features(symbol_df)
            if len(features) == 0:
                continue
            
            # Get latest row with features
            latest_features = features.iloc[-1:]
            latest_features = latest_features[self.feature_cols + ['asth_symbol', 'asth_bidPrice', 'asth_askPrice', 'changed_time']].dropna()
            
            if len(latest_features) == 0:
                continue
            
            # Get prediction
            X = latest_features[self.feature_cols].values
            if len(X) == 0:
                continue
                
            prob = self.model.predict_proba(X)[0, 1]
            mid_price = (row['asth_bidPrice'] + row['asth_askPrice']) / 2
            timestamp = pd.to_datetime(row['changed_time'])
            
            # Check existing positions
            if symbol in self.positions:
                self._update_position(symbol, mid_price, prob, timestamp)
            
            # Check for new entry
            if symbol not in self.positions and prob >= self.entry_threshold:
                self._enter_position(symbol, mid_price, prob, timestamp)
    
    def _enter_position(self, symbol, price, probability, timestamp):
        """Enter a new position."""
        # Paper money: use 10% of balance per trade
        position_size = self.current_balance * 0.1
        quantity = position_size / price
        
        self.positions[symbol] = {
            'entry_price': price,
            'entry_probability': probability,
            'entry_time': timestamp,
            'quantity': quantity,
            'entry_balance': self.current_balance
        }
        
        print(f"{GREEN}[BUY] {symbol} @ ${price:.6f} (prob: {probability:.2%}){RESET}")
    
    def _update_position(self, symbol, current_price, probability, timestamp):
        """Update existing position and check exit conditions."""
        pos = self.positions[symbol]
        current_return = (current_price - pos['entry_price']) / pos['entry_price']
        
        # Exit conditions
        should_exit = False
        exit_reason = None
        
        if current_return >= self.profit_target:
            exit_reason = "PROFIT"
            should_exit = True
        elif current_return <= -self.stop_loss:
            exit_reason = "STOP_LOSS"
            should_exit = True
        elif probability < 0.3 and current_return < -0.002:
            exit_reason = "SIGNAL_LOST"
            should_exit = True
        
        if should_exit:
            self._exit_position(symbol, current_price, exit_reason, timestamp)
    
    def _exit_position(self, symbol, exit_price, reason, timestamp):
        """Exit a position."""
        pos = self.positions[symbol]
        return_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        pnl = pos['quantity'] * (exit_price - pos['entry_price'])
        
        # Update balance
        self.current_balance += pnl
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'return_pct': return_pct * 100,
            'pnl': pnl,
            'reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'duration': (timestamp - pos['entry_time']).total_seconds()
        }
        self.trades.append(trade)
        
        # Print trade result
        if return_pct > 0:
            print(f"{GREEN}[SELL] {symbol} @ ${exit_price:.6f} | +{return_pct*100:.2f}% | {reason}{RESET}")
        else:
            print(f"{RED}[SELL] {symbol} @ ${exit_price:.6f} | {return_pct*100:.2f}% | {reason}{RESET}")
        
        del self.positions[symbol]
    
    def get_statistics(self):
        """Get current trading statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'current_balance': self.current_balance,
                'return_pct': 0
            }
        
        df_trades = pd.DataFrame(self.trades)
        profitable = (df_trades['return_pct'] > 0).sum()
        total_pnl = df_trades['pnl'].sum()
        
        return {
            'total_trades': len(self.trades),
            'profitable': profitable,
            'losses': len(self.trades) - profitable,
            'win_rate': (profitable / len(self.trades)) * 100 if len(self.trades) > 0 else 0,
            'total_pnl': total_pnl,
            'current_balance': self.current_balance,
            'return_pct': ((self.current_balance - self.start_balance) / self.start_balance) * 100,
            'avg_return': df_trades['return_pct'].mean()
        }
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.get_statistics()
        runtime = (datetime.now() - self.start_time).total_seconds() / 60  # minutes
        
        print(f"\n--- Statistics (Runtime: {runtime:.1f} min) ---")
        print(f"Trades: {stats['total_trades']} | Profitable: {stats.get('profitable', 0)} | Losses: {stats.get('losses', 0)}")
        print(f"Win Rate: {stats['win_rate']:.1f}% | Avg Return: {stats.get('avg_return', 0):+.2f}%")
        print(f"Balance: ${stats['current_balance']:.2f} | PnL: ${stats['total_pnl']:+.2f} | Total Return: {stats['return_pct']:+.2f}%")
        print(f"Active Positions: {len(self.positions)}")
        if len(self.positions) > 0:
            print(f"Open positions: {', '.join(self.positions.keys())}")

def main():
    """Main live trading loop."""
    print("Live Trading - Single-Task Model")
    print("=" * 40)
    
    # Initialize
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/glitchcatcher_model_single.pkl')
    trader = LiveTrader(model_path, entry_threshold=0.85, profit_target=0.01, stop_loss=0.005)
    
    # Connect to database
    db = CryptoDatabase()
    if not db.connect():
        print("Failed to connect to database")
        return
    
    print("Connected to database")
    print("Monitoring assets_history for new data...")
    print("Starting balance: $10,000.00")
    print("\n" + "=" * 40)
    
    last_timestamp = None
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Get new data
            df = db.get_latest_history(limit=100, since_timestamp=last_timestamp)
            
            if df is not None and len(df) > 0:
                # Update last timestamp
                if 'changed_time' in df.columns:
                    df['changed_time'] = pd.to_datetime(df['changed_time'])
                    new_timestamp = df['changed_time'].max()
                    if last_timestamp is None or new_timestamp > last_timestamp:
                        last_timestamp = new_timestamp
                
                # Process data
                trader.process_market_data(df)
            
            # Print statistics periodically
            if len(trader.trades) > 0:
                if len(trader.trades) % 5 == 0:  # Every 5 trades
                    trader.print_statistics()
            else:
                # Print every 60 seconds if no trades yet
                if int(time.time()) % 60 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring... (No trades yet)")
            
            # Sleep before next check
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nStopping live trading...")
        trader.print_statistics()
        db.disconnect()

if __name__ == "__main__":
    main()
