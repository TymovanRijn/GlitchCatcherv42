#!/usr/bin/env python3
"""
General Live Testing Script
Monitors database and executes trades with paper money using any selected model.
Supports both multi-task and single-task models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from pathlib import Path
from database import CryptoDatabase

# ANSI colors for terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def convert_to_float(df):
    """Convert all numeric columns (including Decimal) to float."""
    df = df.copy()
    numeric_cols = ['asth_bidPrice', 'asth_askPrice', 'asth_bidSize', 'asth_askSize', 
                    'asth_ticketCount', 'asth_lastPriceWeight', 'asth_spread',
                    'asth_symbolValue', 'asth_changedPercentage', 'asth_symbolValue_USD',
                    'asth_symbolValue_BTC', 'asth_available', 'asth_inOrder', 'asth_actualValue',
                    'asth_pricePrecision', 'asth_lastPriceCount']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    return df

def create_features(df):
    """Create features from market data."""
    # Convert Decimal types to float first
    features = convert_to_float(df)
    
    if 'asth_symbol' not in features.columns:
        features['asth_symbol'] = 'UNKNOWN'
    
    # Ensure we have required columns
    required_cols = ['asth_bidPrice', 'asth_askPrice', 'asth_ticketCount', 
                     'asth_lastPriceWeight', 'asth_bidSize', 'asth_askSize']
    for col in required_cols:
        if col not in features.columns:
            features[col] = 0.0
    
    # Fill NaN values
    features[required_cols] = features[required_cols].fillna(0.0)
    
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
    
    # Fill NaN values in calculated features
    feature_cols = ['price_momentum_5', 'price_momentum_10', 'volume_zscore', 
                    'spread_pct', 'spread_abs', 'weight_change', 'weight_velocity',
                    'bid_ask_ratio', 'order_flow_imbalance']
    for col in feature_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    
    return features

def list_available_models():
    """List all available model files in the models directory."""
    models_dir = Path(__file__).parent.parent / 'models'
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob('*.pkl'))
    return sorted([f.name for f in model_files])

def load_model(model_name):
    """Load a model and return model data with type information."""
    models_dir = Path(__file__).parent.parent / 'models'
    model_path = models_dir / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_data = joblib.load(model_path)
    
    # Determine model type
    is_multi_task = model_data.get('multi_task', False)
    
    if is_multi_task:
        return {
            'type': 'multi_task',
            'model_detection': model_data['model_detection'],
            'model_persistence': model_data['model_persistence'],
            'feature_cols': model_data['feature_cols'],
            'name': model_name,
            'path': str(model_path),
            'metadata': model_data
        }
    else:
        # Single-task model
        return {
            'type': 'single_task',
            'model': model_data.get('model'),
            'feature_cols': model_data.get('feature_cols', []),
            'name': model_name,
            'path': str(model_path),
            'metadata': model_data
        }

def select_model():
    """Interactive model selection."""
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print("📦 AVAILABLE MODELS")
    print(f"{'='*60}{RESET}\n")
    
    models = list_available_models()
    
    if not models:
        print(f"{RED}❌ No models found in models/ directory!{RESET}")
        print(f"   Please train a model first using train_glitchcatcher.py")
        return None
    
    # Display available models
    for i, model in enumerate(models, 1):
        try:
            model_info = load_model(model)
            model_type = model_info['type'].replace('_', ' ').title()
            print(f"   {i}. {BOLD}{model}{RESET} ({model_type})")
            
            # Show additional info if available
            if model_type == 'Multi Task':
                det_metrics = model_info['metadata'].get('detection_test_metrics', {})
                per_metrics = model_info['metadata'].get('persistence_test_metrics', {})
                if det_metrics:
                    print(f"      Detection Accuracy: {det_metrics.get('accuracy', 0)*100:.2f}%")
                if per_metrics:
                    print(f"      Persistence Accuracy: {per_metrics.get('accuracy', 0)*100:.2f}%")
            else:
                test_metrics = model_info['metadata'].get('test_metrics', {})
                if test_metrics:
                    print(f"      Test Accuracy: {test_metrics.get('accuracy', 0)*100:.2f}%")
        except Exception as e:
            print(f"   {i}. {model} {RED}(Error loading: {str(e)}){RESET}")
    
    print()
    
    # Get user selection
    while True:
        try:
            choice = input(f"{BOLD}Select model (1-{len(models)}) or 'q' to quit: {RESET}").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(models):
                selected_model = models[choice_num - 1]
                print(f"\n{BOLD}{GREEN}✅ Selected: {selected_model}{RESET}\n")
                return load_model(selected_model)
            else:
                print(f"{RED}Invalid choice. Please enter a number between 1 and {len(models)}.{RESET}")
        except ValueError:
            print(f"{RED}Invalid input. Please enter a number or 'q' to quit.{RESET}")
        except KeyboardInterrupt:
            print(f"\n{RED}Interrupted by user.{RESET}")
            return None

class LiveTraderGeneral:
    """General live trading engine that works with both multi-task and single-task models."""
    
    def __init__(self, model_info, detection_threshold=0.7, persistence_threshold=0.5, 
                 profit_target=0.02, stop_loss=0.005, max_positions=5):
        """Initialize live trader with model info."""
        self.model_info = model_info
        self.model_type = model_info['type']
        self.model_name = model_info['name']
        
        # Load models based on type
        if self.model_type == 'multi_task':
            self.model_detection = model_info['model_detection']
            self.model_persistence = model_info['model_persistence']
        else:
            self.model = model_info['model']
        
        self.feature_cols = model_info['feature_cols']
        
        # Trading parameters
        self.detection_threshold = detection_threshold
        self.persistence_threshold = persistence_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_positions = max_positions
        
        # State
        self.positions = {}
        self.trades = []
        self.start_time = datetime.now()
        self.start_balance = 10000.0
        self.current_balance = self.start_balance
        
        # Data buffer
        self.data_buffer = {}
        
        # Display model info
        print(f"{BOLD}{CYAN}{'='*60}")
        print(f"🤖 RUNNING MODEL: {self.model_name}")
        print(f"   Type: {self.model_type.replace('_', ' ').title()}")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"{'='*60}{RESET}\n")
    
    def process_market_data(self, df):
        """Process new market data and make trading decisions."""
        if df is None or len(df) == 0:
            return
        
        try:
            # Convert to float first to avoid Decimal issues
            df = convert_to_float(df)
            
            # Ensure required columns exist
            if 'asth_symbol' not in df.columns:
                return
            
            # Filter out invalid data
            df = df[df['asth_bidPrice'] > 0]
            df = df[df['asth_askPrice'] > 0]
            df = df[df['asth_ticketCount'] > 5]  # Minimum liquidity
            
            if len(df) == 0:
                return
            
            # Add to buffer (group by symbol)
            for _, row in df.iterrows():
                try:
                    symbol = str(row['asth_symbol']) if pd.notna(row['asth_symbol']) else 'UNKNOWN'
                    if symbol == 'UNKNOWN' or symbol == '':
                        continue
                    
                    if symbol not in self.data_buffer:
                        self.data_buffer[symbol] = []
                    
                    # Convert row to dict and ensure all values are serializable
                    row_dict = {}
                    for key, value in row.items():
                        if pd.isna(value):
                            row_dict[key] = None
                        else:
                            row_dict[key] = float(value) if isinstance(value, (int, float)) else value
                    
                    self.data_buffer[symbol].append(row_dict)
                    if len(self.data_buffer[symbol]) > 200:
                        self.data_buffer[symbol] = self.data_buffer[symbol][-200:]
                except Exception as e:
                    continue  # Skip problematic rows
            
            # Process each new row
            for _, row in df.iterrows():
                try:
                    symbol = str(row['asth_symbol']) if pd.notna(row['asth_symbol']) else 'UNKNOWN'
                    if symbol == 'UNKNOWN' or symbol == '':
                        continue
                    
                    if symbol not in self.data_buffer or len(self.data_buffer[symbol]) < 50:
                        continue
                    
                    # Create DataFrame from buffer
                    symbol_df = pd.DataFrame(self.data_buffer[symbol])
                    if len(symbol_df) == 0:
                        continue
                    
                    # Sort by time
                    if 'changed_time' in symbol_df.columns:
                        symbol_df['changed_time'] = pd.to_datetime(symbol_df['changed_time'], errors='coerce')
                        symbol_df = symbol_df.sort_values('changed_time').reset_index(drop=True)
                    
                    # Create features
                    features = create_features(symbol_df)
                    if len(features) == 0:
                        continue
                    
                    # Get latest row
                    latest_features = features.iloc[-1:]
                    
                    # Check if we have all required feature columns
                    missing_cols = [col for col in self.feature_cols if col not in latest_features.columns]
                    if missing_cols:
                        continue
                    
                    latest_features = latest_features[self.feature_cols + ['asth_symbol', 'asth_bidPrice', 'asth_askPrice', 'changed_time']].dropna(subset=self.feature_cols)
                    
                    if len(latest_features) == 0:
                        continue
                    
                    # Get predictions based on model type
                    X = latest_features[self.feature_cols].values
                    if len(X) == 0 or X.shape[1] != len(self.feature_cols):
                        continue
                    
                    if self.model_type == 'multi_task':
                        det_prob = self.model_detection.predict_proba(X)[0, 1]
                        per_prob = self.model_persistence.predict_proba(X)[0, 1]
                    else:
                        # Single-task model: use prediction as both detection and persistence
                        prob = self.model.predict_proba(X)[0, 1]
                        det_prob = prob
                        per_prob = prob  # Use same probability for both
                    
                    # Get price from current row (already converted to float)
                    bid_price = float(row['asth_bidPrice'])
                    ask_price = float(row['asth_askPrice'])
                    mid_price = (bid_price + ask_price) / 2.0
                    
                    timestamp = pd.to_datetime(row['changed_time'], errors='coerce')
                    if pd.isna(timestamp):
                        timestamp = datetime.now()
                    
                    # Check existing positions
                    if symbol in self.positions:
                        self._update_position(symbol, mid_price, det_prob, per_prob, timestamp)
                    
                    # Check for new entry (both models must agree, and we haven't reached max positions)
                    if symbol not in self.positions:
                        if det_prob >= self.detection_threshold and per_prob >= self.persistence_threshold:
                            if len(self.positions) < self.max_positions:
                                self._enter_position(symbol, mid_price, det_prob, per_prob, timestamp)
                            
                except Exception as e:
                    continue  # Skip problematic rows
                    
        except Exception as e:
            # Log error but don't crash
            pass
    
    def _enter_position(self, symbol, price, det_prob, per_prob, timestamp):
        """Enter a new position."""
        position_size = self.current_balance * 0.1
        quantity = position_size / price
        
        self.positions[symbol] = {
            'entry_price': price,
            'entry_detection_prob': det_prob,
            'entry_persistence_prob': per_prob,
            'entry_time': timestamp,
            'quantity': quantity,
            'entry_balance': self.current_balance
        }
        
        print(f"{GREEN}[BUY] {symbol} @ ${price:.6f} (det: {det_prob:.2%}, per: {per_prob:.2%}) | Open: {len(self.positions)}/{self.max_positions}{RESET}")
    
    def _update_position(self, symbol, current_price, det_prob, per_prob, timestamp):
        """Update existing position and check exit conditions."""
        pos = self.positions[symbol]
        current_return = (current_price - pos['entry_price']) / pos['entry_price']
        
        should_exit = False
        exit_reason = None
        
        if current_return >= self.profit_target:
            exit_reason = "PROFIT"
            should_exit = True
        elif current_return <= -self.stop_loss:
            exit_reason = "STOP_LOSS"
            should_exit = True
        elif det_prob < self.detection_threshold * 0.5 and per_prob < self.persistence_threshold * 0.5 and current_return < -0.002:
            exit_reason = "SIGNAL_LOST"
            should_exit = True
        
        if should_exit:
            self._exit_position(symbol, current_price, exit_reason, timestamp)
    
    def _exit_position(self, symbol, exit_price, reason, timestamp):
        """Exit a position."""
        pos = self.positions[symbol]
        return_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        pnl = pos['quantity'] * (exit_price - pos['entry_price'])
        
        self.current_balance += pnl
        
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
        
        # Calculate total profit/loss
        total_pnl = self.current_balance - self.start_balance
        total_return_pct = (self.current_balance / self.start_balance - 1) * 100
        
        # Print trade result with capital update
        if return_pct > 0:
            print(f"{GREEN}[SELL] {symbol} @ ${exit_price:.6f} | +{return_pct*100:.2f}% | {reason}{RESET}")
        else:
            print(f"{RED}[SELL] {symbol} @ ${exit_price:.6f} | {return_pct*100:.2f}% | {reason}{RESET}")
        
        # Print capital update
        print(f"  Balance: ${self.current_balance:.2f} | PnL: ${total_pnl:+.2f} ({total_return_pct:+.2f}%) | Open: {len(self.positions)}/{self.max_positions}")
        
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
        
        print(f"\n{BOLD}{CYAN}--- Statistics (Runtime: {runtime:.1f} min) ---{RESET}")
        print(f"Model: {self.model_name}")
        print(f"Trades: {stats['total_trades']} | Profitable: {stats.get('profitable', 0)} | Losses: {stats.get('losses', 0)}")
        print(f"Win Rate: {stats['win_rate']:.1f}% | Avg Return: {stats.get('avg_return', 0):+.2f}%")
        print(f"Balance: ${stats['current_balance']:.2f} | PnL: ${stats['total_pnl']:.2f} | Total Return: {stats['return_pct']:+.2f}%")
        print(f"Active Positions: {len(self.positions)}")
        if len(self.positions) > 0:
            print(f"Open positions: {', '.join(self.positions.keys())}")

def main():
    """Main live trading loop."""
    print(f"{BOLD}{CYAN}{'='*60}")
    print("🚀 GLITCHCATCHER - GENERAL LIVE TESTING")
    print(f"{'='*60}{RESET}\n")
    
    # Select model interactively
    model_info = select_model()
    if model_info is None:
        print(f"{RED}No model selected. Exiting.{RESET}")
        return
    
    # Initialize trader
    trader = LiveTraderGeneral(
        model_info,
        detection_threshold=0.7,
        persistence_threshold=0.5,
        profit_target=0.02,
        stop_loss=0.005,
        max_positions=5
    )
    
    # Connect to database
    db = CryptoDatabase()
    if not db.connect():
        print(f"{RED}Failed to connect to database{RESET}")
        return
    
    print(f"{GREEN}✅ Connected to database{RESET}")
    print(f"Monitoring assets_history for new data...")
    print(f"Starting balance: $10,000.00")
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}\n")
    
    last_timestamp = None
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            df = db.get_latest_history(limit=100, since_timestamp=last_timestamp)
            
            if df is not None and len(df) > 0:
                if 'changed_time' in df.columns:
                    df['changed_time'] = pd.to_datetime(df['changed_time'])
                    new_timestamp = df['changed_time'].max()
                    if last_timestamp is None or new_timestamp > last_timestamp:
                        last_timestamp = new_timestamp
                
                trader.process_market_data(df)
            
            # Print statistics periodically
            if len(trader.trades) > 0:
                if len(trader.trades) % 5 == 0:  # Every 5 trades
                    trader.print_statistics()
            else:
                # Print every 60 seconds if no trades yet
                if int(time.time()) % 60 == 0:
                    print(f"{YELLOW}[{datetime.now().strftime('%H:%M:%S')}] Monitoring... (No trades yet) | Model: {trader.model_name}{RESET}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n\n{BOLD}{YELLOW}Stopping live trading...{RESET}")
        trader.print_statistics()
        db.disconnect()

if __name__ == "__main__":
    main()
