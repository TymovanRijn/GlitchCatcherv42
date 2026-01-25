#!/usr/bin/env python3
"""
GlitchCatcher Trading Test - Single-Task Model (Original)
Test the original single-task model for comparison
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def create_features(df):
    """Create features (same as training)."""
    features = df.copy()
    
    if 'asth_symbol' not in features.columns:
        features['asth_symbol'] = 'UNKNOWN'
    
    # Price momentum
    features['price_momentum_5'] = features.groupby('asth_symbol')['asth_bidPrice'].pct_change(5)
    features['price_momentum_10'] = features.groupby('asth_symbol')['asth_bidPrice'].pct_change(10)
    
    # Volume anomaly
    features['volume_mean'] = features.groupby('asth_symbol')['asth_ticketCount'].rolling(50, min_periods=1).mean().reset_index(0, drop=True)
    features['volume_std'] = features.groupby('asth_symbol')['asth_ticketCount'].rolling(50, min_periods=1).std().reset_index(0, drop=True)
    features['volume_zscore'] = (features['asth_ticketCount'] - features['volume_mean']) / (features['volume_std'] + 1e-8)
    
    # Spread
    features['spread_pct'] = (features['asth_askPrice'] - features['asth_bidPrice']) / (features['asth_bidPrice'] + 1e-8)
    features['spread_abs'] = features['asth_askPrice'] - features['asth_bidPrice']
    
    # Weight changes
    features['weight_change'] = features.groupby('asth_symbol')['asth_lastPriceWeight'].diff()
    features['weight_velocity'] = features.groupby('asth_symbol')['asth_lastPriceWeight'].diff(2)
    
    # Order flow
    features['bid_ask_ratio'] = features['asth_bidSize'] / (features['asth_askSize'] + 1e-8)
    features['order_flow_imbalance'] = (features['asth_bidSize'] - features['asth_askSize']) / (features['asth_bidSize'] + features['asth_askSize'] + 1e-8)
    
    return features

def simulate_trading(model, feature_cols, df, entry_threshold=0.85, profit_target=0.01, stop_loss=0.005, max_hold_ticks=30, use_trailing_stop=True):
    """Simulate trading with single-task model."""
    print("="*60)
    print("📈 SINGLE-TASK TRADING SIMULATION")
    print("="*60)
    print(f"\n⚙️  Trading Parameters:")
    print(f"   Entry Threshold: {entry_threshold*100:.0f}% (probability)")
    print(f"   Profit Target: {profit_target*100:.2f}%")
    print(f"   Stop Loss: {stop_loss*100:.2f}%")
    print(f"   Max Hold: {max_hold_ticks} ticks")
    print(f"   Trailing Stop: {'Yes' if use_trailing_stop else 'No'}")
    
    # Create features
    print("\n🔧 Creating features...")
    features = create_features(df)
    features = features[feature_cols + ['asth_symbol', 'asth_bidPrice', 'asth_askPrice']].dropna()
    
    if len(features) == 0:
        print("❌ No valid features found")
        return None
    
    # Get predictions
    X = features[feature_cols].values
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    
    print(f"   Processed: {len(features):,} samples")
    print(f"   Signals detected: {predictions.sum():,} ({predictions.mean()*100:.2f}%)")
    
    # Simulate trades
    print("\n💹 Simulating trades...")
    
    trades = []
    current_position = None
    
    for i in range(len(features)):
        row = features.iloc[i]
        prob = probabilities[i]
        mid_price = (row['asth_bidPrice'] + row['asth_askPrice']) / 2
        
        # Check if we should enter
        if current_position is None and prob >= entry_threshold:
            current_position = {
                'entry_index': i,
                'entry_price': mid_price,
                'entry_probability': prob,
                'symbol': row['asth_symbol'],
                'entry_tick': i
            }
        
        # Check if we should exit
        if current_position is not None:
            current_return = (mid_price - current_position['entry_price']) / current_position['entry_price']
            ticks_held = i - current_position['entry_tick']
            
            # Track highest return
            if 'highest_return' not in current_position:
                current_position['highest_return'] = current_return
            else:
                current_position['highest_return'] = max(current_position['highest_return'], current_return)
            
            # Exit conditions
            exit_reason = None
            should_exit = False
            
            # 1. Profit target
            if current_return >= profit_target:
                if current_return > 0.5:
                    exit_reason = "PROFIT_TARGET_CAPPED"
                else:
                    exit_reason = "PROFIT_TARGET"
                should_exit = True
            
            # 2. Stop loss
            elif current_return <= -stop_loss or current_return <= -0.1:
                exit_reason = "STOP_LOSS"
                should_exit = True
            
            # 3. Trailing stop
            elif use_trailing_stop and current_position['highest_return'] > 0.003:
                trailing_stop_level = current_position['highest_return'] * 0.5
                if current_return < trailing_stop_level:
                    exit_reason = "TRAILING_STOP"
                    should_exit = True
            
            # 4. Max hold
            elif ticks_held >= max_hold_ticks:
                exit_reason = "MAX_HOLD"
                should_exit = True
            
            # 5. Signal lost
            elif prob < 0.3 and current_return < -0.002:
                exit_reason = "SIGNAL_LOST"
                should_exit = True
            
            if should_exit:
                trade = {
                    'symbol': current_position['symbol'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': mid_price,
                    'entry_probability': current_position['entry_probability'],
                    'exit_probability': prob,
                    'return_pct': current_return * 100,
                    'ticks_held': ticks_held,
                    'exit_reason': exit_reason,
                    'entry_index': current_position['entry_index'],
                    'exit_index': i
                }
                trades.append(trade)
                current_position = None
    
    # Close open position
    if current_position is not None:
        last_row = features.iloc[-1]
        last_price = (last_row['asth_bidPrice'] + last_row['asth_askPrice']) / 2
        current_return = (last_price - current_position['entry_price']) / current_position['entry_price']
        
        trade = {
            'symbol': current_position['symbol'],
            'entry_price': current_position['entry_price'],
            'exit_price': last_price,
            'entry_probability': current_position['entry_probability'],
            'exit_probability': probabilities[-1],
            'return_pct': current_return * 100,
            'ticks_held': len(features) - current_position['entry_tick'],
            'exit_reason': "END_OF_DATA",
            'entry_index': current_position['entry_index'],
            'exit_index': len(features) - 1
        }
        trades.append(trade)
    
    return trades

def analyze_trades(trades, model_name="Model"):
    """Analyze trading results."""
    if not trades:
        print(f"\n❌ No trades executed for {model_name}!")
        return None
    
    df_trades = pd.DataFrame(trades)
    
    # Cap extreme returns
    df_trades['return_pct_capped'] = df_trades['return_pct'].clip(lower=-10, upper=50)
    original_extreme_count = ((df_trades['return_pct'] > 50) | (df_trades['return_pct'] < -10)).sum()
    if original_extreme_count > 0:
        print(f"   ⚠️  Capped {original_extreme_count} extreme returns for realistic analysis...")
    df_trades['return_pct'] = df_trades['return_pct_capped']
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name.upper()} - TRADING RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {len(trades):,}")
    print(f"   Profitable: {(df_trades['return_pct'] > 0).sum():,}")
    print(f"   Losses: {(df_trades['return_pct'] <= 0).sum():,}")
    
    win_rate = (df_trades['return_pct'] > 0).mean() * 100
    avg_return = df_trades['return_pct'].mean()
    total_return = df_trades['return_pct'].sum()
    avg_win = df_trades[df_trades['return_pct'] > 0]['return_pct'].mean() if (df_trades['return_pct'] > 0).any() else 0
    avg_loss = df_trades[df_trades['return_pct'] <= 0]['return_pct'].mean() if (df_trades['return_pct'] <= 0).any() else 0
    
    print(f"\n💰 Performance Metrics:")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Average Return: {avg_return:.4f}%")
    print(f"   Total Return: {total_return:.4f}%")
    print(f"   Average Win: {avg_win:.4f}%")
    print(f"   Average Loss: {avg_loss:.4f}%")
    
    if avg_loss != 0:
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        print(f"   Profit Factor: {profit_factor:.2f}")
    
    print(f"\n📊 Exit Reasons:")
    exit_reasons = df_trades['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = (count / len(trades)) * 100
        avg_return_reason = df_trades[df_trades['exit_reason'] == reason]['return_pct'].mean()
        print(f"   {reason:15s}: {count:4d} ({pct:5.1f}%) | Avg Return: {avg_return_reason:+.2f}%")
    
    # Expected value
    expected_value = avg_return * (win_rate / 100) - abs(avg_loss) * ((100 - win_rate) / 100)
    print(f"\n💡 Expected Value per Trade: {expected_value:.4f}%")
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
        'expected_value': expected_value,
        'exit_reasons': exit_reasons.to_dict()
    }

def main():
    """Main trading test for single-task model."""
    print("="*60)
    print("🚀 GLITCHCATCHER TRADING TEST - SINGLE-TASK MODEL")
    print("="*60)
    
    # Load model
    print("\n📦 Loading single-task model...")
    try:
        model_data = joblib.load('models/glitchcatcher_model_single.pkl')
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        print(f"   ✅ Model loaded!")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Test Accuracy: {model_data.get('test_metrics', {}).get('accuracy', 0)*100:.2f}%")
    except FileNotFoundError:
        print("   ❌ Model not found! Run 'python train_glitchcatcher_single.py' first.")
        return
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Load test data
    print("\n📊 Loading test data...")
    df_all = pd.read_csv('assets_history_cleaned.csv', nrows=350000)
    df = df_all.iloc[300000:].copy()
    df = df[df['asth_bidPrice'] > 0].copy()
    df = df.sort_values(['asth_symbol', 'changed_time']).reset_index(drop=True)
    
    # Filter extreme movements
    print("   Filtering extreme price movements...")
    df['price_change'] = df.groupby('asth_symbol')['asth_bidPrice'].pct_change().abs()
    df = df[df['price_change'] < 0.5].copy()
    df = df.drop('price_change', axis=1)
    df = df[df['asth_ticketCount'] > 5].copy()
    
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Symbols: {df['asth_symbol'].nunique()} unique")
    
    # Simulate trading
    trades = simulate_trading(
        model, 
        feature_cols, 
        df,
        entry_threshold=0.85,
        profit_target=0.01,
        stop_loss=0.005,
        max_hold_ticks=30,
        use_trailing_stop=True
    )
    
    # Analyze results
    if trades:
        results = analyze_trades(trades, "Single-Task Model")
        
        print(f"\n{'='*60}")
        print("✅ TRADING TEST COMPLETE")
        print(f"{'='*60}")
    else:
        print("\n⚠️  No trades were executed.")

if __name__ == "__main__":
    main()
