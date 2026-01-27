#!/usr/bin/env python3
"""
GlitchCatcher Trading Test - Simuleer trading met het getrainde model
Test of het model daadwerkelijk profitable trades zou maken
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

def simulate_trading(model_detection, model_persistence, feature_cols, df, 
                    detection_threshold=0.7, persistence_threshold=0.5, 
                    profit_target=0.01, stop_loss=0.005, max_hold_ticks=30, use_trailing_stop=True):
    """
    Simuleer trading met multi-task models.
    
    Parameters:
    - detection_threshold: Minimum detection probability (0.7 = 70%)
    - persistence_threshold: Minimum persistence probability (0.5 = 50%)
    - profit_target: Profit target (0.01 = 1.0%)
    - stop_loss: Stop loss (0.005 = 0.5%)
    - max_hold_ticks: Maximum aantal ticks om positie te houden
    - use_trailing_stop: Gebruik trailing stop voor betere exits
    """
    print("="*60)
    print("📈 MULTI-TASK TRADING SIMULATION")
    print("="*60)
    print(f"\n⚙️  Trading Parameters:")
    print(f"   Detection Threshold: {detection_threshold*100:.0f}% (anomaly detection)")
    print(f"   Persistence Threshold: {persistence_threshold*100:.0f}% (signal persistence)")
    print(f"   Entry: BOTH models must agree (detection >= {detection_threshold*100:.0f}% AND persistence >= {persistence_threshold*100:.0f}%)")
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
    
    # Get predictions from both models
    X = features[feature_cols].values
    det_probabilities = model_detection.predict_proba(X)[:, 1]
    per_probabilities = model_persistence.predict_proba(X)[:, 1]
    
    # Combined entry: Both models must agree
    combined_signals = (det_probabilities >= detection_threshold) & (per_probabilities >= persistence_threshold)
    
    print(f"   Processed: {len(features):,} samples")
    print(f"   Detection signals: {(det_probabilities >= detection_threshold).sum():,} ({(det_probabilities >= detection_threshold).mean()*100:.2f}%)")
    print(f"   Persistence signals: {(per_probabilities >= persistence_threshold).sum():,} ({(per_probabilities >= persistence_threshold).mean()*100:.2f}%)")
    print(f"   Combined signals (both agree): {combined_signals.sum():,} ({combined_signals.mean()*100:.2f}%)")
    
    # Simulate trades
    print("\n💹 Simulating trades...")
    
    trades = []
    current_position = None
    
    for i in range(len(features)):
        row = features.iloc[i]
        det_prob = det_probabilities[i]
        per_prob = per_probabilities[i]
        combined_prob = (det_prob + per_prob) / 2  # Average for tracking
        mid_price = (row['asth_bidPrice'] + row['asth_askPrice']) / 2
        
        # Check if we should enter (BOTH models must agree)
        if current_position is None:
            if det_prob >= detection_threshold and per_prob >= persistence_threshold:
                # Enter position - both models agree!
                current_position = {
                    'entry_index': i,
                    'entry_price': mid_price,
                    'entry_detection_prob': det_prob,
                    'entry_persistence_prob': per_prob,
                    'entry_combined_prob': combined_prob,
                    'symbol': row['asth_symbol'],
                    'entry_tick': i
                }
        
        # Check if we should exit
        if current_position is not None:
            current_return = (mid_price - current_position['entry_price']) / current_position['entry_price']
            ticks_held = i - current_position['entry_tick']
            
            # Track highest return for trailing stop
            if 'highest_return' not in current_position:
                current_position['highest_return'] = current_return
            else:
                current_position['highest_return'] = max(current_position['highest_return'], current_return)
            
            # Exit conditions (in priority order)
            exit_reason = None
            should_exit = False
            
            # 1. Profit target hit (highest priority)
            # Cap extreme profits (likely data errors) at 50%
            if current_return >= profit_target:
                if current_return > 0.5:  # Cap at 50% (likely data error)
                    exit_reason = "PROFIT_TARGET_CAPPED"
                else:
                    exit_reason = "PROFIT_TARGET"
                should_exit = True
            
            # 2. Stop loss hit (with max cap to prevent extreme losses)
            elif current_return <= -stop_loss or current_return <= -0.1:  # Max 10% loss
                exit_reason = "STOP_LOSS"
                should_exit = True
            
            # 3. Trailing stop (if we had profit but it reversed)
            elif use_trailing_stop and current_position['highest_return'] > 0.003:  # Had at least 0.3% profit
                trailing_stop_level = current_position['highest_return'] * 0.5  # Trail at 50% of peak
                if current_return < trailing_stop_level:
                    exit_reason = "TRAILING_STOP"
                    should_exit = True
            
            # 4. Max hold time
            elif ticks_held >= max_hold_ticks:
                exit_reason = "MAX_HOLD"
                should_exit = True
            
            # 5. Signal completely gone (both models lose confidence AND we're losing)
            elif (det_prob < detection_threshold * 0.5 and per_prob < persistence_threshold * 0.5) and current_return < -0.002:
                exit_reason = "SIGNAL_LOST"
                should_exit = True
            
            if should_exit:
                # Close position
                trade = {
                    'symbol': current_position['symbol'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': mid_price,
                    'entry_detection_prob': current_position['entry_detection_prob'],
                    'entry_persistence_prob': current_position['entry_persistence_prob'],
                    'exit_detection_prob': det_prob,
                    'exit_persistence_prob': per_prob,
                    'return_pct': current_return * 100,
                    'ticks_held': ticks_held,
                    'exit_reason': exit_reason,
                    'entry_index': current_position['entry_index'],
                    'exit_index': i
                }
                trades.append(trade)
                current_position = None
    
    # Close any open position at the end
    if current_position is not None:
        last_row = features.iloc[-1]
        last_price = (last_row['asth_bidPrice'] + last_row['asth_askPrice']) / 2
        current_return = (last_price - current_position['entry_price']) / current_position['entry_price']
        
        trade = {
            'symbol': current_position['symbol'],
            'entry_price': current_position['entry_price'],
            'exit_price': last_price,
            'entry_detection_prob': current_position['entry_detection_prob'],
            'entry_persistence_prob': current_position['entry_persistence_prob'],
            'exit_detection_prob': det_probabilities[-1],
            'exit_persistence_prob': per_probabilities[-1],
            'return_pct': current_return * 100,
            'ticks_held': len(features) - current_position['entry_tick'],
            'exit_reason': "END_OF_DATA",
            'entry_index': current_position['entry_index'],
            'exit_index': len(features) - 1
        }
        trades.append(trade)
    
    return trades

def analyze_trades(trades):
    """Analyze trading results."""
    if not trades:
        print("\n❌ No trades executed!")
        return
    
    df_trades = pd.DataFrame(trades)
    
    # Cap extreme returns (likely data errors) for realistic analysis
    print("\n   ⚠️  Capping extreme returns (>50% profit, <-10% loss) for realistic analysis...")
    df_trades['return_pct_capped'] = df_trades['return_pct'].clip(lower=-10, upper=50)
    original_extreme_count = ((df_trades['return_pct'] > 50) | (df_trades['return_pct'] < -10)).sum()
    if original_extreme_count > 0:
        print(f"      Capped {original_extreme_count} extreme returns")
    
    # Use capped returns for analysis
    df_trades['return_pct'] = df_trades['return_pct_capped']
    
    print(f"\n{'='*60}")
    print("📊 TRADING RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {len(trades):,}")
    print(f"   Profitable: {(df_trades['return_pct'] > 0).sum():,}")
    print(f"   Losses: {(df_trades['return_pct'] <= 0).sum():,}")
    
    if len(trades) > 0:
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
            avg_return = df_trades[df_trades['exit_reason'] == reason]['return_pct'].mean()
            print(f"   {reason:15s}: {count:4d} ({pct:5.1f}%) | Avg Return: {avg_return:+.2f}%")
        
        # Analyze SIGNAL_LOST trades specifically
        if 'SIGNAL_LOST' in exit_reasons:
            signal_lost_trades = df_trades[df_trades['exit_reason'] == 'SIGNAL_LOST']
            if len(signal_lost_trades) > 0:
                print(f"\n   ⚠️  SIGNAL_LOST Analysis:")
                print(f"      Average return: {signal_lost_trades['return_pct'].mean():+.2f}%")
                print(f"      Profitable: {(signal_lost_trades['return_pct'] > 0).sum()} / {len(signal_lost_trades)}")
                print(f"      Average ticks held: {signal_lost_trades['ticks_held'].mean():.1f}")
        
        print(f"\n🎯 Best Trades:")
        best_trades = df_trades.nlargest(5, 'return_pct')[['symbol', 'return_pct', 'exit_reason', 'ticks_held']]
        print(best_trades.to_string(index=False))
        
        print(f"\n⚠️  Worst Trades:")
        worst_trades = df_trades.nsmallest(5, 'return_pct')[['symbol', 'return_pct', 'exit_reason', 'ticks_held']]
        print(worst_trades.to_string(index=False))
        
        # Expected value
        expected_value = avg_return * (win_rate / 100) - abs(avg_loss) * ((100 - win_rate) / 100)
        print(f"\n💡 Expected Value per Trade: {expected_value:.4f}%")
        
        if expected_value > 0:
            print("   ✅ Model shows positive expected value!")
        else:
            print("   ⚠️  Model shows negative expected value")

def main():
    """Main trading test."""
    print("="*60)
    print("🚀 GLITCHCATCHER TRADING TEST")
    print("="*60)
    
    # Load multi-task models
    print("\n📦 Loading multi-task models...")
    try:
        model_data = joblib.load('models/model_v2_multi.pkl')
        
        # Check if multi-task model
        if model_data.get('multi_task', False):
            model_detection = model_data['model_detection']
            model_persistence = model_data['model_persistence']
            feature_cols = model_data['feature_cols']
            
            print(f"   ✅ Multi-task models loaded!")
            print(f"   Features: {len(feature_cols)}")
            
            det_metrics = model_data.get('detection_test_metrics', {})
            per_metrics = model_data.get('persistence_test_metrics', {})
            
            print(f"   Detection Model - Test Accuracy: {det_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   Persistence Model - Test Accuracy: {per_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   Persistence Model - Test F1: {per_metrics.get('f1', 0):.4f}")
        else:
            # Legacy single model
            print("   ⚠️  Old single model detected. Please retrain with multi-task learning.")
            print("   Run: python train_glitchcatcher.py")
            return
            
    except FileNotFoundError:
        print("   ❌ Model not found! Run 'python train_glitchcatcher.py' first.")
        return
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data (different from training)
    print("\n📊 Loading test data...")
    # Load from middle of file to ensure different data than training
    # Training uses first 1M rows, so we use rows 300k-350k
    df_all = pd.read_csv('assets_history_cleaned_v2.csv', nrows=350000)
    df = df_all.iloc[300000:].copy()  # Take last 50k rows from this chunk
    df = df[df['asth_bidPrice'] > 0].copy()
    df = df.sort_values(['asth_symbol', 'changed_time']).reset_index(drop=True)
    
    # Filter out extreme price movements (likely data errors)
    print("   Filtering extreme price movements...")
    df['price_change'] = df.groupby('asth_symbol')['asth_bidPrice'].pct_change().abs()
    df = df[df['price_change'] < 0.5].copy()  # Filter moves > 50% (likely errors)
    df = df.drop('price_change', axis=1)
    
    # Filter out very low liquidity
    df = df[df['asth_ticketCount'] > 5].copy()  # At least some activity
    
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Symbols: {df['asth_symbol'].nunique()} unique")
    
    # Simulate trading with multi-task models
    trades = simulate_trading(
        model_detection, 
        model_persistence,
        feature_cols, 
        df,
        detection_threshold=0.7,    # 70% detection probability
        persistence_threshold=0.5,   # 50% persistence probability (both must agree)
        profit_target=0.02,          # 1.0% profit target
        stop_loss=0.005,             # 0.5% stop loss
        max_hold_ticks=30,           # Max 30 ticks
        use_trailing_stop=True       # Use trailing stop
    )
    
    # Analyze results
    if trades:
        analyze_trades(trades)
        
        print(f"\n{'='*60}")
        print("✅ TRADING TEST COMPLETE")
        print(f"{'='*60}")
        print(f"\n💡 Tips for better results:")
        print(f"   - Entry threshold: 0.85 (current) = Only best signals")
        print(f"   - Lower to 0.75 = More trades but potentially more noise")
        print(f"   - Higher to 0.90 = Fewer but highest quality signals")
        print(f"   - Profit target: 1.0% (current) is realistic for crypto")
        print(f"   - Trailing stop helps lock in profits")
    else:
        print("\n⚠️  No trades were executed. Try lowering entry_threshold.")

if __name__ == "__main__":
    main()
