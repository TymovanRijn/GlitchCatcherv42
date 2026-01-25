#!/usr/bin/env python3
"""
GlitchCatcher Model Comparison
Compare Single-Task vs Multi-Task models side-by-side
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Import both trading functions
from test_trading_single import simulate_trading as simulate_single, create_features, analyze_trades
from test_trading import simulate_trading as simulate_multi

def compare_models():
    """Compare both models on the same test data."""
    print("="*70)
    print("🔬 GLITCHCATCHER MODEL COMPARISON")
    print("="*70)
    print("\nComparing:")
    print("  1. Single-Task Model (Original)")
    print("  2. Multi-Task Model (Detection + Persistence)")
    
    # Load both models
    print("\n📦 Loading models...")
    
    try:
        # Single-task model
        single_data = joblib.load('models/glitchcatcher_model_single.pkl')
        single_model = single_data['model']
        single_feature_cols = single_data['feature_cols']
        single_test_acc = single_data.get('test_metrics', {}).get('accuracy', 0)*100
        
        print(f"   ✅ Single-Task Model loaded (Test Accuracy: {single_test_acc:.2f}%)")
    except FileNotFoundError:
        print("   ❌ Single-task model not found! Run 'python train_glitchcatcher_single.py' first.")
        return
    except Exception as e:
        print(f"   ❌ Error loading single-task model: {e}")
        return
    
    try:
        # Multi-task model
        multi_data = joblib.load('models/glitchcatcher_model.pkl')
        if not multi_data.get('multi_task', False):
            print("   ❌ Multi-task model not found! Run 'python train_glitchcatcher.py' first.")
            return
        
        multi_model_detection = multi_data['model_detection']
        multi_model_persistence = multi_data['model_persistence']
        multi_feature_cols = multi_data['feature_cols']
        multi_test_acc = multi_data.get('persistence_test_metrics', {}).get('accuracy', 0)*100
        
        print(f"   ✅ Multi-Task Model loaded (Persistence Test Accuracy: {multi_test_acc:.2f}%)")
    except FileNotFoundError:
        print("   ❌ Multi-task model not found! Run 'python train_glitchcatcher.py' first.")
        return
    except Exception as e:
        print(f"   ❌ Error loading multi-task model: {e}")
        return
    
    # Load test data (same for both)
    print("\n📊 Loading test data (same for both models)...")
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
    
    # Test Single-Task Model
    print("\n" + "="*70)
    print("📊 TESTING SINGLE-TASK MODEL")
    print("="*70)
    
    single_trades = simulate_single(
        single_model,
        single_feature_cols,
        df.copy(),
        entry_threshold=0.85,
        profit_target=0.01,
        stop_loss=0.005,
        max_hold_ticks=30,
        use_trailing_stop=True
    )
    
    single_results = analyze_trades(single_trades, "Single-Task Model") if single_trades else None
    
    # Test Multi-Task Model
    print("\n" + "="*70)
    print("📊 TESTING MULTI-TASK MODEL")
    print("="*70)
    
    multi_trades = simulate_multi(
        multi_model_detection,
        multi_model_persistence,
        multi_feature_cols,
        df.copy(),
        detection_threshold=0.7,
        persistence_threshold=0.5,
        profit_target=0.01,
        stop_loss=0.005,
        max_hold_ticks=30,
        use_trailing_stop=True
    )
    
    multi_results = analyze_trades(multi_trades, "Multi-Task Model") if multi_trades else None
    
    # Side-by-side comparison
    print("\n" + "="*70)
    print("📊 SIDE-BY-SIDE COMPARISON")
    print("="*70)
    
    if single_results and multi_results:
        comparison_df = pd.DataFrame({
            'Metric': [
                'Total Trades',
                'Win Rate (%)',
                'Average Return (%)',
                'Total Return (%)',
                'Average Win (%)',
                'Average Loss (%)',
                'Profit Factor',
                'Expected Value (%)',
                'SIGNAL_LOST %',
                'PROFIT_TARGET %',
                'STOP_LOSS %'
            ],
            'Single-Task': [
                single_results['total_trades'],
                f"{single_results['win_rate']:.2f}",
                f"{single_results['avg_return']:.4f}",
                f"{single_results['total_return']:.4f}",
                f"{single_results['avg_win']:.4f}",
                f"{single_results['avg_loss']:.4f}",
                f"{single_results['profit_factor']:.2f}",
                f"{single_results['expected_value']:.4f}",
                f"{(single_results['exit_reasons'].get('SIGNAL_LOST', 0) / single_results['total_trades'] * 100):.1f}",
                f"{(single_results['exit_reasons'].get('PROFIT_TARGET', 0) / single_results['total_trades'] * 100):.1f}",
                f"{(single_results['exit_reasons'].get('STOP_LOSS', 0) / single_results['total_trades'] * 100):.1f}"
            ],
            'Multi-Task': [
                multi_results['total_trades'],
                f"{multi_results['win_rate']:.2f}",
                f"{multi_results['avg_return']:.4f}",
                f"{multi_results['total_return']:.4f}",
                f"{multi_results['avg_win']:.4f}",
                f"{multi_results['avg_loss']:.4f}",
                f"{multi_results['profit_factor']:.2f}",
                f"{multi_results['expected_value']:.4f}",
                f"{(multi_results['exit_reasons'].get('SIGNAL_LOST', 0) / multi_results['total_trades'] * 100):.1f}",
                f"{(multi_results['exit_reasons'].get('PROFIT_TARGET', 0) / multi_results['total_trades'] * 100):.1f}",
                f"{(multi_results['exit_reasons'].get('STOP_LOSS', 0) / multi_results['total_trades'] * 100):.1f}"
            ]
        })
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Winner analysis
        print("\n" + "="*70)
        print("🏆 WINNER ANALYSIS")
        print("="*70)
        
        winners = []
        
        if multi_results['expected_value'] > single_results['expected_value']:
            winners.append("Multi-Task (Higher Expected Value)")
        elif single_results['expected_value'] > multi_results['expected_value']:
            winners.append("Single-Task (Higher Expected Value)")
        
        if multi_results['win_rate'] > single_results['win_rate']:
            winners.append("Multi-Task (Higher Win Rate)")
        elif single_results['win_rate'] > multi_results['win_rate']:
            winners.append("Single-Task (Higher Win Rate)")
        
        if multi_results['profit_factor'] > single_results['profit_factor']:
            winners.append("Multi-Task (Higher Profit Factor)")
        elif single_results['profit_factor'] > multi_results['profit_factor']:
            winners.append("Single-Task (Higher Profit Factor)")
        
        signal_lost_single = single_results['exit_reasons'].get('SIGNAL_LOST', 0) / single_results['total_trades'] * 100
        signal_lost_multi = multi_results['exit_reasons'].get('SIGNAL_LOST', 0) / multi_results['total_trades'] * 100
        
        if signal_lost_multi < signal_lost_single:
            winners.append("Multi-Task (Less SIGNAL_LOST)")
        elif signal_lost_single < signal_lost_multi:
            winners.append("Single-Task (Less SIGNAL_LOST)")
        
        if winners:
            print("\n📈 Best Performance:")
            for winner in winners:
                print(f"   ✅ {winner}")
        else:
            print("\n🤝 Models perform similarly!")
        
        print(f"\n💡 Recommendation:")
        if multi_results['expected_value'] > single_results['expected_value'] * 1.1:
            print(f"   🏆 Use Multi-Task Model - {((multi_results['expected_value'] / single_results['expected_value'] - 1) * 100):.1f}% better expected value")
        elif single_results['expected_value'] > multi_results['expected_value'] * 1.1:
            print(f"   🏆 Use Single-Task Model - {((single_results['expected_value'] / multi_results['expected_value'] - 1) * 100):.1f}% better expected value")
        else:
            print(f"   🤝 Both models perform well. Choose based on:")
            print(f"      - Multi-Task: More selective, fewer trades, better signal quality")
            print(f"      - Single-Task: More trades, potentially more opportunities")
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")

if __name__ == "__main__":
    compare_models()
