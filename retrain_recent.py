#!/usr/bin/env python3
"""
Retrain GlitchCatcher on recent data only.
Use this periodically (e.g. every 2–3 days) to keep the model adapted to current market regime.
"""

import sys
from datetime import datetime

# Add project root so we can import train_glitchcatcher
sys.path.insert(0, '.')

from train_glitchcatcher import train_glitchcatcher_model

def main():
    print("=" * 60)
    print("🔄 GLITCHCATCHER - RETRAIN ON RECENT DATA")
    print("=" * 60)
    print()
    print("Train only on the last N days to adapt to current market conditions.")
    print("(Useful when performance drops after a few days – retrain on fresh data.)")
    print()
    
    # Ask for number of days
    while True:
        try:
            days_input = input("Last N days to use for training (e.g. 3 or 4, default: 3): ").strip()
            if not days_input:
                recent_days = 3
                break
            recent_days = int(days_input)
            if 1 <= recent_days <= 30:
                break
            print("Please enter a number between 1 and 30.")
        except ValueError:
            print("Invalid input. Enter a whole number (e.g. 3).")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return
    
    # Default model name with date
    date_str = datetime.now().strftime("%Y%m%d")
    default_name = f"model_recent_{date_str}.pkl"
    
    name_input = input(f"Model name (default: {default_name}): ").strip()
    model_name = name_input if name_input else default_name
    
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    
    print()
    print(f"→ Training on last {recent_days} days, saving as: {model_name}")
    print()
    
    train_glitchcatcher_model(
        csv_path='assets_history_cleaned_v2.csv',
        recent_days=recent_days,
        model_name_override=model_name
    )
    
    print()
    print("=" * 60)
    print("✅ Retrain complete. Use this model in live_testing.py for the next days.")
    print("=" * 60)

if __name__ == "__main__":
    main()
