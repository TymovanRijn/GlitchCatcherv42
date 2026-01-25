# GlitchCatcher v4.2 - High-Frequency Anomaly Detection

Professional machine learning model voor het detecteren van momentum anomalies in cryptocurrency markets.

## 🎯 Features

- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Train/Validation/Test Split**: 60/20/20 voor betrouwbare evaluatie
- **Test Data Evaluation**: Volledige metrics op unseen test data
- **Confusion Matrix**: Detailed breakdown van predictions
- **9 Professional Features**: Market microstructure analysis

## 🚀 Gebruik

### Train Models

**Multi-Task Model (Aanbevolen):**
```bash
cd /Users/tymo/Documents/GlitchCatcherv42
source glitch_env/bin/activate
python train_glitchcatcher.py
```
→ Slaat op in: `models/glitchcatcher_model.pkl`

**Single-Task Model (Original - voor vergelijking):**
```bash
python train_glitchcatcher_single.py
```
→ Slaat op in: `models/glitchcatcher_model_single.pkl`

### Test Models

**Test Multi-Task Model:**
```bash
python test_trading.py
```

**Test Single-Task Model:**
```bash
python test_trading_single.py
```

**Vergelijk Beide Modellen:**
```bash
python compare_models.py
```
→ Test beide modellen op dezelfde data en toont side-by-side vergelijking

### Output
Het script geeft je:
- ✅ **Validation Metrics**: Performance op validation set
- ✅ **Test Metrics**: Performance op unseen test data (belangrijkste!)
- ✅ **Accuracy**: Percentage correct predictions
- ✅ **Precision/Recall/F1**: Detailed classification metrics
- ✅ **Confusion Matrix**: True/False positives/negatives
- ✅ **Classification Report**: Per-class performance

### Model Opslaan
Het model wordt opgeslagen in `models/glitchcatcher_model.pkl` met:
- Trained XGBoost model
- Feature columns
- Training metadata
- Validation & Test metrics

## 📊 Model Features

Het model gebruikt 9 professionele features:

1. **price_momentum_5** - 5-tick price momentum
2. **price_momentum_10** - 10-tick price momentum  
3. **volume_zscore** - Volume anomaly (statistical Z-score)
4. **spread_pct** - Bid-ask spread percentage
5. **spread_abs** - Absolute spread
6. **weight_change** - Weight indicator change
7. **weight_velocity** - Weight indicator velocity
8. **bid_ask_ratio** - Bid/ask size ratio
9. **order_flow_imbalance** - Order flow imbalance

## 📈 Metrics Uitleg

### Accuracy
Percentage van alle predictions die correct zijn.

### Precision  
Van alle "anomaly" predictions, hoeveel zijn echt anomalies?  
**Hoe hoger = minder false positives**

### Recall
Van alle echte anomalies, hoeveel detecteert het model?  
**Hoe hoger = minder false negatives**

### F1-Score
Balans tussen Precision en Recall.  
**Hoe hoger = betere overall performance**

### Test Set Accuracy
**Dit is de belangrijkste metric!** Performance op data die het model nog nooit heeft gezien. Dit geeft de echte performance.

## 💹 Trading Test

Test of het model daadwerkelijk profitable trades zou maken:

```bash
python test_trading.py
```

Dit script:
- ✅ Laadt het getrainde model
- ✅ Simuleert trading op nieuwe data
- ✅ Toont win rate, average return, profit factor
- ✅ Analyseert exit reasons (profit target, stop loss, etc.)
- ✅ Toont beste en slechtste trades
- ✅ Berekent expected value per trade

**Trading Parameters** (aanpasbaar in script):
- Entry Threshold: 70% probability
- Profit Target: 0.5%
- Stop Loss: 0.5%
- Max Hold: 20 ticks

## 💡 Model Gebruiken

```python
import joblib
import pandas as pd

# Load model
data = joblib.load('models/glitchcatcher_model.pkl')
model = data['model']
feature_cols = data['feature_cols']

# Check test accuracy
print(f"Test Accuracy: {data['test_metrics']['accuracy']*100:.2f}%")
print(f"Test F1-Score: {data['test_metrics']['f1']:.4f}")

# Use model for predictions
# ... (create features and predict)
```

## 📁 Project Structure

```
GlitchCatcherv42/
├── train_glitchcatcher.py    # 🤖 Main training script
├── assets_history_cleaned.csv # 📈 Market data
├── models/                    # 💾 Trained models
│   └── glitchcatcher_model.pkl
└── requirements.txt          # 📦 Dependencies
```

**Professional ML model voor anomaly detection! 🚀**
