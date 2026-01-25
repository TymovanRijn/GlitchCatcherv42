#!/usr/bin/env python3
"""
GlitchCatcher v4.2 - Single-Task Model (Original)
Original model without multi-task learning - for comparison
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
from datetime import datetime

def create_features(df):
    """Create professional features from market microstructure data."""
    features = df.copy()
    
    if 'asth_symbol' not in features.columns:
        features['asth_symbol'] = 'UNKNOWN'
    
    # Price momentum (5-tick and 10-tick)
    features['price_momentum_5'] = features.groupby('asth_symbol')['asth_bidPrice'].pct_change(5)
    features['price_momentum_10'] = features.groupby('asth_symbol')['asth_bidPrice'].pct_change(10)
    
    # Volume anomaly detection (Z-score)
    features['volume_mean'] = features.groupby('asth_symbol')['asth_ticketCount'].rolling(50, min_periods=1).mean().reset_index(0, drop=True)
    features['volume_std'] = features.groupby('asth_symbol')['asth_ticketCount'].rolling(50, min_periods=1).std().reset_index(0, drop=True)
    features['volume_zscore'] = (features['asth_ticketCount'] - features['volume_mean']) / (features['volume_std'] + 1e-8)
    
    # Spread analysis
    features['spread_pct'] = (features['asth_askPrice'] - features['asth_bidPrice']) / (features['asth_bidPrice'] + 1e-8)
    features['spread_abs'] = features['asth_askPrice'] - features['asth_bidPrice']
    
    # Weight indicator changes (momentum signal)
    features['weight_change'] = features.groupby('asth_symbol')['asth_lastPriceWeight'].diff()
    features['weight_velocity'] = features.groupby('asth_symbol')['asth_lastPriceWeight'].diff(2)
    
    # Order flow imbalance
    features['bid_ask_ratio'] = features['asth_bidSize'] / (features['asth_askSize'] + 1e-8)
    features['order_flow_imbalance'] = (features['asth_bidSize'] - features['asth_askSize']) / (features['asth_bidSize'] + features['asth_askSize'] + 1e-8)
    
    return features

def evaluate_model(y_true, y_pred, y_proba, dataset_name="Dataset"):
    """Comprehensive model evaluation with all metrics."""
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name.upper()} EVALUATION")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = 0.0
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            pass
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    if auc > 0:
        print(f"  AUC-ROC:   {auc:.4f}")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"  Actual Neg    {tn:6d}    {fp:6d}")
    print(f"  Actual Pos    {fn:6d}    {tp:6d}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=['Normal', 'Anomaly']))
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    print(f"  Total samples: {len(y_true):,}")
    print(f"  Negative (0):   {(y_true == 0).sum():,} ({(y_true == 0).mean()*100:.1f}%)")
    print(f"  Positive (1):   {(y_true == 1).sum():,} ({(y_true == 1).mean()*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def train_single_task_model():
    """Train original single-task model."""
    print("="*60)
    print("🚀 GLITCHCATCHER v4.2 - SINGLE-TASK MODEL (ORIGINAL)")
    print("="*60)
    
    # Load data
    print("\n📊 Loading cryptocurrency market data...")
    df = pd.read_csv('assets_history_cleaned.csv', nrows=1000000)
    df = df[df['asth_bidPrice'] > 0].copy()
    df = df.sort_values(['asth_symbol', 'changed_time']).reset_index(drop=True)
    
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Symbols: {df['asth_symbol'].nunique()} unique assets")
    
    # Create features
    print("\n🔧 Creating features from market microstructure...")
    features = create_features(df)
    
    # Original single-task labels
    print("\n🎯 Creating target labels (original method)...")
    momentum_threshold = features['price_momentum_5'].quantile(0.85)
    volume_threshold = features['volume_zscore'].quantile(0.75)
    
    features['label'] = (
        (features['price_momentum_5'] > momentum_threshold) &
        (features['volume_zscore'] > volume_threshold)
    ).astype(int)
    
    # Ensure we have enough positive labels
    if features['label'].sum() < 100:
        print("   ⚠️  Adjusting label threshold for better class balance...")
        features['label'] = (features['price_momentum_5'] > features['price_momentum_5'].quantile(0.90)).astype(int)
    
    # Select feature columns
    feature_cols = [
        'price_momentum_5', 'price_momentum_10',
        'volume_zscore',
        'spread_pct', 'spread_abs',
        'weight_change', 'weight_velocity',
        'bid_ask_ratio', 'order_flow_imbalance'
    ]
    
    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in features.columns]
    features = features[feature_cols + ['label', 'asth_symbol']].dropna()
    
    X = features[feature_cols].values
    y = features['label'].values
    
    print(f"   Features created: {len(feature_cols)}")
    print(f"   Valid samples: {len(X):,}")
    print(f"   Positive labels: {y.sum():,} ({y.mean()*100:.2f}%)")
    
    # Train/Validation/Test split (60/20/20)
    print("\n📦 Splitting data: Train (60%) / Validation (20%) / Test (20%)")
    train_size = int(0.6 * len(X))
    val_size = int(0.8 * len(X))
    
    X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
    
    print(f"   Training set:   {len(X_train):,} samples")
    print(f"   Validation set: {len(X_val):,} samples")
    print(f"   Test set:       {len(X_test):,} samples")
    
    # Train model
    print("\n🤖 Training XGBoost Classifier (Single-Task)...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Train with validation monitoring
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print("   ✅ Training completed!")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_model(y_val, y_val_pred, y_val_proba, "Validation Set")
    
    # Evaluate on test set (unseen data)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, "Test Set")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\nValidation Set:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
    print(f"\nTest Set (Unseen Data):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    
    # Save model with metadata
    print("\n💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,  # Single model (original)
        'feature_cols': feature_cols,
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'multi_task': False  # Mark as single-task
    }
    
    model_path = 'models/glitchcatcher_model_single.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"   ✅ Model saved to: {model_path}")
    print(f"   📈 Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"   📈 Test F1-Score: {test_metrics['f1']:.4f}")
    
    return model, feature_cols, test_metrics

if __name__ == "__main__":
    # Train single-task model
    model, feature_cols, test_metrics = train_single_task_model()
    
    print(f"\n{'='*60}")
    print("✅ SINGLE-TASK TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n💡 Model saved to: models/glitchcatcher_model_single.pkl")
    print(f"📊 Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"📊 Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"\nThis is the ORIGINAL single-task model for comparison.")
    print(f"Use 'python train_glitchcatcher.py' for the multi-task version.")
