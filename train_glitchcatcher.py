#!/usr/bin/env python3
"""
GlitchCatcher v4.2 - High-Frequency Anomaly Detection Model
Multi-Task Learning: Detection + Signal Persistence Prediction
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
    
    # Ensure we have symbol column
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
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC if we have probabilities
    auc = 0.0
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Print metrics
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
    
    # Class distribution
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

def train_glitchcatcher_model():
    """Train GlitchCatcher model with comprehensive evaluation."""
    print("="*60)
    print("🚀 GLITCHCATCHER v4.2 - MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n📊 Loading cryptocurrency market data...")
    df = pd.read_csv('assets_history_cleaned.csv', nrows=1000000)  # More data for better model
    df = df[df['asth_bidPrice'] > 0].copy()
    df = df.sort_values(['asth_symbol', 'changed_time']).reset_index(drop=True)
    
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Symbols: {df['asth_symbol'].nunique()} unique assets")
    
    # Create features
    print("\n🔧 Creating features from market microstructure...")
    features = create_features(df)
    
    # Multi-Task Learning: Create two types of labels
    print("\n🎯 Creating multi-task labels...")
    print("   Task 1: Anomaly Detection (is there an anomaly now?)")
    print("   Task 2: Signal Persistence (does signal persist for 5+ ticks?)")
    
    momentum_threshold = features['price_momentum_5'].quantile(0.85)
    volume_threshold = features['volume_zscore'].quantile(0.75)
    
    # Task 1: Detection label (instantaneous anomaly)
    features['label_detection'] = (
        (features['price_momentum_5'] > momentum_threshold) &
        (features['volume_zscore'] > volume_threshold)
    ).astype(int)
    
    # Ensure we have enough detection labels
    if features['label_detection'].sum() < 100:
        print("   ⚠️  Adjusting detection threshold for better class balance...")
        features['label_detection'] = (features['price_momentum_5'] > features['price_momentum_5'].quantile(0.90)).astype(int)
    
    # Task 2: Persistence label (signal persists for at least 5 ticks)
    print("   Creating persistence labels...")
    features['label_persistence'] = 0
    
    # More efficient: use vectorized operations where possible
    # For each symbol, check if anomaly persists
    for symbol in features['asth_symbol'].unique():
        symbol_mask = features['asth_symbol'] == symbol
        symbol_data = features[symbol_mask].copy()
        symbol_indices = symbol_data.index.values
        
        # Get detection labels for this symbol
        detection_mask = symbol_data['label_detection'] == 1
        detection_indices = symbol_indices[detection_mask]
        
        # For each detection, check persistence
        for det_idx in detection_indices:
            # Find position in symbol_indices
            det_pos = np.where(symbol_indices == det_idx)[0][0]
            
            # Get next 5 ticks (if available)
            if det_pos + 5 < len(symbol_indices):
                future_positions = range(det_pos + 1, min(det_pos + 6, len(symbol_indices)))
                future_indices = symbol_indices[future_positions]
                
                if len(future_indices) >= 5:
                    # Check if momentum and volume stay high
                    future_momentum = features.loc[future_indices, 'price_momentum_5'].values
                    future_volume = features.loc[future_indices, 'volume_zscore'].values
                    
                    # Signal persists if at least 3 out of 5 future ticks maintain high momentum/volume
                    momentum_persists = (future_momentum > momentum_threshold * 0.7).sum() >= 3
                    volume_persists = (future_volume > volume_threshold * 0.7).sum() >= 3
                    
                    if momentum_persists and volume_persists:
                        features.loc[det_idx, 'label_persistence'] = 1
    
    # Use persistence label as main label (more conservative, better for trading)
    features['label'] = features['label_persistence'].copy()
    
    print(f"   Detection labels (instant): {features['label_detection'].sum():,} ({features['label_detection'].mean()*100:.2f}%)")
    print(f"   Persistence labels (5+ ticks): {features['label_persistence'].sum():,} ({features['label_persistence'].mean()*100:.2f}%)")
    
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
    features = features[feature_cols + ['label', 'label_detection', 'label_persistence', 'asth_symbol']].dropna()
    
    X = features[feature_cols].values
    y_detection = features['label_detection'].values
    y_persistence = features['label_persistence'].values
    
    print(f"   Features created: {len(feature_cols)}")
    print(f"   Valid samples: {len(X):,}")
    print(f"   Detection labels: {y_detection.sum():,} ({y_detection.mean()*100:.2f}%)")
    print(f"   Persistence labels: {y_persistence.sum():,} ({y_persistence.mean()*100:.2f}%)")
    
    # Train/Validation/Test split (60/20/20)
    print("\n📦 Splitting data: Train (60%) / Validation (20%) / Test (20%)")
    train_size = int(0.6 * len(X))
    val_size = int(0.8 * len(X))
    
    X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]
    y_det_train, y_det_val, y_det_test = y_detection[:train_size], y_detection[train_size:val_size], y_detection[val_size:]
    y_per_train, y_per_val, y_per_test = y_persistence[:train_size], y_persistence[train_size:val_size], y_persistence[val_size:]
    
    print(f"   Training set:   {len(X_train):,} samples")
    print(f"   Validation set: {len(X_val):,} samples")
    print(f"   Test set:       {len(X_test):,} samples")
    
    # Multi-Task Learning: Train two models
    print("\n🤖 Training Multi-Task Models...")
    
    # Model 1: Detection Model (detects anomalies)
    print("\n   📍 Task 1: Training Detection Model...")
    model_detection = xgb.XGBClassifier(
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
    
    model_detection.fit(
        X_train, y_det_train,
        eval_set=[(X_train, y_det_train), (X_val, y_det_val)],
        verbose=False
    )
    
    # Evaluate detection model
    y_det_val_pred = model_detection.predict(X_val)
    y_det_val_proba = model_detection.predict_proba(X_val)[:, 1]
    det_val_metrics = evaluate_model(y_det_val, y_det_val_pred, y_det_val_proba, "Detection Model - Validation")
    
    y_det_test_pred = model_detection.predict(X_test)
    y_det_test_proba = model_detection.predict_proba(X_test)[:, 1]
    det_test_metrics = evaluate_model(y_det_test, y_det_test_pred, y_det_test_proba, "Detection Model - Test")
    
    # Model 2: Persistence Model (predicts if signal persists)
    print("\n   📍 Task 2: Training Persistence Model...")
    model_persistence = xgb.XGBClassifier(
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
    
    model_persistence.fit(
        X_train, y_per_train,
        eval_set=[(X_train, y_per_train), (X_val, y_per_val)],
        verbose=False
    )
    
    # Evaluate persistence model
    y_per_val_pred = model_persistence.predict(X_val)
    y_per_val_proba = model_persistence.predict_proba(X_val)[:, 1]
    per_val_metrics = evaluate_model(y_per_val, y_per_val_pred, y_per_val_proba, "Persistence Model - Validation")
    
    y_per_test_pred = model_persistence.predict(X_test)
    y_per_test_proba = model_persistence.predict_proba(X_test)[:, 1]
    per_test_metrics = evaluate_model(y_per_test, y_per_test_pred, y_per_test_proba, "Persistence Model - Test")
    
    # Combined model (use persistence as primary, detection as filter)
    print("\n   ✅ Both models trained!")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 MULTI-TASK MODEL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n🔍 Detection Model (Task 1: Anomaly Detection):")
    print(f"   Validation - Accuracy: {det_val_metrics['accuracy']:.4f} | Precision: {det_val_metrics['precision']:.4f} | Recall: {det_val_metrics['recall']:.4f} | F1: {det_val_metrics['f1']:.4f}")
    print(f"   Test        - Accuracy: {det_test_metrics['accuracy']:.4f} | Precision: {det_test_metrics['precision']:.4f} | Recall: {det_test_metrics['recall']:.4f} | F1: {det_test_metrics['f1']:.4f}")
    
    print(f"\n⏱️  Persistence Model (Task 2: Signal Persistence):")
    print(f"   Validation - Accuracy: {per_val_metrics['accuracy']:.4f} | Precision: {per_val_metrics['precision']:.4f} | Recall: {per_val_metrics['recall']:.4f} | F1: {per_val_metrics['f1']:.4f}")
    print(f"   Test        - Accuracy: {per_test_metrics['accuracy']:.4f} | Precision: {per_test_metrics['precision']:.4f} | Recall: {per_test_metrics['recall']:.4f} | F1: {per_test_metrics['f1']:.4f}")
    
    print(f"\n💡 Usage Strategy:")
    print(f"   - Use Detection Model to identify anomalies (high recall)")
    print(f"   - Use Persistence Model to filter for signals that last (high precision)")
    print(f"   - Combined: Enter only if BOTH models agree (detection high AND persistence high)")
    
    # Save models with metadata
    print("\n💾 Saving multi-task models...")
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model_detection': model_detection,      # Detects anomalies
        'model_persistence': model_persistence,   # Predicts persistence
        'feature_cols': feature_cols,
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test),
        'detection_validation_metrics': det_val_metrics,
        'detection_test_metrics': det_test_metrics,
        'persistence_validation_metrics': per_val_metrics,
        'persistence_test_metrics': per_test_metrics,
        'multi_task': True
    }
    
    model_path = 'models/glitchcatcher_model.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"   ✅ Models saved to: {model_path}")
    print(f"   📈 Detection Test Accuracy: {det_test_metrics['accuracy']*100:.2f}%")
    print(f"   📈 Persistence Test Accuracy: {per_test_metrics['accuracy']*100:.2f}%")
    print(f"   📈 Persistence Test F1-Score: {per_test_metrics['f1']:.4f}")
    
    return model_detection, model_persistence, feature_cols, per_test_metrics

def predict_on_data(model_detection, model_persistence, feature_cols, nrows=5000, combined_threshold=0.7):
    """Make predictions using multi-task models."""
    print(f"\n{'='*60}")
    print("🔮 MULTI-TASK PREDICTIONS ON NEW DATA")
    print(f"{'='*60}")
    
    df = pd.read_csv('assets_history_cleaned.csv', nrows=nrows)
    df = df[df['asth_bidPrice'] > 0].copy()
    df = df.sort_values(['asth_symbol', 'changed_time']).reset_index(drop=True)
    
    features = create_features(df)
    features = features[feature_cols + ['asth_symbol']].dropna()
    
    if len(features) == 0:
        print("⚠️  No valid features found")
        return None, None
    
    X = features[feature_cols].values
    
    # Get predictions from both models
    det_predictions = model_detection.predict(X)
    det_probabilities = model_detection.predict_proba(X)[:, 1]
    
    per_predictions = model_persistence.predict(X)
    per_probabilities = model_persistence.predict_proba(X)[:, 1]
    
    # Combined strategy: Both models must agree
    combined_predictions = (det_predictions == 1) & (per_predictions == 1)
    combined_probabilities = (det_probabilities + per_probabilities) / 2  # Average of both
    
    print(f"\n📊 Prediction Results:")
    print(f"   Total samples: {len(X):,}")
    print(f"\n   Detection Model:")
    print(f"      Positive signals: {det_predictions.sum():,} ({det_predictions.mean()*100:.2f}%)")
    print(f"      Avg probability: {det_probabilities.mean():.4f}")
    print(f"\n   Persistence Model:")
    print(f"      Positive signals: {per_predictions.sum():,} ({per_predictions.mean()*100:.2f}%)")
    print(f"      Avg probability: {per_probabilities.mean():.4f}")
    print(f"\n   Combined (Both agree):")
    print(f"      Positive signals: {combined_predictions.sum():,} ({combined_predictions.mean()*100:.2f}%)")
    print(f"      Avg probability: {combined_probabilities.mean():.4f}")
    
    # Top signals (combined)
    top_df = pd.DataFrame({
        'symbol': features['asth_symbol'].values,
        'detection_prob': det_probabilities,
        'persistence_prob': per_probabilities,
        'combined_prob': combined_probabilities,
        'combined_pred': combined_predictions.astype(int)
    }).nlargest(10, 'combined_prob')
    
    print(f"\n🎯 Top 10 Combined Signals (Both Models Agree):")
    print(top_df.to_string(index=False))
    
    return combined_predictions, combined_probabilities

if __name__ == "__main__":
    # Train multi-task models
    model_detection, model_persistence, feature_cols, test_metrics = train_glitchcatcher_model()
    
    # Make predictions with both models
    predict_on_data(model_detection, model_persistence, feature_cols)
    
    print(f"\n{'='*60}")
    print("✅ MULTI-TASK TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n💡 Models saved to: models/glitchcatcher_model.pkl")
    print(f"📊 Persistence Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"📊 Persistence Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"\nTo load models later:")
    print(f"  import joblib")
    print(f"  data = joblib.load('models/glitchcatcher_model.pkl')")
    print(f"  model_detection = data['model_detection']")
    print(f"  model_persistence = data['model_persistence']")
    print(f"  feature_cols = data['feature_cols']")
