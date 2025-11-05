"""
Hull Tactical Market Prediction - Kaggle Submission
Enhanced v1 with LightGBM + XGBoost Ensemble
Generated on 2025-11-05
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle

import kaggle_evaluation.default_inference_server

# Load models and metadata
lgb_model = lgb.Booster(model_file='models/lgb_final.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgb_final.json')

with open('models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

feature_list = metadata['feature_list']

# Feature engineering function
def create_features_for_prediction(test_data, historical_data=None):
    """
    Create features for prediction (must match training features)

    Args:
        test_data: Current test data
        historical_data: Historical data for lag/rolling features
    """
    # Combine with historical data for lag/rolling calculations
    if historical_data is not None:
        combined = pd.concat([historical_data, test_data], ignore_index=False)
    else:
        combined = test_data.copy()

    df = combined.copy()

    # Original features
    feature_cols = [c for c in df.columns if c not in ['date_id', 'forward_returns', 'risk_free_rate']]

    # Top features for lag/rolling (from training)
    top_features = ['P8', 'P10', 'S5', 'E3', 'M3', 'V13', 'P11', 'E2', 'P12', 'M1', 'S7', 'E5', 'M9', 'V7', 'I5']

    # Lag features
    for feat in top_features:
        if feat in df.columns:
            for lag in [1, 5, 10, 20, 60]:
                df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

    # Rolling statistics
    for feat in top_features:
        if feat in df.columns:
            for window in [5, 20, 60]:
                df[f'{feat}_mean{window}'] = df[feat].rolling(window).mean()
                df[f'{feat}_std{window}'] = df[feat].rolling(window).std()

    # Returns-based features (using lagged_forward_returns if available)
    if 'lagged_forward_returns' in df.columns:
        returns_col = 'lagged_forward_returns'

        df['returns_lag1'] = df[returns_col].shift(1)
        df['returns_lag5'] = df[returns_col].shift(5)

        df['momentum_5'] = df[returns_col].rolling(5).sum()
        df['momentum_20'] = df[returns_col].rolling(20).sum()
        df['momentum_60'] = df[returns_col].rolling(60).sum()

        df['returns_mean_20'] = df[returns_col].rolling(20).mean()
        df['returns_std_20'] = df[returns_col].rolling(20).std()
        df['returns_mean_60'] = df[returns_col].rolling(60).mean()
        df['returns_std_60'] = df[returns_col].rolling(60).std()

        df['volatility_5'] = df[returns_col].rolling(5).std()
        df['volatility_20'] = df[returns_col].rolling(20).std()
        df['volatility_60'] = df[returns_col].rolling(60).std()

        df['vol_of_vol_20'] = df['volatility_20'].rolling(20).std()

        vol_percentile = df['volatility_20'].rolling(252).rank(pct=True)
        df['vol_regime_low'] = (vol_percentile < 0.33).astype(int)
        df['vol_regime_high'] = (vol_percentile > 0.67).astype(int)

        df['bull_regime'] = (df['returns_mean_60'] > 0).astype(int)

        df['risk_adj_momentum_20'] = df['momentum_20'] / (df['volatility_20'] + 1e-8)
        df['risk_adj_momentum_60'] = df['momentum_60'] / (df['volatility_60'] + 1e-8)

    # Cross-sectional features
    feature_categories = {
        'M': [c for c in feature_cols if c.startswith('M')],
        'E': [c for c in feature_cols if c.startswith('E')],
        'I': [c for c in feature_cols if c.startswith('I')],
        'P': [c for c in feature_cols if c.startswith('P')],
        'V': [c for c in feature_cols if c.startswith('V')],
        'S': [c for c in feature_cols if c.startswith('S')],
    }

    for cat, feats in feature_categories.items():
        if feats:
            df[f'{cat}_mean'] = df[feats].mean(axis=1)
            df[f'{cat}_std'] = df[feats].std(axis=1)
            df[f'{cat}_min'] = df[feats].min(axis=1)
            df[f'{cat}_max'] = df[feats].max(axis=1)

    # Feature interactions
    if len(top_features) >= 2:
        for i in range(min(5, len(top_features))):
            for j in range(i+1, min(5, len(top_features))):
                feat1, feat2 = top_features[i], top_features[j]
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

    # Missing value indicator
    df['n_missing_orig'] = df[feature_cols].isnull().sum(axis=1)

    # Return only the test data rows
    return df.loc[test_data.index]

# Allocation function
def predict_allocation(predicted_returns, base_allocation=1.0, sensitivity=100):
    """
    Convert predicted returns to allocation (0-2)
    """
    allocation = base_allocation + np.tanh(predicted_returns * sensitivity)
    return np.clip(allocation, 0, 2)

# Global variables for maintaining history
historical_data = None
first_call = True

def predict(test: pd.DataFrame) -> pd.DataFrame:
    """
    Main prediction function called by Kaggle evaluation API

    Args:
        test: Test data for current batch

    Returns:
        DataFrame with date_id and prediction columns
    """
    global historical_data, first_call

    # On first call, load training data for historical features
    if first_call:
        train_data = pd.read_csv('data/train.csv')
        # Keep last 300 days for lag/rolling features
        historical_data = train_data.tail(300).copy()
        first_call = False

    # Create features
    test_with_features = create_features_for_prediction(test, historical_data)

    # Fill missing features with 0 (or use more sophisticated imputation)
    for feat in feature_list:
        if feat not in test_with_features.columns:
            test_with_features[feat] = 0

    test_with_features = test_with_features[feature_list].fillna(0)

    # Predict with ensemble
    pred_lgb = lgb_model.predict(test_with_features)
    pred_xgb = xgb_model.predict(xgb.DMatrix(test_with_features))
    pred_ensemble = 0.5 * pred_lgb + 0.5 * pred_xgb

    # Convert to allocation
    allocations = predict_allocation(pred_ensemble, sensitivity=100)

    # Update historical data (keep last 300 days)
    historical_data = pd.concat([historical_data, test], ignore_index=False).tail(300)

    # Return predictions
    result = pd.DataFrame({
        'date_id': test['date_id'],
        'prediction': allocations
    })

    return result

# Initialize inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # Local testing
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
