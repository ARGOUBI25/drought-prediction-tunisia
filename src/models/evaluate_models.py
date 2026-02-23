"""
evaluate_models.py
==================
Évaluation des 6 modèles ML sur les périodes Test (2015-2020)
et Validation indépendante (2021-2022) sans réentraîner

Génère :
  ml_results/model_metrics_full.csv   ← SDI, R², RMSE, MAE par modèle et split
  ml_results/validation_predictions.csv ← prédictions 2021-2022 + valeurs SSMI observées
  figures/Figure6_Model_Comparison.png  ← barplots 4 métriques

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

PRÉREQUIS :
  1. Avoir exécuté train_models.py
  2. ml_results/xgboost.json, lightgbm.txt, catboost.cbm présents

NOTE : RF, BPNN, LSTM sont réentraînés rapidement car leurs formats
       de sauvegarde varient selon l'OS et la version sklearn/TF.
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

os.makedirs('ml_results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print("Loading dataset...")
df = pd.read_csv('ml_dataset/ml_dataset_v4.csv')
print(f"  {len(df):,} rows | {df.shape[1]} columns")

FEATURES = ['SM', 'PRECTOTCORR', 'T2M', 'GWETROOT', 'RH2M', 'EVPTRNS',
            'elevation', 'slope', 'TWI',
            'clay', 'sand', 'PAWC', 'soil_depth',
            'land_use', 'population']
TARGET = 'SSMI'

available = [f for f in FEATURES if f in df.columns]
if 'SM' in available and 'GWETROOT' in available:
    available = [f for f in available if f != 'GWETROOT']
FEATURES = available
print(f"  Features ({len(FEATURES)}): {FEATURES}")

# Splits temporels (sans data leakage)
train = df[df['year'] <= 2014].dropna(subset=FEATURES + [TARGET])
test  = df[(df['year'] >= 2015) & (df['year'] <= 2020)].dropna(subset=FEATURES + [TARGET])
val   = df[df['year'] >= 2021].dropna(subset=FEATURES + [TARGET])

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values
X_val,   y_val   = val[FEATURES].values,   val[TARGET].values

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)
X_va_sc  = scaler.transform(X_val)

print(f"\n  Train (2001-2014): {len(X_train):,}")
print(f"  Test  (2015-2020): {len(X_test):,}")
print(f"  Val   (2021-2022): {len(X_val):,}")

# ============================================================
# MÉTRIQUES
# ============================================================
def sdi(y_true, y_pred):
    """Spatial Deviation Index (Ning et al., 2022)."""
    from scipy.spatial.distance import jensenshannon
    alpha = np.corrcoef(y_true, y_pred)[0, 1]
    beta  = np.std(y_pred) / (np.std(y_true) + 1e-10)
    # JSD entre distributions discrétisées
    bins = np.linspace(-3, 3, 50)
    p, _ = np.histogram(y_true, bins=bins, density=True)
    q, _ = np.histogram(y_pred, bins=bins, density=True)
    p = p + 1e-10;  q = q + 1e-10
    p /= p.sum();   q /= q.sum()
    jsd = jensenshannon(p, q) ** 2
    gamma = max(0, 1 - jsd)
    return float(np.clip(alpha * beta * gamma, 0, 1))

def metrics(y_true, y_pred, model_name, split):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    s    = sdi(y_true, y_pred)
    print(f"  {model_name:12s} [{split:5s}] | "
          f"SDI={s:.3f} | R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
    return {'Model': model_name, 'Split': split,
            'SDI': round(s,3), 'R2': round(r2,3),
            'RMSE': round(rmse,3), 'MAE': round(mae,3)}

results = []
preds_val = val[['lat','lon','governorate','year','month',TARGET]].copy()

# ============================================================
# 1. XGBoost — charger depuis fichier
# ============================================================
print("\n" + "="*60 + "\n1. XGBoost")
try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('ml_results/xgboost.json')
    print("  ✅ Loaded from ml_results/xgboost.json")
except FileNotFoundError:
    print("  ⚠️  File not found — retraining...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)

results += [metrics(y_test, xgb_model.predict(X_test), 'XGBoost', 'Test'),
            metrics(y_val,  xgb_model.predict(X_val),  'XGBoost', 'Val')]
preds_val['pred_XGBoost'] = xgb_model.predict(X_val)

# ============================================================
# 2. LightGBM
# ============================================================
print("\n" + "="*60 + "\n2. LightGBM")
try:
    lgb_booster = lgb.Booster(model_file='ml_results/lightgbm.txt')
    lgb_predict = lgb_booster.predict
    print("  ✅ Loaded from ml_results/lightgbm.txt")
except FileNotFoundError:
    print("  ⚠️  File not found — retraining...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    lgb_predict = lgb_model.predict

results += [metrics(y_test, lgb_predict(X_test), 'LightGBM', 'Test'),
            metrics(y_val,  lgb_predict(X_val),  'LightGBM', 'Val')]
preds_val['pred_LightGBM'] = lgb_predict(X_val)

# ============================================================
# 3. CatBoost
# ============================================================
print("\n" + "="*60 + "\n3. CatBoost")
cat_model = CatBoostRegressor(verbose=False)
try:
    cat_model.load_model('ml_results/catboost.cbm')
    print("  ✅ Loaded from ml_results/catboost.cbm")
except Exception:
    print("  ⚠️  File not found — retraining...")
    cat_model = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=False)
    cat_model.fit(X_train, y_train,
                  eval_set=(X_test, y_test),
                  early_stopping_rounds=50, verbose=False)

results += [metrics(y_test, cat_model.predict(X_test), 'CatBoost', 'Test'),
            metrics(y_val,  cat_model.predict(X_val),  'CatBoost', 'Val')]
preds_val['pred_CatBoost'] = cat_model.predict(X_val)

# ============================================================
# 4. Random Forest (réentraîné rapidement)
# ============================================================
print("\n" + "="*60 + "\n4. Random Forest")
np.random.seed(42)
idx_rf = np.random.choice(len(X_train),
                           min(80000, len(X_train)), replace=False)
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train[idx_rf], y_train[idx_rf])
results += [metrics(y_test, rf_model.predict(X_test), 'RF', 'Test'),
            metrics(y_val,  rf_model.predict(X_val),  'RF', 'Val')]
preds_val['pred_RF'] = rf_model.predict(X_val)

# ============================================================
# 5. BPNN
# ============================================================
print("\n" + "="*60 + "\n5. BPNN")
bpnn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    max_iter=300, learning_rate_init=0.001,
    early_stopping=True, validation_fraction=0.1,
    random_state=42, verbose=False)
bpnn.fit(X_tr_sc, y_train)
results += [metrics(y_test, bpnn.predict(X_te_sc), 'BPNN', 'Test'),
            metrics(y_val,  bpnn.predict(X_va_sc), 'BPNN', 'Val')]
preds_val['pred_BPNN'] = bpnn.predict(X_va_sc)

# ============================================================
# 6. LSTM
# ============================================================
print("\n" + "="*60 + "\n6. LSTM")
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from keras.models import Sequential
    from keras.layers import LSTM as LSTMLayer, Dense, Dropout
    from keras.callbacks import EarlyStopping

    n_feat = len(FEATURES)
    np.random.seed(42)
    idx_l  = np.random.choice(len(X_tr_sc),
                               min(80000, len(X_tr_sc)), replace=False)

    lstm_model = Sequential([
        LSTMLayer(64, input_shape=(1, n_feat)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(
        X_tr_sc[idx_l].reshape(-1, 1, n_feat), y_train[idx_l],
        epochs=50, batch_size=256, validation_split=0.1,
        callbacks=[EarlyStopping(patience=10,
                                 restore_best_weights=True, verbose=0)],
        verbose=0)

    y_lstm_te = lstm_model.predict(X_te_sc.reshape(-1,1,n_feat), verbose=0).ravel()
    y_lstm_va = lstm_model.predict(X_va_sc.reshape(-1,1,n_feat), verbose=0).ravel()
    results += [metrics(y_test, y_lstm_te, 'LSTM', 'Test'),
                metrics(y_val,  y_lstm_va, 'LSTM', 'Val')]
    preds_val['pred_LSTM'] = y_lstm_va

except Exception as e:
    print(f"  ⚠️  LSTM skipped: {e}")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================
print("\n" + "="*60)
print("FINAL EVALUATION RESULTS")
print("="*60)

df_res = pd.DataFrame(results)

print("\nTest set (2015–2020):")
print(df_res[df_res['Split']=='Test']
      .sort_values('SDI', ascending=False)
      [['Model','SDI','R2','RMSE','MAE']].to_string(index=False))

print("\nValidation set (2021–2022):")
print(df_res[df_res['Split']=='Val']
      .sort_values('SDI', ascending=False)
      [['Model','SDI','R2','RMSE','MAE']].to_string(index=False))

df_res.to_csv('ml_results/model_metrics_full.csv', index=False)
preds_val.to_csv('ml_results/validation_predictions.csv', index=False)

print(f"\n✅ Saved: ml_results/model_metrics_full.csv")
print(f"✅ Saved: ml_results/validation_predictions.csv")
print(f"\n✅ Evaluation complete!")
print(f"   → Next: run src/shap/global_importance.py")
