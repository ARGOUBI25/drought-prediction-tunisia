"""
Script : Entraînement des 6 modèles ML — VERSION FINALE v3
Dataset : ml_dataset_v4.csv (15 features incluant SM GLEAM 0.05°)
Article : Drought prediction in semi-arid Tunisia
Author  : Majdi Argoubi, University of Sousse
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

OUTPUT_DIR = 'ml_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CHARGEMENT
# ============================================================
print("Loading dataset v4...")
df = pd.read_csv('ml_dataset/ml_dataset_v4.csv')
print(f"  Rows: {len(df):,} | Columns: {df.shape[1]}")

FEATURES = ['SM', 'PRECTOTCORR', 'T2M', 'GWETROOT', 'RH2M', 'EVPTRNS',
            'elevation', 'slope', 'TWI',
            'clay', 'sand', 'PAWC', 'soil_depth',
            'land_use', 'population']
TARGET = 'SSMI'

available = [f for f in FEATURES if f in df.columns]
missing   = [f for f in FEATURES if f not in df.columns]
print(f"  Features ({len(available)}): {available}")
if missing:
    print(f"  Missing: {missing}")
FEATURES = available

# ============================================================
# SPLIT TEMPOREL
# ============================================================
print("\nSplitting dataset...")
train = df[df['year'] <= 2014].dropna(subset=FEATURES + [TARGET])
test  = df[(df['year'] >= 2015) & (df['year'] <= 2020)].dropna(subset=FEATURES + [TARGET])
val   = df[df['year'] >= 2021].dropna(subset=FEATURES + [TARGET])

X_train = train[FEATURES].values
y_train = train[TARGET].values
X_test  = test[FEATURES].values
y_test  = test[TARGET].values
X_val   = val[FEATURES].values
y_val   = val[TARGET].values

print(f"  Train (2001-2014): {len(X_train):,}")
print(f"  Test  (2015-2020): {len(X_test):,}")
print(f"  Val   (2021-2022): {len(X_val):,}")

# ============================================================
# MÉTRIQUES
# ============================================================
def sdi(y_true, y_pred):
    mse = np.mean((y_pred - y_true) ** 2)
    var = np.var(y_true)
    return float(1 - np.sqrt(mse / (var + 1e-10)))

def metrics(y_true, y_pred, name):
    s    = sdi(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  {name:12s} | SDI={s:.3f} | R²={r2:.3f} | "
          f"RMSE={rmse:.3f} | MAE={mae:.3f}")
    return {'Model': name, 'SDI': round(s,3), 'R2': round(r2,3),
            'RMSE': round(rmse,3), 'MAE': round(mae,3)}

results = []

# ============================================================
# 1. XGBoost
# ============================================================
print("\n" + "="*55)
print("1. XGBoost")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
results.append(metrics(y_test, xgb_model.predict(X_test), "XGBoost"))
xgb_model.save_model(os.path.join(OUTPUT_DIR, 'xgboost.json'))
print("  ✅ Saved")

# ============================================================
# 2. LightGBM
# ============================================================
print("\n" + "="*55)
print("2. LightGBM")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
results.append(metrics(y_test, lgb_model.predict(X_test), "LightGBM"))
lgb_model.booster_.save_model(os.path.join(OUTPUT_DIR, 'lightgbm.txt'))
print("  ✅ Saved")

# ============================================================
# 3. CatBoost
# ============================================================
print("\n" + "="*55)
print("3. CatBoost")
cat_model = CatBoostRegressor(
    iterations=500, depth=6, learning_rate=0.05,
    l2_leaf_reg=3, random_seed=42, verbose=False)
cat_model.fit(X_train, y_train,
              eval_set=(X_test, y_test),
              early_stopping_rounds=50, verbose=False)
results.append(metrics(y_test, cat_model.predict(X_test), "CatBoost"))
cat_model.save_model(os.path.join(OUTPUT_DIR, 'catboost.cbm'))
print("  ✅ Saved")

# ============================================================
# 4. Random Forest
# ============================================================
print("\n" + "="*55)
print("4. Random Forest")
np.random.seed(42)
idx = np.random.choice(len(X_train), min(80000, len(X_train)), replace=False)
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train[idx], y_train[idx])
results.append(metrics(y_test, rf_model.predict(X_test), "RF"))
print("  ✅ Done")

# ============================================================
# 5. BPNN
# ============================================================
print("\n" + "="*55)
print("5. BPNN")
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
X_va_sc = scaler.transform(X_val)

bpnn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    max_iter=300, learning_rate_init=0.001,
    early_stopping=True, validation_fraction=0.1,
    random_state=42, verbose=False)
bpnn.fit(X_tr_sc, y_train)
results.append(metrics(y_test, bpnn.predict(X_te_sc), "BPNN"))
print("  ✅ Done")

# ============================================================
# 6. LSTM
# ============================================================
print("\n" + "="*55)
print("6. LSTM")
HAVE_LSTM = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping

    np.random.seed(42)
    n_feat = len(FEATURES)
    idx_l  = np.random.choice(len(X_tr_sc),
                               min(80000, len(X_tr_sc)), replace=False)
    X_tl = X_tr_sc[idx_l].reshape(-1, 1, n_feat)
    X_el = X_te_sc.reshape(-1, 1, n_feat)
    X_vl = X_va_sc.reshape(-1, 1, n_feat)

    lstm_model = Sequential([
        LSTM(64, input_shape=(1, n_feat)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(
        X_tl, y_train[idx_l],
        epochs=50, batch_size=256,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10,
                                 restore_best_weights=True,
                                 verbose=0)],
        verbose=0)
    y_pred_lstm = lstm_model.predict(X_el, verbose=0).ravel()
    results.append(metrics(y_test, y_pred_lstm, "LSTM"))
    lstm_model.save(os.path.join(OUTPUT_DIR, 'lstm.keras'))
    print("  ✅ Saved")
    HAVE_LSTM = True
except Exception as e:
    print(f"  ⚠️  LSTM skipped: {e}")

# ============================================================
# TABLEAU FINAL
# ============================================================
print("\n" + "="*55)
print("FINAL RESULTS — Test set 2015-2020")
print("="*55)
df_res = pd.DataFrame(results).sort_values('SDI', ascending=False)
print(df_res.to_string(index=False))
df_res.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'), index=False)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*55)
print("FEATURE IMPORTANCE (XGBoost)")
print("="*55)
fi = pd.DataFrame({
    'feature':    FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(fi.to_string(index=False))
fi.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_xgb.csv'), index=False)

# ============================================================
# VALIDATION 2021-2022
# ============================================================
print("\n" + "="*55)
print("VALIDATION 2021-2022")
print("="*55)
df_val_out = val[['lat','lon','governorate','year','month',TARGET]].copy()
df_val_out['pred_XGBoost']  = xgb_model.predict(X_val)
df_val_out['pred_LightGBM'] = lgb_model.predict(X_val)
df_val_out['pred_CatBoost'] = cat_model.predict(X_val)
df_val_out['pred_RF']       = rf_model.predict(X_val)
df_val_out['pred_BPNN']     = bpnn.predict(X_va_sc)
if HAVE_LSTM:
    df_val_out['pred_LSTM'] = lstm_model.predict(X_vl, verbose=0).ravel()

print("Validation metrics:")
model_list = ['XGBoost','LightGBM','CatBoost','RF','BPNN']
if HAVE_LSTM:
    model_list.append('LSTM')
for m in model_list:
    col = f'pred_{m}'
    if col in df_val_out.columns:
        metrics(y_val, df_val_out[col].values, m)

df_val_out.to_csv(
    os.path.join(OUTPUT_DIR, 'validation_predictions.csv'), index=False)

print(f"\n✅ model_metrics.csv")
print(f"✅ validation_predictions.csv")
print(f"✅ feature_importance_xgb.csv")
print(f"\n✅ Training complete! → Next: SHAP analysis")