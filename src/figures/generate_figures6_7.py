"""
generate_figures6_7.py
======================
Figure 6 — Comparaison des performances (SDI, R², RMSE, MAE) des 6 modèles
Figure 7 — Scatter plots Observé vs Prédit (BPNN et XGBoost)

FICHIERS REQUIS :
  ml_dataset/ml_dataset_v4.csv
  ml_results/model_metrics.csv
  ml_results/xgboost.json   (ou recalcul si absent)

SORTIES :
  figures/Figure6_Model_Comparison.png
  figures/Figure7_Scatter_Obs_Pred.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

COLORS = {
    'BPNN': '#2196F3', 'XGBoost': '#FF5722', 'LightGBM': '#4CAF50',
    'LSTM': '#9C27B0', 'CatBoost': '#FF9800', 'RF': '#607D8B',
}

# ============================================================
# CHARGEMENT DES MÉTRIQUES
# ============================================================
print("Loading metrics and data...")

df = pd.read_csv('ml_dataset/ml_dataset_v4.csv')
df_metrics = pd.read_csv('ml_results/model_metrics.csv')

# Filtrer sur Test seulement
if 'Split' in df_metrics.columns:
    df_metrics_test = df_metrics[df_metrics['Split'] == 'Test'].copy()
else:
    df_metrics_test = df_metrics.copy()

print(f"  Metrics loaded: {len(df_metrics_test)} models")
print(df_metrics_test[['Model','SDI','R2','RMSE','MAE']].to_string(index=False))

# ============================================================
# RÉENTRAÎNER XGBoost ET BPNN POUR SCATTER
# ============================================================
FEATURES = ['SM', 'PRECTOTCORR', 'T2M', 'GWETROOT', 'RH2M', 'EVPTRNS',
            'elevation', 'slope', 'TWI',
            'clay', 'sand', 'PAWC', 'soil_depth',
            'land_use', 'population']
TARGET = 'SSMI'

available = [f for f in FEATURES if f in df.columns]
if 'SM' in available and 'GWETROOT' in available:
    available = [f for f in available if f != 'GWETROOT']
FEATURES = available

train = df[df['year'] <= 2014].dropna(subset=FEATURES + [TARGET])
test  = df[(df['year'] >= 2015) & (df['year'] <= 2020)].dropna(subset=FEATURES + [TARGET])

X_train = train[FEATURES].values
y_train = train[TARGET].values
X_test  = test[FEATURES].values
y_test  = test[TARGET].values

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

print("\nTraining XGBoost and BPNN for scatter plots...")

try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('ml_results/xgboost.json')
    print("  ✅ XGBoost loaded from file")
except:
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("  ✅ XGBoost retrained")

bpnn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    max_iter=300, learning_rate_init=0.001,
    early_stopping=True, validation_fraction=0.1,
    random_state=42, verbose=False)
bpnn.fit(X_tr_sc, y_train)
print("  ✅ BPNN trained")

y_pred_xgb  = xgb_model.predict(X_test)
y_pred_bpnn = bpnn.predict(X_te_sc)

# ============================================================
# FIGURE 6 — COMPARAISON MÉTRIQUES
# ============================================================
print("\nGenerating Figure 6 (Model Comparison)...")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Performance Comparison of Six ML Models\n(Test Period: 2015–2020)',
             fontsize=14, fontweight='bold')

metrics_order = df_metrics_test.sort_values('SDI', ascending=False)
model_names   = metrics_order['Model'].tolist()
colors        = [COLORS.get(m, '#888888') for m in model_names]

for ax, (metric, label) in zip(axes, [
    ('SDI', 'SDI'), ('R2', 'R²'), ('RMSE', 'RMSE'), ('MAE', 'MAE')
]):
    vals = metrics_order[metric].values
    bars = ax.barh(model_names, vals, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.02,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel(label, fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0, max(vals)*1.2)

plt.tight_layout()
path6 = 'figures/Figure6_Model_Comparison.png'
plt.savefig(path6, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {path6}")

# ============================================================
# FIGURE 7 — SCATTER OBSERVÉ VS PRÉDIT
# ============================================================
print("Generating Figure 7 (Scatter Obs vs Pred)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Observed vs. Predicted SSMI (Test Period 2015–2020)',
             fontsize=13, fontweight='bold')

for ax, (name, y_pred, color) in zip(axes, [
    ('BPNN',    y_pred_bpnn, '#2196F3'),
    ('XGBoost', y_pred_xgb,  '#FF5722'),
]):
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    hb = ax.hexbin(y_test, y_pred, gridsize=50, cmap='YlOrRd',
                   mincnt=1, alpha=0.85)
    plt.colorbar(hb, ax=ax, label='Count')

    lims = [-3.5, 3.5]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='1:1 line', zorder=5)
    z = np.polyfit(y_test, y_pred, 1)
    x_line = np.linspace(-3.5, 3.5, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color=color, linewidth=2,
            label='Regression', zorder=6)

    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Observed SSMI', fontsize=11)
    ax.set_ylabel('Predicted SSMI', fontsize=11)
    ax.set_title(f'{name}\nR² = {r2:.3f} | RMSE = {rmse:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(y=-1, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=-2, color='red',    linestyle=':', alpha=0.5, linewidth=1)
    ax.text(3.2, -0.85, 'Mild drought',   ha='right', fontsize=7, color='orange')
    ax.text(3.2, -1.85, 'Severe drought', ha='right', fontsize=7, color='red')

plt.tight_layout()
path7 = 'figures/Figure7_Scatter_Obs_Pred.png'
plt.savefig(path7, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {path7}")

print(f"\n✅ Done!")
print(f"   {path6}")
print(f"   {path7}")