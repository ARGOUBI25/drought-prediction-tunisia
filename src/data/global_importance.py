"""
global_importance.py
====================
Tree SHAP — Importance globale des features + Beeswarm plot (Top 10)
Génère : Figure8_SHAP_Summary.png + ml_results/shap_values.csv

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

ENTRÉES :
  ml_dataset/ml_dataset_v4.csv
  ml_results/xgboost.json

SORTIES :
  figures/Figure8_SHAP_Summary.png
  ml_results/shap_values.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('ml_results', exist_ok=True)

# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================
print("Loading data and model...")

df = pd.read_csv('ml_dataset/ml_dataset_v4.csv')

FEATURES = ['SM', 'PRECTOTCORR', 'T2M', 'GWETROOT', 'RH2M', 'EVPTRNS',
            'elevation', 'slope', 'TWI',
            'clay', 'sand', 'PAWC', 'soil_depth',
            'land_use', 'population']
TARGET = 'SSMI'

available = [f for f in FEATURES if f in df.columns]
if 'SM' in available and 'GWETROOT' in available:
    available = [f for f in available if f != 'GWETROOT']
FEATURES = available

test = df[(df['year'] >= 2015) & (df['year'] <= 2020)].dropna(subset=FEATURES + [TARGET])
X_test = test[FEATURES].values
y_test = test[TARGET].values

# Charger modèle XGBoost entraîné
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('ml_results/xgboost.json')
print(f"  ✅ XGBoost loaded")

# Labels lisibles pour les features
FEAT_LABELS = {
    'SM':         'MERRA-2 SM (SM)',
    'GWETROOT':   'MERRA-2 SM (SM)',
    'PRECTOTCORR':'Precipitation',
    'T2M':        'Temperature (T2M)',
    'RH2M':       'Relative Humidity',
    'EVPTRNS':    'Evapotranspiration',
    'elevation':  'Elevation',
    'slope':      'Slope',
    'TWI':        'TWI',
    'clay':       'Clay Content',
    'sand':       'Sand Content',
    'PAWC':       'PAWC',
    'soil_depth': 'Soil Depth',
    'land_use':   'Land Use',
    'population': 'Population Density',
}
labels = [FEAT_LABELS.get(f, f) for f in FEATURES]

# ============================================================
# CALCUL TREE SHAP
# n=2000, seed=42 (reproductible)
# ============================================================
print("\nComputing Tree SHAP values (n=2000, seed=42)...")
N_SHAP = 2000
np.random.seed(42)
idx_shap  = np.random.choice(len(X_test), min(N_SHAP, len(X_test)), replace=False)
X_shap    = X_test[idx_shap]
y_shap    = y_test[idx_shap]

explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(X_shap)   # (N_SHAP, n_feat)

n_samples, n_feats = shap_vals.shape
print(f"  SHAP shape: {shap_vals.shape} ✅")

# ============================================================
# IMPORTANCE GLOBALE
# ============================================================
mean_shap = np.abs(shap_vals).mean(axis=0)   # (n_feat,)
shap_df = pd.DataFrame({
    'feature':    labels,
    'feat_idx':   list(range(len(FEATURES))),
    'importance': mean_shap,
    'importance_pct': mean_shap / mean_shap.sum() * 100
}).sort_values('importance', ascending=True)

print("\nGlobal SHAP importance:")
for _, row in shap_df.sort_values('importance', ascending=False).iterrows():
    print(f"  {row['feature']:28s}: {row['importance']:.4f}  ({row['importance_pct']:.1f}%)")

# ============================================================
# FIGURE 8 — BAR CHART + BEESWARM
# ============================================================
print("\nGenerating Figure 8 (SHAP Summary)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('SHAP Feature Importance Analysis (XGBoost — Tree SHAP)',
             fontsize=13, fontweight='bold')

# --- Panel A : Bar chart global importance ---
ax = axes[0]
n = len(shap_df)
bar_colors = ['#D32F2F' if i >= n-5 else '#1976D2' if i >= n-10
              else '#78909C' for i in range(n)]
bars = ax.barh(shap_df['feature'], shap_df['importance'],
               color=bar_colors, edgecolor='white', height=0.7)
for bar, val in zip(bars, shap_df['importance']):
    ax.text(bar.get_width() + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax.set_title('(a) Global Feature Importance', fontsize=11, fontweight='bold')
ax.spines[['top','right']].set_visible(False)

# --- Panel B : Beeswarm (Top 10) ---
ax2 = axes[1]
top_n    = 10
top_rows  = shap_df.nlargest(top_n, 'importance')
top_feats = top_rows['feature'].tolist()
top_fidxs = top_rows['feat_idx'].tolist()

sc = None
for plot_i, (fi, fname) in enumerate(zip(top_fidxs[::-1], top_feats[::-1])):
    sv     = shap_vals[:, fi]
    fv     = X_shap[:, fi]
    norm_c = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
    jitter = np.random.uniform(-0.2, 0.2, n_samples)
    sc = ax2.scatter(sv, np.full(n_samples, plot_i) + jitter,
                     c=norm_c, cmap='coolwarm', alpha=0.3, s=8,
                     vmin=0, vmax=1)

ax2.set_yticks(range(top_n))
ax2.set_yticklabels(top_feats[::-1], fontsize=9)
ax2.axvline(x=0, color='black', linewidth=1, linestyle='--')
ax2.set_xlabel('SHAP Value (impact on model output)', fontsize=11)
ax2.set_title('(b) SHAP Beeswarm Plot (Top 10 Features)',
              fontsize=11, fontweight='bold')
ax2.spines[['top','right']].set_visible(False)
if sc is not None:
    plt.colorbar(sc, ax=ax2, label='Feature value\n(normalized)', shrink=0.6)

plt.tight_layout()
path8 = 'figures/Figure8_SHAP_Summary.png'
plt.savefig(path8, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {path8}")

# ============================================================
# SAUVEGARDE SHAP VALUES
# ============================================================
df_shap = pd.DataFrame(shap_vals, columns=labels)
df_shap['y_true'] = y_shap
df_shap.to_csv('ml_results/shap_values.csv', index=False)
print(f"  ✅ Saved: ml_results/shap_values.csv")
shap_df.sort_values('importance', ascending=False).to_csv(
    'ml_results/shap_importance.csv', index=False)
print(f"  ✅ Saved: ml_results/shap_importance.csv")

print(f"\n✅ Done! → Next: run shap/dependence_plots.py")