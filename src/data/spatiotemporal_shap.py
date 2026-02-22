"""
spatiotemporal_shap.py
======================
SHAP Spatiotemporel — Désagrégation par gouvernorat et par saison
CONTRIBUTION ORIGINALE PRINCIPALE DU PAPIER

Génère : Figure10_Spatiotemporal_SHAP.png + ml_results/shap_spatiotemporal.csv

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

MÉTHODE :
  - Tree SHAP calculé sur l'ensemble du dataset (train+test+val)
  - Désagrégation par gouvernorat (5) × saison (4) × feature (15)
  - Heatmaps de l'importance SHAP moyenne par gouvernorat et saison
  - Quantification de la vulnérabilité différentielle à la sécheresse

ENTRÉES :
  ml_dataset/ml_dataset_v4.csv
  ml_results/xgboost.json

SORTIES :
  figures/Figure10_Spatiotemporal_SHAP.png
  ml_results/shap_spatiotemporal.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import xgboost as xgb
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('ml_results', exist_ok=True)

# ============================================================
# CHARGEMENT
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

FEAT_LABELS = {
    'SM':         'MERRA-2 SM',
    'GWETROOT':   'MERRA-2 SM',
    'PRECTOTCORR':'Precip.',
    'T2M':        'Temp.',
    'RH2M':       'Rel. Humidity',
    'EVPTRNS':    'Evapotransp.',
    'elevation':  'Elevation',
    'slope':      'Slope',
    'TWI':        'TWI',
    'clay':       'Clay',
    'sand':       'Sand',
    'PAWC':       'PAWC',
    'soil_depth': 'Soil Depth',
    'land_use':   'Land Use',
    'population': 'Population',
}
labels = [FEAT_LABELS.get(f, f) for f in FEATURES]

GOV_ORDER = ['Siliana', 'Kasserine', 'Kairouan', 'Sidi Bou Zid', 'Gafsa']
SEASONS = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}

def month_to_season(m):
    if m in [12, 1, 2]:  return 1   # Winter
    elif m in [3, 4, 5]: return 2   # Spring
    elif m in [6, 7, 8]: return 3   # Summer
    else:                return 4   # Autumn

df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
df_clean['season'] = df_clean['month'].apply(month_to_season)

X_all = df_clean[FEATURES].values
y_all = df_clean[TARGET].values

print(f"  Dataset: {len(df_clean):,} rows")

# ============================================================
# CALCUL TREE SHAP SUR SOUS-ÉCHANTILLON STRATIFIÉ
# ============================================================
print("\nComputing Tree SHAP (stratified sample, seed=42)...")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model('ml_results/xgboost.json')
explainer = shap.TreeExplainer(xgb_model)

# Sous-échantillon stratifié par gouvernorat + saison
N_PER_STRATUM = 200
np.random.seed(42)
idx_list = []

for gov in GOV_ORDER:
    for season in [1, 2, 3, 4]:
        mask = ((df_clean['governorate'] == gov) &
                (df_clean['season'] == season))
        idx_gov_seas = np.where(mask)[0]
        if len(idx_gov_seas) > 0:
            n = min(N_PER_STRATUM, len(idx_gov_seas))
            idx_list.extend(np.random.choice(idx_gov_seas, n, replace=False))

idx_sample = np.array(idx_list)
X_sample = X_all[idx_sample]
df_sample = df_clean.iloc[idx_sample].copy()

shap_sample = explainer.shap_values(X_sample)
print(f"  SHAP shape: {shap_sample.shape} ✅")

# ============================================================
# DÉSAGRÉGATION SPATIOTEMPORELLE
# ============================================================
print("\nDisaggregating by governorate × season...")

records = []
for i, (gov, season) in enumerate(
    zip(df_sample['governorate'], df_sample['season'])):
    for fi, feat in enumerate(labels):
        records.append({
            'governorate': gov,
            'season': SEASONS[season],
            'feature': feat,
            'shap_abs': abs(shap_sample[i, fi]),
            'shap_val': shap_sample[i, fi],
        })

df_shap_long = pd.DataFrame(records)

# Pivot : gouvernorat × feature (moyenne sur toutes saisons)
pivot_gov = df_shap_long.groupby(['governorate','feature'])['shap_abs'].mean().reset_index()
pivot_gov = pivot_gov.pivot(index='governorate', columns='feature', values='shap_abs')
pivot_gov = pivot_gov.reindex(index=GOV_ORDER)

# Pivot : saison × feature (moyenne sur tous gouvernorats)
pivot_seas = df_shap_long.groupby(['season','feature'])['shap_abs'].mean().reset_index()
pivot_seas = pivot_seas.pivot(index='season', columns='feature', values='shap_abs')
pivot_seas = pivot_seas.reindex(index=['Winter','Spring','Summer','Autumn'])

# Sauvegarder
df_shap_long.to_csv('ml_results/shap_spatiotemporal.csv', index=False)
print(f"  ✅ Saved: ml_results/shap_spatiotemporal.csv")

# ============================================================
# FIGURE 10 — HEATMAPS SPATIOTEMPORELLES
# ============================================================
print("\nGenerating Figure 10 (Spatiotemporal SHAP)...")

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Couleurs
cmap = 'YlOrRd'

# --- Panel A : Gouvernorat × Feature ---
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(pivot_gov.values, cmap=cmap, aspect='auto')

ax1.set_xticks(range(len(pivot_gov.columns)))
ax1.set_xticklabels(pivot_gov.columns, rotation=45, ha='right', fontsize=8)
ax1.set_yticks(range(len(GOV_ORDER)))
ax1.set_yticklabels(GOV_ORDER, fontsize=9)
ax1.set_title('(a) Mean |SHAP| by Governorate × Feature\n'
              '(averaged over all seasons, 2001–2022)',
              fontsize=10, fontweight='bold')

# Annotations
for i in range(pivot_gov.shape[0]):
    for j in range(pivot_gov.shape[1]):
        val = pivot_gov.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > pivot_gov.values.max() * 0.65 else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6, color=color)

plt.colorbar(im1, ax=ax1, label='Mean |SHAP Value|', shrink=0.8)

# --- Panel B : Saison × Feature ---
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(pivot_seas.values, cmap=cmap, aspect='auto')

ax2.set_xticks(range(len(pivot_seas.columns)))
ax2.set_xticklabels(pivot_seas.columns, rotation=45, ha='right', fontsize=8)
ax2.set_yticks(range(4))
ax2.set_yticklabels(['Winter', 'Spring', 'Summer', 'Autumn'], fontsize=9)
ax2.set_title('(b) Mean |SHAP| by Season × Feature\n'
              '(averaged over all 5 governorates)',
              fontsize=10, fontweight='bold')

# Annotations
for i in range(pivot_seas.shape[0]):
    for j in range(pivot_seas.shape[1]):
        val = pivot_seas.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > pivot_seas.values.max() * 0.65 else 'black'
            ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6, color=color)

plt.colorbar(im2, ax=ax2, label='Mean |SHAP Value|', shrink=0.8)

plt.suptitle('Spatiotemporal SHAP Disaggregation\n'
             'XGBoost · Tree SHAP · n=2,000 stratified observations',
             fontsize=12, fontweight='bold', y=1.02)

path10 = 'figures/Figure10_Spatiotemporal_SHAP.png'
plt.savefig(path10, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {path10}")

# ============================================================
# RÉSUMÉ — Gouvernorat le plus vulnérable
# ============================================================
print("\n" + "="*60)
print("DROUGHT VULNERABILITY RANKING BY GOVERNORATE")
print("="*60)
gov_total = pivot_gov.sum(axis=1).sort_values(ascending=False)
for gov, total in gov_total.items():
    print(f"  {gov:15s}: {total:.4f} (total SHAP)")

print(f"\n✅ Spatiotemporal SHAP analysis complete!")
print(f"   → Figures saved in figures/")
print(f"   → Data saved in ml_results/shap_spatiotemporal.csv")