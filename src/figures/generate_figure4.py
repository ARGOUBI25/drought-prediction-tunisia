"""
generate_figure4.py
===================
Figure 4 — Flowchart méthodologique de l'étude

Étapes :
  Step 1 : Multi-Source Data Integration (5 sources)
  Step 2 : Dataset Assembly (154,704 obs, 15 features)
           + Temporal Split (Train/Test/Val)
  Step 3 : Six ML Models — Training & Evaluation
           + Métriques (SDI, R², RMSE, MAE)
  Step 4 : Tree SHAP — XAI Interpretation
           (i)  Global Importance
           (ii) SHAP Dependence & Beeswarm
           (iii) Spatiotemporal SHAP

Aucun fichier de données requis — figure 100% vectorielle matplotlib.

SORTIE :
  figures/Figure4_Flowchart.png  (300 dpi)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('figures', exist_ok=True)

OUT = 'figures/Figure4_Flowchart.png'

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Couleurs
C = {
    'data':   '#E3F2FD',   # bleu clair — données
    'data_b': '#1565C0',   # bleu foncé — bordure
    'proc':   '#E8F5E9',   # vert clair — traitement
    'proc_b': '#2E7D32',   # vert foncé
    'model':  '#FFF3E0',   # orange clair — modèles
    'model_b':'#E65100',
    'xai':    '#F3E5F5',   # violet clair — SHAP
    'xai_b':  '#6A1B9A',
    'out':    '#FCE4EC',   # rose clair — résultats
    'out_b':  '#880E4F',
    'arrow':  '#455A64',
}

def box(ax, x, y, w, h, label, sublabel, fc, ec, fs=9, sfs=7.5):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle='round,pad=0.08',
                        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(b)
    ax.text(x, y + 0.12, label,
            ha='center', va='center', fontsize=fs,
            fontweight='bold', color=ec, zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha='center', va='center', fontsize=sfs,
                color='#37474F', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arrow'],
                                lw=2.0, mutation_scale=18),
                zorder=2)

def small_box(ax, x, y, w, h, label, fc, ec, fs=7.5):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle='round,pad=0.06',
                        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fs, color=ec, fontweight='bold', zorder=4)

# Titre
ax.text(7, 9.6, 'Explainable AI Framework for Agricultural Drought Prediction',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color='#212121')

# STEP 1 — DATA SOURCES
ax.text(1.1, 9.0, 'STEP 1', ha='center', fontsize=8,
        fontweight='bold', color=C['data_b'])

sources = [
    (1.3, 8.2, 'GLEAM v4.2a\n0.05° · 1980–2022\nSM → SSMI (target)'),
    (3.5, 8.2, 'NASA POWER\nMERRA-2 0.5°\nSM + 5 met. vars'),
    (5.7, 8.2, 'SoilGrids v2.0\n250 m\nclay·sand·PAWC'),
    (7.9, 8.2, 'SRTM DEM\n90 m\nelev·slope·TWI'),
    (10.1,8.2, 'ESA WorldCover\n+ WorldPop\nlanduse·pop'),
]
for x, y, lbl in sources:
    b = FancyBboxPatch((x-0.9, y-0.52), 1.8, 1.04,
                        boxstyle='round,pad=0.06',
                        facecolor=C['data'], edgecolor=C['data_b'],
                        linewidth=1.4, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, lbl, ha='center', va='center', fontsize=7.2,
            color='#1A237E', zorder=4)

for x, y, _ in sources:
    arrow(ax, x, y-0.52, x, 7.05)

# STEP 2 — DATASET ASSEMBLY
ax.text(1.1, 7.3, 'STEP 2', ha='center', fontsize=8,
        fontweight='bold', color=C['proc_b'])

box(ax, 6.7, 6.6, 9.0, 0.85,
    'Multi-Source Data Integration  —  154,704 pixel-month observations · 586 pixels · 2001–2022',
    '15 predictors: SM · PRECTOTCORR · T2M · GWETROOT · RH2M · EVPTRNS · '
    'elevation · slope · TWI · clay · sand · PAWC · soil depth · land use · pop',
    C['proc'], C['proc_b'], fs=8.5, sfs=7)

arrow(ax, 6.7, 6.17, 6.7, 5.62)

box(ax, 6.7, 5.25, 9.0, 0.70,
    'Temporal Split (no data leakage)',
    '', C['proc'], C['proc_b'], fs=8.5)

small_box(ax, 3.8, 5.25, 2.4, 0.42,
          'Training: 2001–2014\n98,448 obs (63.6%)',
          '#E8F5E9', C['proc_b'], fs=7.2)
small_box(ax, 6.7, 5.25, 2.4, 0.42,
          'Test: 2015–2020\n42,192 obs (27.3%)',
          '#E8F5E9', C['proc_b'], fs=7.2)
small_box(ax, 9.6, 5.25, 2.4, 0.42,
          'Validation: 2021–2022\n14,064 obs (9.1%)',
          '#E8F5E9', C['proc_b'], fs=7.2)

# STEP 3 — MODELS
ax.text(1.1, 4.8, 'STEP 3', ha='center', fontsize=8,
        fontweight='bold', color=C['model_b'])

arrow(ax, 6.7, 4.89, 6.7, 4.38)

box(ax, 6.7, 4.0, 9.0, 0.70,
    'ML Model Benchmarking — Six Models — Training & Evaluation',
    '', C['model'], C['model_b'], fs=8.5)

models = [
    (2.8, 4.0, 'XGBoost'),
    (4.5, 4.0, 'LightGBM'),
    (6.1, 4.0, 'CatBoost'),
    (7.7, 4.0, 'RF'),
    (9.3, 4.0, 'BPNN'),
    (10.9,4.0, 'LSTM'),
]
for x, y, lbl in models:
    small_box(ax, x, y, 1.3, 0.38, lbl, '#FFF3E0', C['model_b'], fs=7.5)

arrow(ax, 6.7, 3.64, 6.7, 3.18)

box(ax, 6.7, 2.85, 9.0, 0.60,
    'Evaluation Metrics: SDI · R² · RMSE · MAE',
    'Test set (2015–2020)  +  Independent validation set (2021–2022)',
    C['model'], C['model_b'], fs=8.5, sfs=7.2)

# STEP 4 — SHAP
ax.text(1.1, 2.3, 'STEP 4', ha='center', fontsize=8,
        fontweight='bold', color=C['xai_b'])

arrow(ax, 6.7, 2.54, 6.7, 2.05)

box(ax, 6.7, 1.67, 9.0, 0.70,
    'XAI Interpretation — Tree SHAP (XGBoost, n = 2,000 obs, seed = 42)',
    '', C['xai'], C['xai_b'], fs=8.5)

shap_boxes = [
    (3.5,  1.67, '(i) Global\nImportance\nmean |φᵢ|'),
    (6.7,  1.67, '(ii) SHAP Dependence\n& Beeswarm\n(Top 10 features)'),
    (9.9,  1.67, '(iii) Spatiotemporal\nSHAP by gov.\n& season'),
]
for x, y, lbl in shap_boxes:
    small_box(ax, x, y, 2.2, 0.52, lbl, '#F3E5F5', C['xai_b'], fs=7.2)

# OUTPUT
arrow(ax, 6.7, 1.30, 6.7, 0.88)

box(ax, 6.7, 0.57, 9.0, 0.56,
    '🏁  Drought prediction maps · SHAP importance · Governorate vulnerability ranking',
    '', C['out'], C['out_b'], fs=8, sfs=7)

# Légende
legend_items = [
    mpatches.Patch(facecolor=C['data'],  edgecolor=C['data_b'],  label='Data sources'),
    mpatches.Patch(facecolor=C['proc'],  edgecolor=C['proc_b'],  label='Preprocessing'),
    mpatches.Patch(facecolor=C['model'], edgecolor=C['model_b'], label='ML models'),
    mpatches.Patch(facecolor=C['xai'],   edgecolor=C['xai_b'],   label='XAI (SHAP)'),
    mpatches.Patch(facecolor=C['out'],   edgecolor=C['out_b'],   label='Outputs'),
]
ax.legend(handles=legend_items, loc='lower left',
          fontsize=7.5, framealpha=0.9,
          bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.5)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Saved → {OUT}")