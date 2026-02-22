"""
generate_figure5.py
===================
Figure 5 — Distribution spatiotemporelle du SSMI
           (5 gouvernorats semi-arides de Tunisie centrale, 2001–2022)

Subplots :
  (a) Série temporelle mensuelle SSMI par gouvernorat (2001–2022)
       avec bandes de couleur pour les catégories de sécheresse
  (b1–b4) Cartes spatiales SSMI moyen annuel :
       2002 (Dry), 2003 (Wet), 2016 (Near-normal), 2021 (Dry)

FICHIERS REQUIS :
  ssmi_output/ssmi_monthly_by_gov.csv
  ssmi_output/ssmi_annual_by_gov.csv
  ssmi_output/ssmi_tunisia_2001_2022.csv

SORTIE :
  figures/Figure5_SSMI_Timeseries.png  (300 dpi)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

# Fichiers d'entrée
MONTHLY = 'ssmi_output/ssmi_monthly_by_gov.csv'
ANNUAL  = 'ssmi_output/ssmi_annual_by_gov.csv'
FULL    = 'ssmi_output/ssmi_tunisia_2001_2022.csv'
OUT     = 'figures/Figure5_SSMI_Timeseries.png'

# Couleurs
DROUGHT_BANDS = [
    (-3.0, -2.0, '#7B0000', 0.18, 'Extreme drought'),
    (-2.0, -1.5, '#D32F2F', 0.18, 'Severe drought'),
    (-1.5, -1.0, '#FF7043', 0.18, 'Moderate drought'),
    (-1.0, -0.5, '#FFA726', 0.15, 'Mild drought'),
    (-0.5,  0.5, '#A5D6A7', 0.12, 'Normal'),
    ( 0.5,  1.5, '#4FC3F7', 0.15, 'Mild humid'),
    ( 1.5,  3.0, '#1976D2', 0.18, 'Moderate–extreme humid'),
]

GOV_COLORS = {
    'Kairouan':    '#1565C0',
    'Kasserine':   '#2E7D32',
    'Sidi Bou Zid':'#E65100',
    'Gafsa':       '#B71C1C',
    'Siliana':     '#6A1B9A',
}
GOV_ORDER = ['Siliana', 'Kasserine', 'Kairouan', 'Sidi Bou Zid', 'Gafsa']

# ============================================================
# CHARGEMENT
# ============================================================
print("Loading data...")
df_m   = pd.read_csv(MONTHLY)
df_a   = pd.read_csv(ANNUAL)
df_all = pd.read_csv(FULL)

df_m['date_dec'] = df_m['year'] + (df_m['month'] - 0.5) / 12
print(f"  Monthly: {len(df_m):,} | Annual: {len(df_a):,} | Full: {len(df_all):,}")

# ============================================================
# LAYOUT
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 4, figure=fig,
               height_ratios=[1.4, 1],
               hspace=0.38, wspace=0.35)

ax_ts   = fig.add_subplot(gs[0, :])
ax_maps = [fig.add_subplot(gs[1, i]) for i in range(4)]

# ============================================================
# (a) Série temporelle SSMI par gouvernorat
# ============================================================
print("Generating time series panel...")

ax_ts.set_facecolor('#FAFAFA')

for ymin, ymax, col, alpha, lbl in DROUGHT_BANDS:
    ax_ts.axhspan(ymin, ymax, color=col, alpha=alpha, zorder=0)

ax_ts.axhline(0, color='#546E7A', lw=0.8, ls='--', alpha=0.7, zorder=1)

for gov in GOV_ORDER:
    sub = df_m[df_m['governorate'] == gov].sort_values('date_dec')
    ax_ts.plot(sub['date_dec'], sub['SSMI'],
               color=GOV_COLORS[gov], lw=1.3, alpha=0.85,
               label=gov, zorder=3)

# Marqueurs années sèches majeures
drought_yrs = {2002: 'S2002', 2007: 'S2007', 2014: 'S2014',
               2021: 'S2021', 2022: 'S2022'}
for yr, lbl in drought_yrs.items():
    ax_ts.axvline(yr + 0.5, color='#BF360C', lw=1.0, ls=':', alpha=0.6, zorder=2)
    ax_ts.text(yr + 0.5, 2.55, str(yr), ha='center', va='bottom',
               fontsize=7, color='#BF360C', fontweight='bold')

xticks = list(range(2001, 2023, 2))
ax_ts.set_xticks(xticks)
ax_ts.set_xticklabels([str(y) for y in xticks], fontsize=8)
ax_ts.set_yticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
ax_ts.set_yticklabels([str(v) for v in [-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]],
                       fontsize=8)
ax_ts.set_xlim(2001, 2023)
ax_ts.set_ylim(-3.0, 3.0)
ax_ts.set_xlabel('Year', fontsize=9, labelpad=3)
ax_ts.set_ylabel('SSMI', fontsize=9, labelpad=3)
ax_ts.set_title('(a) Monthly SSMI time series by governorate (2001–2022)',
                fontsize=10, fontweight='bold', pad=5)
ax_ts.legend(loc='upper left', fontsize=8, framealpha=0.9,
             ncol=5, bbox_to_anchor=(0.0, 1.0))

# Annotations catégories sécheresse
cat_labels = [
    (-2.5, 'Extreme\ndrought', '#7B0000'),
    (-1.75,'Severe\ndrought',  '#D32F2F'),
    (-1.25,'Moderate\ndrought','#E64A19'),
    (-0.75,'Mild\ndrought',    '#F57C00'),
    ( 0.0, 'Normal',           '#388E3C'),
    ( 1.0, 'Humid',            '#1976D2'),
]
for y, lbl, col in cat_labels:
    ax_ts.text(2022.7, y, lbl, ha='left', va='center',
               fontsize=6.5, color=col, fontweight='bold',
               clip_on=False)

# ============================================================
# (b) Cartes spatiales SSMI pour 4 années
# ============================================================
print("Generating spatial maps...")

YEARS_TO_MAP = [
    (2002, 'Dry'),
    (2003, 'Wet'),
    (2016, 'Near-normal'),
    (2021, 'Dry'),
]

pixels = df_all[['lat','lon','governorate']].drop_duplicates().reset_index(drop=True)

RES  = 0.05
lo0, lo1 = pixels['lon'].min()-0.12, pixels['lon'].max()+0.12
la0, la1 = pixels['lat'].min()-0.12, pixels['lat'].max()+0.12
glon_r, glat_r = np.meshgrid(np.arange(lo0, lo1, RES),
                               np.arange(la0, la1, RES))
hull  = ConvexHull(pixels[['lon','lat']].values)
hpath = Path(pixels[['lon','lat']].values[hull.vertices])
flat  = np.column_stack([glon_r.ravel(), glat_r.ravel()])
mask  = hpath.contains_points(flat, radius=0.08).reshape(glon_r.shape)

cmap_ssmi = LinearSegmentedColormap.from_list(
    'ssmi',
    ['#7B0000','#D32F2F','#FF7043','#FFA726',
     '#FFFDE7',
     '#A5D6A7','#4FC3F7','#1976D2','#0D1B6E'],
    N=256
)

for k, (yr, label_yr) in enumerate(YEARS_TO_MAP):
    ax = ax_maps[k]
    sub_yr = df_all[df_all['year'] == yr].groupby(['lat','lon'])['SSMI'].mean().reset_index()
    pts  = sub_yr[['lon','lat']].values
    vals = sub_yr['SSMI'].values
    zi   = griddata(pts, vals, (glon_r, glat_r), method='linear')
    zn   = griddata(pts, vals, (glon_r, glat_r), method='nearest')
    zi   = np.where(np.isnan(zi), zn, zi)
    zi   = gaussian_filter(zi, sigma=0.7)
    zi_m = np.where(mask, zi, np.nan)

    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    im   = ax.pcolormesh(glon_r, glat_r, zi_m,
                          cmap=cmap_ssmi, norm=norm,
                          shading='auto', rasterized=True)

    for gov, col in GOV_COLORS.items():
        s = pixels[pixels['governorate'] == gov]
        ax.scatter(s['lon'], s['lat'], s=0.8, color=col,
                   alpha=0.2, linewidths=0, rasterized=True)

    gov_c = pixels.groupby('governorate')[['lat','lon']].mean()
    for gov, row in gov_c.iterrows():
        short = gov.replace('Sidi Bou Zid','SBZ')
        ax.text(row['lon'], row['lat'], short,
                ha='center', va='center', fontsize=5.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.12', fc='white',
                          alpha=0.65, lw=0))

    mean_yr_df = df_a[df_a['year'] == yr]['SSMI'].mean() if 'year' in df_a.columns else np.nan

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.88,
                 label='SSMI').ax.tick_params(labelsize=7)

    status_color = '#D32F2F' if mean_yr_df < -0.3 else ('#1565C0' if mean_yr_df > 0.3 else '#37474F')
    ax.set_title(f'(b{k+1}) {yr}  [{label_yr}]\nMean SSMI = {mean_yr_df:.2f}',
                 fontsize=8.5, fontweight='bold', color=status_color, pad=3)
    ax.set_xlabel('Lon (°E)', fontsize=7.5, labelpad=2)
    ax.set_ylabel('Lat (°N)', fontsize=7.5, labelpad=2)
    ax.tick_params(labelsize=7)

plt.suptitle(
    'Standardized Soil Moisture Index (SSMI) across five semi-arid '
    'governorates of central Tunisia, 2001–2022\n'
    '(GLEAM v4.2a, 0.05° resolution)',
    fontsize=11, fontweight='bold', y=1.01
)

plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Saved → {OUT}")