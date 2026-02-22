"""
merge_dataset.py
================
Assemblage du dataset ML complet — VERSION FINALE
15 features (dont SM MERRA-2) + SSMI (GLEAM v4.2a) pour 2001-2022
154,704 observations · 586 pixels · 0.05° résolution

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

NOTE MÉTHODOLOGIQUE :
  - Variable CIBLE (SSMI) : dérivée EXCLUSIVEMENT de GLEAM v4.2a
  - Prédicteur SM : provient de NASA POWER MERRA-2 (source INDÉPENDANTE)
  => Pas de circularité mathématique entre target et prédicteurs

FICHIERS D'ENTRÉE :
  ssmi_output/ssmi_tunisia_2001_2022.csv   ← GLEAM v4.2a (target only)
  nasa_monthly/nasa_power_monthly.csv      ← MERRA-2 (predictors)
  soil_data/soilgrids_tunisia.csv          ← SoilGrids v2.0
  gee_data/gee_terrain_landuse_pop.csv     ← SRTM + ESA + WorldPop

SORTIE :
  ml_dataset/ml_dataset_v4.csv            ← dataset complet avec métadonnées
  ml_dataset/ml_features_target_v4.csv    ← 15 features + SSMI, prêt pour ML
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'ml_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CHARGEMENT
# ============================================================
print("Loading data files...")

# SSMI pixel par pixel — GLEAM v4.2a (TARGET ONLY, not used as predictor)
df_ssmi = pd.read_csv('ssmi_output/ssmi_tunisia_2001_2022.csv')
print(f"  ✅ SSMI (GLEAM v4.2a): {len(df_ssmi):,} rows")

# NASA POWER MERRA-2 mensuel (PREDICTORS — independent from GLEAM)
df_nasa = pd.read_csv('nasa_monthly/nasa_power_monthly.csv')
print(f"  ✅ NASA POWER MERRA-2 monthly: {len(df_nasa):,} rows")
print(f"     Columns: {list(df_nasa.columns)}")

# SoilGrids v2.0
df_soil = pd.read_csv('soil_data/soilgrids_tunisia.csv')
print(f"  ✅ SoilGrids v2.0: {len(df_soil):,} rows")

# GEE (SRTM + ESA WorldCover + WorldPop)
df_gee = pd.read_csv('gee_data/gee_terrain_landuse_pop.csv')
print(f"  ✅ GEE (SRTM+ESA+WorldPop): {len(df_gee):,} rows")

# ============================================================
# GOUVERNORATS (5 gouvernorats semi-arides)
# ============================================================
gov_bounds = {
    'Kairouan':     {'lat': (35.0, 36.2), 'lon': (9.0,  10.3)},
    'Kasserine':    {'lat': (34.7, 36.0), 'lon': (7.8,   9.2)},
    'Sidi Bou Zid': {'lat': (34.4, 35.5), 'lon': (9.0,  10.2)},
    'Gafsa':        {'lat': (33.8, 35.0), 'lon': (7.8,   9.5)},
    'Siliana':      {'lat': (35.8, 36.7), 'lon': (8.8,  10.0)},
}

def assign_gov(lat, lon):
    for gov, b in gov_bounds.items():
        if (b['lat'][0] <= lat <= b['lat'][1] and
                b['lon'][0] <= lon <= b['lon'][1]):
            return gov
    return 'Other'

# Filtrer SSMI sur les 5 gouvernorats d'étude
df_ssmi['governorate'] = df_ssmi.apply(
    lambda r: assign_gov(r['lat'], r['lon']), axis=1)
df_ssmi = df_ssmi[df_ssmi['governorate'] != 'Other'].copy()
n_pixels = df_ssmi[['lat','lon']].drop_duplicates().shape[0]
print(f"\n  Study pixels: {n_pixels}")
for gov in gov_bounds:
    n = df_ssmi[df_ssmi['governorate'] == gov][['lat','lon']].drop_duplicates().shape[0]
    print(f"     {gov}: {n} pixels")

# ============================================================
# PIXELS UNIQUES (grille GLEAM 0.05°)
# ============================================================
pixels = df_ssmi[['lat','lon']].drop_duplicates().reset_index(drop=True)
pts_pixels = pixels[['lon','lat']].values
print(f"\n  Unique pixels: {len(pixels)}")

# ============================================================
# INTERPOLATION FEATURES STATIQUES SUR PIXELS SSMI
# ============================================================
print("\nInterpolating static features onto SSMI pixels...")

def interp_on_pixels(df_src, col, lat_col='lat', lon_col='lon'):
    """Interpolation bilinéaire + nearest neighbor fallback."""
    pts  = df_src[[lon_col, lat_col]].values
    vals = pd.to_numeric(df_src[col], errors='coerce').values
    ok   = ~np.isnan(vals)
    if ok.sum() < 5:
        return np.full(len(pts_pixels), np.nan)
    v = griddata(pts[ok], vals[ok], pts_pixels, method='linear')
    v = np.where(np.isnan(v),
                 griddata(pts[ok], vals[ok], pts_pixels, method='nearest'), v)
    return v

# SoilGrids v2.0
for col in ['clay', 'sand', 'silt', 'bdod', 'PAWC']:
    if col in df_soil.columns:
        pixels[col] = interp_on_pixels(df_soil, col)
        print(f"  ✅ {col}: {pixels[col].min():.2f} – {pixels[col].max():.2f}")

# Soil depth — proxy depuis bdod (bulk density)
if 'bdod' in pixels.columns:
    bdod_n = (pixels['bdod'] - pixels['bdod'].min()) / \
             (pixels['bdod'].max() - pixels['bdod'].min() + 1e-9)
    pixels['soil_depth'] = (150 - bdod_n * 100).round(1)
    print(f"  ✅ soil_depth: {pixels['soil_depth'].min():.1f} – {pixels['soil_depth'].max():.1f} cm")

# GEE (SRTM + ESA WorldCover + WorldPop)
for col in ['elevation', 'slope', 'TWI', 'land_use', 'population']:
    if col in df_gee.columns:
        pixels[col] = interp_on_pixels(df_gee, col)
        print(f"  ✅ {col}: {pixels[col].min():.2f} – {pixels[col].max():.2f}")

pixels['governorate'] = pixels.apply(
    lambda r: assign_gov(r['lat'], r['lon']), axis=1)

# ============================================================
# INTERPOLATION NASA POWER MERRA-2 MENSUEL
# SM (GWETROOT) de MERRA-2 est un PRÉDICTEUR INDÉPENDANT du SSMI GLEAM
# ============================================================
print("\nInterpolating NASA POWER MERRA-2 monthly onto SSMI pixels...")

# Tous les paramètres disponibles (excl. coordonnées et temps)
nasa_params = [c for c in df_nasa.columns
               if c not in ['lat', 'lon', 'year', 'month']]
print(f"  NASA MERRA-2 params: {nasa_params}")

monthly_dfs = []
months_done = 0
total_months = df_nasa[['year','month']].drop_duplicates().shape[0]

for (year, month), df_nm in df_nasa.groupby(['year', 'month']):
    pts_nasa = df_nm[['lon', 'lat']].values

    df_m = pixels.copy()
    df_m['year']  = year
    df_m['month'] = month

    for param in nasa_params:
        if param in df_nm.columns:
            vals = df_nm[param].values.astype(float)
            ok   = ~np.isnan(vals)
            if ok.sum() >= 3:
                v = griddata(pts_nasa[ok], vals[ok],
                             pts_pixels, method='linear')
                v = np.where(
                    np.isnan(v),
                    griddata(pts_nasa[ok], vals[ok],
                             pts_pixels, method='nearest'), v)
            else:
                v = np.full(len(pts_pixels), np.nan)
            df_m[param] = v

    monthly_dfs.append(df_m)
    months_done += 1
    if months_done % 48 == 0:
        print(f"  Processed {months_done}/{total_months} months...")

df_all = pd.concat(monthly_dfs, ignore_index=True)
print(f"  ✅ Monthly dataset: {len(df_all):,} rows")

# ============================================================
# FUSION AVEC SSMI GLEAM PIXEL PAR PIXEL
# ============================================================
print("\nMerging with SSMI (GLEAM v4.2a) pixel by pixel...")

df_ssmi['lat'] = df_ssmi['lat'].round(3)
df_ssmi['lon'] = df_ssmi['lon'].round(3)
df_all['lat']  = df_all['lat'].round(3)
df_all['lon']  = df_all['lon'].round(3)

df_final = df_all.merge(
    df_ssmi[['lat', 'lon', 'year', 'month', 'SSMI']],
    on=['lat', 'lon', 'year', 'month'],
    how='inner'
)

print(f"  ✅ Merged: {len(df_final):,} rows | SSMI valid: {df_final['SSMI'].notna().sum():,}")

# ============================================================
# FEATURES FINALES — 15 PRÉDICTEURS
# NOTE: SM ici = GWETROOT de MERRA-2, PAS de GLEAM
# ============================================================
FEATURES = [
    # --- NASA POWER MERRA-2 (dynamiques, mensuels) ---
    'SM',           # GWETROOT MERRA-2 (rename pour clarté dans le papier)
    'PRECTOTCORR',  # Précipitation mensuelle corrigée
    'T2M',          # Température moyenne 2m
    'GWETROOT',     # Soil moisture root zone MERRA-2 (alias SM)
    'RH2M',         # Humidité relative 2m
    'EVPTRNS',      # Évapotranspiration
    # --- SoilGrids v2.0 (statiques) ---
    'clay',         # Teneur en argile (%)
    'sand',         # Teneur en sable (%)
    'PAWC',         # Plant Available Water Capacity (mm)
    'soil_depth',   # Profondeur de sol (cm)
    # --- SRTM DEM (statiques) ---
    'elevation',    # Altitude (m)
    'slope',        # Pente (degrés)
    'TWI',          # Topographic Wetness Index
    # --- ESA WorldCover + WorldPop (statiques) ---
    'land_use',     # Occupation du sol
    'population',   # Densité de population (inh/km²)
]

TARGET = 'SSMI'   # Dérivé EXCLUSIVEMENT de GLEAM v4.2a

# Renommer GWETROOT en SM si nécessaire (pour correspondre au papier)
if 'GWETROOT' in df_final.columns and 'SM' not in df_final.columns:
    df_final['SM'] = df_final['GWETROOT']

available = [f for f in FEATURES if f in df_final.columns
             and not df_final[f].isna().all()]
# Éviter les doublons (SM et GWETROOT identiques)
if 'SM' in available and 'GWETROOT' in available:
    available = [f for f in available if f != 'GWETROOT']

missing = [f for f in ['SM','PRECTOTCORR','T2M','RH2M','EVPTRNS',
                        'elevation','slope','TWI','clay','sand',
                        'PAWC','soil_depth','land_use','population']
           if f not in available]

print(f"\n  ✅ Features ({len(available)}): {available}")
if missing:
    print(f"  ⚠️  Missing: {missing}")

cols = ['lat', 'lon', 'governorate', 'year', 'month'] + available + [TARGET]
df_final = df_final[[c for c in cols if c in df_final.columns]].dropna(subset=[TARGET])

# ============================================================
# STATISTIQUES FINALES
# ============================================================
print(f"\n{'='*60}")
print(f"FINAL ML DATASET STATISTICS")
print(f"{'='*60}")
print(f"  Rows       : {len(df_final):,}")
print(f"  Features   : {len(available)}")
print(f"  Target     : SSMI (GLEAM v4.2a, standardized)")
print(f"  Pixels     : {df_final[['lat','lon']].drop_duplicates().shape[0]}")
print(f"  Period     : {df_final['year'].min()}–{df_final['year'].max()}")

print(f"\nBy governorate:")
print(df_final.groupby('governorate').size().to_string())

feat_cols = [f for f in available if f in df_final.columns]
print(f"\nFeature statistics:")
print(df_final[feat_cols + [TARGET]].describe().round(3).to_string())

# ============================================================
# SAUVEGARDE
# ============================================================
out_full  = os.path.join(OUTPUT_DIR, 'ml_dataset_v4.csv')
out_feat  = os.path.join(OUTPUT_DIR, 'ml_features_target_v4.csv')

df_final.to_csv(out_full, index=False)
df_final[feat_cols + [TARGET]].dropna().to_csv(out_feat, index=False)

print(f"\n✅ Saved: {out_full}  ({len(df_final):,} rows)")
print(f"✅ Saved: {out_feat}  (features + SSMI only)")
print(f"\n✅ Dataset assembly complete!")
print(f"   → Ready for train_models.py")