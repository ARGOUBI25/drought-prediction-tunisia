"""
download_soilgrids.py
=====================
Téléchargement SoilGrids v2.0 (ISRIC) pour la Tunisie centrale
Variables : clay, sand, silt, PAWC, bdod — profondeur 0–30 cm

MÉTHODE : Échantillonnage sparse 0.25° (~160 points) via API REST ISRIC
          puis interpolation bilinéaire sur grille cible 0.05°
          (beaucoup plus rapide que point par point à 0.05°)

API     : https://rest.isric.org/soilgrids/v2.0/properties/query
Durée   : ~15–25 min (160 points, rate limit respecté)

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

SORTIE :
  soil_data/soilgrids_tunisia.csv    ← grille 0.05° interpolée
  soil_data/soilgrids_sparse.csv     ← points bruts 0.25°
  Colonnes : lat, lon, clay(%), sand(%), silt(%), bdod(g/cm³),
             wv0010(m³/m³), wv1500(m³/m³), PAWC(mm)
"""

import numpy as np
import pandas as pd
import requests
import os
import time
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'soil_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GRILLE CIBLE 0.05°
# ============================================================
LAT_MIN, LAT_MAX = 33.5, 37.0
LON_MIN, LON_MAX =  7.5, 10.8
RES = 0.05

lats_target = np.arange(LAT_MIN, LAT_MAX + RES, RES)
lons_target = np.arange(LON_MIN, LON_MAX + RES, RES)
LON_G, LAT_G = np.meshgrid(lons_target, lats_target)
pts_target = np.column_stack([LON_G.ravel(), LAT_G.ravel()])

print(f"Target grid: {len(lats_target)} lat × {len(lons_target)} lon = "
      f"{len(pts_target)} points at 0.05°")

# ============================================================
# GRILLE D'ÉCHANTILLONNAGE SPARSE 0.25°
# (on télécharge ~160 points puis on interpole)
# ============================================================
SAMPLE_RES = 0.25
lats_s = np.arange(LAT_MIN, LAT_MAX + SAMPLE_RES, SAMPLE_RES)
lons_s = np.arange(LON_MIN, LON_MAX + SAMPLE_RES, SAMPLE_RES)
total  = len(lats_s) * len(lons_s)
print(f"Sampling {total} points at {SAMPLE_RES}° → interpolating to {RES}°")
print(f"Estimated time: ~{total * 0.45 / 60:.0f}–{total * 0.6 / 60:.0f} minutes\n")

# ============================================================
# API SOILGRIDS v2.0 — CONFIGURATION
# ============================================================
BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Variables et profondeurs à télécharger
SOIL_VARS = [
    ('clay',   '0-30cm'),   # argile (g/kg → %)
    ('sand',   '0-30cm'),   # sable (g/kg → %)
    ('silt',   '0-30cm'),   # limon (g/kg → %)
    ('bdod',   '0-30cm'),   # densité apparente (cg/cm³ → g/cm³)
    ('wv0010', '0-30cm'),   # teneur eau pF=2 (field capacity)
    ('wv1500', '0-30cm'),   # teneur eau pF=4.2 (wilting point)
    ('ocd',    '0-100cm'),  # densité carbone organique (proxy depth)
]

# ============================================================
# TEST CONNEXION API
# ============================================================
print("Testing SoilGrids API...")
test_url = (f"{BASE}?lon=9.0&lat=35.0"
            f"&property=clay&depth=0-30cm&value=mean")
try:
    r = requests.get(test_url, timeout=15)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print("  ✅ SoilGrids API OK")
    else:
        print(f"  ⚠️  Unexpected status. Response: {r.text[:200]}")
except Exception as e:
    print(f"  ❌ API not reachable: {e}")

# ============================================================
# TÉLÉCHARGEMENT POINT PAR POINT (sparse 0.25°)
# ============================================================
print(f"\nDownloading SoilGrids data ({total} points)...\n")

results = []
count   = 0
errors  = 0

for lat in lats_s:
    for lon in lons_s:
        count += 1
        row = {'lat': round(lat, 3), 'lon': round(lon, 3)}

        # Construire URL avec toutes les variables en une seule requête
        prop_q  = '&'.join([f'property={v[0]}' for v in SOIL_VARS])
        depth_q = '&'.join([f'depth={v[1]}'    for v in SOIL_VARS])
        url     = f"{BASE}?lon={lon}&lat={lat}&{prop_q}&{depth_q}&value=mean"

        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=25)

                if resp.status_code == 200:
                    data   = resp.json()
                    layers = data.get('properties', {}).get('layers', [])
                    for layer in layers:
                        pname  = layer.get('name', '')
                        depths = layer.get('depths', [])
                        for d in depths:
                            val = d.get('values', {}).get('mean')
                            row[pname] = val
                    break

                elif resp.status_code == 429:   # Rate limit
                    wait = 3 ** attempt
                    print(f"  Rate limit ({lat:.2f},{lon:.2f}) "
                          f"— waiting {wait}s...")
                    time.sleep(wait)

                else:
                    errors += 1
                    break

            except Exception as e:
                time.sleep(1.5 * (attempt + 1))
                errors += 1

        results.append(row)
        time.sleep(0.4)   # Pause obligatoire

        if count % 20 == 0 or count == total:
            print(f"  Progress: {count}/{total} "
                  f"({count/total*100:.0f}%) | Errors: {errors}")

# ============================================================
# CONVERSION DES UNITÉS
# ============================================================
print("\nConverting units...")
df = pd.DataFrame(results)
print(f"  Raw columns: {list(df.columns)}")

# clay, sand, silt : g/kg × 0.1 = %
for col in ['clay', 'sand', 'silt']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 10.0

# bdod : cg/cm³ × 0.01 = g/cm³
if 'bdod' in df.columns:
    df['bdod'] = pd.to_numeric(df['bdod'], errors='coerce') / 100.0

# wv0010, wv1500 : cm³/dm³ × 0.001 = m³/m³
for col in ['wv0010', 'wv1500']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 1000.0

# ocd : hg/m³ × 0.1 = kg/m³
if 'ocd' in df.columns:
    df['ocd'] = pd.to_numeric(df['ocd'], errors='coerce') / 10.0

# PAWC = (field capacity - wilting point) × 300 mm (0–30cm)
if 'wv0010' in df.columns and 'wv1500' in df.columns:
    df['PAWC'] = (df['wv0010'] - df['wv1500']) * 300
    df['PAWC'] = df['PAWC'].clip(0, 250)
    print(f"  ✅ PAWC: {df['PAWC'].min():.1f}–{df['PAWC'].max():.1f} mm")

# Soil depth depuis ocd (proxy)
if 'ocd' in df.columns:
    df['soil_depth'] = df['ocd']

print(f"  ✅ Columns: {list(df.columns)}")

# ============================================================
# INTERPOLATION SUR GRILLE FINE 0.05°
# ============================================================
print(f"\nInterpolating {total} sparse points → {len(pts_target)} grid points...")

soil_vars = [c for c in df.columns if c not in ['lat', 'lon']]
pts_src   = df[['lon', 'lat']].values

df_fine = pd.DataFrame({'lat': LAT_G.ravel(), 'lon': LON_G.ravel()})

for var in soil_vars:
    valid = df[var].notna()
    if valid.sum() < 5:
        print(f"  ⚠️  {var}: only {valid.sum()} valid points")
        df_fine[var] = np.nan
        continue

    pts_v = pts_src[valid]
    vals  = df[var].values[valid]

    interp = griddata(pts_v, vals, pts_target, method='cubic')
    interp = np.where(
        np.isnan(interp),
        griddata(pts_v, vals, pts_target, method='nearest'),
        interp
    )
    df_fine[var] = interp
    print(f"  ✅ {var:10s}: {np.nanmin(interp):.2f}–{np.nanmax(interp):.2f}")

# ============================================================
# STATISTIQUES ET SAUVEGARDE
# ============================================================
print(f"\n{'='*55}")
print(f"SOILGRIDS DOWNLOAD SUMMARY")
print(f"{'='*55}")
print(f"  Sparse points   : {len(df)}")
print(f"  Fine grid points: {len(df_fine)}")
print(f"  Errors          : {errors}")

show = [c for c in ['clay','sand','silt','PAWC'] if c in df_fine.columns]
if show:
    print(f"\nStatistics (0.05° grid):")
    print(df_fine[show].describe().round(2).to_string())

# Sauvegarder les deux versions
df.to_csv(os.path.join(OUTPUT_DIR, 'soilgrids_sparse.csv'), index=False)
df_fine.to_csv(os.path.join(OUTPUT_DIR, 'soilgrids_tunisia.csv'), index=False)

print(f"\n✅ Saved: soil_data/soilgrids_sparse.csv  (raw {SAMPLE_RES}° points)")
print(f"✅ Saved: soil_data/soilgrids_tunisia.csv  (interpolated {RES}° grid)")
print(f"\n✅ SoilGrids download complete!")
print(f"   → Next: run src/data/download_terrain.py")
