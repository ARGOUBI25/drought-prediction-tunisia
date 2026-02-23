"""
download_nasa_power.py
======================
Téléchargement NASA POWER MERRA-2 — données MENSUELLES 2001-2022
7 variables climatiques pour les 5 gouvernorats de Tunisie centrale

API  : https://power.larc.nasa.gov/api/temporal/monthly/point
Résol: 0.5° (résolution native MERRA-2)
Durée: ~5–10 min (70 points × 264 mois)

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

SORTIE :
  nasa_monthly/nasa_power_monthly.csv
  Colonnes : lat, lon, year, month, PRECTOTCORR, T2M, T2M_MAX, T2M_MIN,
             RH2M, GWETROOT, EVPTRNS
"""

import numpy as np
import pandas as pd
import requests
import time
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'nasa_monthly'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GRILLE 0.5° (résolution native NASA POWER MERRA-2)
# ============================================================
LAT_MIN, LAT_MAX = 33.5, 37.0
LON_MIN, LON_MAX =  7.5, 10.8

lats = np.arange(LAT_MIN, LAT_MAX + 0.5, 0.5)
lons = np.arange(LON_MIN, LON_MAX + 0.5, 0.5)
total = len(lats) * len(lons)

print(f"NASA POWER MERRA-2 grid: {len(lats)} lat × {len(lons)} lon = {total} points")
print(f"Period: 2001-2022 = 264 months per point")
print(f"Estimated time: {total * 0.5 / 60:.0f}–{total * 0.8 / 60:.0f} minutes")

# Variables à télécharger
PARAMS = 'PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,GWETROOT,EVPTRNS'
BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# ============================================================
# TEST CONNEXION API
# ============================================================
print("\nTesting NASA POWER API connection...")
test_params = {
    'parameters': PARAMS,
    'community':  'AG',
    'longitude':  9.0,
    'latitude':   35.0,
    'start':      '20010101',
    'end':        '20221231',
    'format':     'JSON',
    'user':       'anonymous',
}

resp = requests.get(BASE_URL, params=test_params, timeout=30)
print(f"  Status: {resp.status_code}")

if resp.status_code != 200:
    print(f"  ⚠️  API not responding. Check: https://power.larc.nasa.gov/")
    print(f"  Response: {resp.text[:200]}")
    raise SystemExit(1)

# Vérifier les paramètres disponibles
data = resp.json()
props = data.get('properties', {}).get('parameter', {})
print(f"  ✅ API OK | Parameters: {list(props.keys())}")

# ============================================================
# TÉLÉCHARGEMENT POINT PAR POINT
# ============================================================
print(f"\nDownloading monthly data ({total} points)...\n")

all_records = []
count  = 0
errors = 0

for lat in lats:
    for lon in lons:
        count += 1

        params_pt = {
            'parameters': PARAMS,
            'community':  'AG',
            'longitude':  lon,
            'latitude':   lat,
            'start':      '20010101',
            'end':        '20221231',
            'format':     'JSON',
            'user':       'anonymous',
        }

        for attempt in range(4):
            try:
                r = requests.get(BASE_URL, params=params_pt, timeout=45)

                if r.status_code == 200:
                    d = r.json()
                    props = d.get('properties', {}).get('parameter', {})

                    prec = props.get('PRECTOTCORR', {})
                    t2m  = props.get('T2M', {})
                    t2mx = props.get('T2M_MAX', {})
                    t2mn = props.get('T2M_MIN', {})
                    rh   = props.get('RH2M', {})
                    gw   = props.get('GWETROOT', {})
                    ev   = props.get('EVPTRNS', {})

                    for k in prec.keys():
                        if len(k) == 6:  # Format YYYYMM
                            yr = int(k[:4])
                            mo = int(k[4:])
                            if 2001 <= yr <= 2022:
                                all_records.append({
                                    'lat':         round(lat, 2),
                                    'lon':         round(lon, 2),
                                    'year':        yr,
                                    'month':       mo,
                                    'PRECTOTCORR': prec.get(k, np.nan),
                                    'T2M':         t2m.get(k, np.nan),
                                    'T2M_MAX':     t2mx.get(k, np.nan),
                                    'T2M_MIN':     t2mn.get(k, np.nan),
                                    'RH2M':        rh.get(k, np.nan),
                                    'GWETROOT':    gw.get(k, np.nan),
                                    'EVPTRNS':     ev.get(k, np.nan),
                                })
                    break

                elif r.status_code == 429:   # Rate limit
                    wait = 5 * (attempt + 1)
                    print(f"  Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ⚠️  ({lat},{lon}): HTTP {r.status_code}")
                    errors += 1
                    break

            except Exception as e:
                time.sleep(2 * (attempt + 1))
                errors += 1

        time.sleep(0.5)   # Respecter rate limit

        if count % 10 == 0 or count == total:
            print(f"  Progress: {count}/{total} pts | "
                  f"Records: {len(all_records):,} | Errors: {errors}")

# ============================================================
# POST-TRAITEMENT
# ============================================================
print("\nPost-processing...")
df = pd.DataFrame(all_records)

# Remplacer les valeurs manquantes NASA POWER (-999, -99)
for col in ['PRECTOTCORR', 'T2M', 'T2M_MAX', 'T2M_MIN',
            'RH2M', 'GWETROOT', 'EVPTRNS']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < -900, col] = np.nan

# ============================================================
# STATISTIQUES
# ============================================================
print(f"\n{'='*55}")
print(f"NASA POWER MERRA-2 DOWNLOAD SUMMARY")
print(f"{'='*55}")
print(f"  Total records : {len(df):,}")
print(f"  Grid points   : {df[['lat','lon']].drop_duplicates().shape[0]}")
print(f"  Months        : {df[['year','month']].drop_duplicates().shape[0]}")
print(f"  Errors        : {errors}")
print(f"\nStatistics:")
print(df[['PRECTOTCORR','T2M','RH2M','GWETROOT','EVPTRNS']].describe().round(3))

# ============================================================
# SAUVEGARDE
# ============================================================
out_path = os.path.join(OUTPUT_DIR, 'nasa_power_monthly.csv')
df.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}  ({len(df):,} rows)")
print(f"✅ NASA POWER MERRA-2 download complete!")
print(f"   → Next: run src/data/extract_gleam_sm.py")
