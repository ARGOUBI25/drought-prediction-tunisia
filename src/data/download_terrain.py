"""
download_terrain.py
===================
Téléchargement SRTM DEM 90m + calcul Slope + TWI pour la Tunisie centrale

SOURCE  : SRTM 90m (SRTMGL3) via OpenTopography API (accès public)
          Fallback 1 : package `elevation` (USGS)
          Fallback 2 : tiles CGIAR SRTM

CALCULS :
  - Slope (degrés) : gradient 2D + arctan
  - TWI : ln(catchment_area / tan(slope))

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

PRÉREQUIS :
  pip install rasterio requests
  (optionnel) pip install elevation  ← fallback 1

SORTIE :
  terrain_data/terrain_tunisia.csv
  Colonnes : lat, lon, elevation(m), slope(°), TWI
"""

import numpy as np
import pandas as pd
import os
import requests
import zipfile
import io
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'terrain_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GRILLE CIBLE 0.05°
# ============================================================
LAT_MIN, LAT_MAX = 33.5, 37.0
LON_MIN, LON_MAX =  7.5, 10.8
RES = 0.05

lats = np.arange(LAT_MIN, LAT_MAX + RES, RES)
lons = np.arange(LON_MIN, LON_MAX + RES, RES)
LON_G, LAT_G = np.meshgrid(lons, lats)
pts_target = np.column_stack([LON_G.ravel(), LAT_G.ravel()])

print(f"Target grid: {len(lats)} × {len(lons)} = {len(pts_target)} points")

# ============================================================
# MÉTHODE 1 — OpenTopography API (SRTMGL3, 90m)
# ============================================================
print("\nTrying OpenTopography API (SRTM 90m)...")

OT_URL = "https://portal.opentopography.org/API/globaldem"
params = {
    'demtype':     'SRTMGL3',
    'south':       LAT_MIN - 0.5,
    'north':       LAT_MAX + 0.5,
    'west':        LON_MIN - 0.5,
    'east':        LON_MAX + 0.5,
    'outputFormat':'GTiff',
    'API_Key':     'demoapikeyot',   # clé demo publique
}

dem_path   = os.path.join(OUTPUT_DIR, 'srtm_tunisia.tif')
USE_RASTER = False

try:
    resp = requests.get(OT_URL, params=params, timeout=90)
    if resp.status_code == 200 and len(resp.content) > 50_000:
        with open(dem_path, 'wb') as f:
            f.write(resp.content)
        print(f"  ✅ DEM downloaded ({len(resp.content)/1e6:.1f} MB)")
        USE_RASTER = True
    else:
        print(f"  ⚠️  Status {resp.status_code} / size {len(resp.content)}")
except Exception as e:
    print(f"  ⚠️  OpenTopography error: {e}")

# ============================================================
# MÉTHODE 2 — Package `elevation` (USGS SRTM)
# ============================================================
if not USE_RASTER:
    print("\nTrying `elevation` package (USGS)...")
    try:
        import elevation
        elevation.clip(
            bounds=(LON_MIN-0.5, LAT_MIN-0.5, LON_MAX+0.5, LAT_MAX+0.5),
            output=os.path.abspath(dem_path),
            product='SRTM3'
        )
        print("  ✅ DEM downloaded via elevation package")
        USE_RASTER = True
    except ImportError:
        print("  ⚠️  Not installed: pip install elevation")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")

# ============================================================
# MÉTHODE 3 — Tiles CGIAR SRTM (fallback final)
# ============================================================
if not USE_RASTER:
    print("\nDownloading SRTM tiles from CGIAR...")

    all_pts  = []
    all_elev = []

    # Tiles nécessaires pour la Tunisie centrale
    for lat_t in [33, 34, 35, 36]:
        for lon_t in [7, 8, 9, 10]:
            col   = int((lon_t + 180) / 5) + 1
            row   = int((60 - lat_t) / 5) + 1
            fname = f"srtm_{col:02d}_{row:02d}.zip"
            url   = (f"https://srtm.csi.cgiar.org/wp-content/uploads/"
                     f"files/srtm_5x5/TIFF/{fname}")

            try:
                print(f"  Downloading tile: {fname}...", end='', flush=True)
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        tif_name = [n for n in z.namelist()
                                    if n.endswith('.tif')][0]
                        z.extract(tif_name, OUTPUT_DIR)

                    import rasterio
                    tile_path = os.path.join(OUTPUT_DIR, tif_name)
                    with rasterio.open(tile_path) as src:
                        data = src.read(1).astype(float)
                        nd   = src.nodata
                        if nd is not None:
                            data[data == nd] = np.nan
                        lons_t = np.linspace(src.bounds.left,
                                              src.bounds.right, src.width)
                        lats_t = np.linspace(src.bounds.top,
                                              src.bounds.bottom, src.height)
                        LN, LT = np.meshgrid(lons_t, lats_t)
                        mask = ((LT >= LAT_MIN) & (LT <= LAT_MAX) &
                                (LN >= LON_MIN) & (LN <= LON_MAX) &
                                ~np.isnan(data))
                        all_pts.append(np.column_stack([LN[mask], LT[mask]]))
                        all_elev.append(data[mask])
                    print(f" ✅")
                else:
                    print(f" ⚠️  HTTP {r.status_code}")

            except Exception as e:
                print(f" ⚠️  {e}")

    if all_pts:
        pts_dem  = np.vstack(all_pts)
        elev_dem = np.concatenate(all_elev)
        print(f"  ✅ {len(elev_dem):,} DEM points loaded from tiles")
        USE_RASTER = False   # On a les points directs, pas besoin de rasterio
        HAVE_DEM   = True
    else:
        HAVE_DEM = False

# ============================================================
# LIRE LE GEOTIFF SI DISPONIBLE
# ============================================================
HAVE_DEM = False
if USE_RASTER and os.path.exists(dem_path):
    try:
        import rasterio
        print("\nReading DEM GeoTiff...")
        with rasterio.open(dem_path) as src:
            dem_data  = src.read(1).astype(float)
            nodata    = src.nodata
            dem_lons  = np.array([src.xy(0, j)[0] for j in range(src.width)])
            dem_lats  = np.array([src.xy(i, 0)[1] for i in range(src.height)])

        if nodata is not None:
            dem_data[dem_data == nodata] = np.nan

        print(f"  DEM shape: {dem_data.shape}")
        print(f"  Elevation: {np.nanmin(dem_data):.0f}–{np.nanmax(dem_data):.0f} m")

        DEM_LON, DEM_LAT = np.meshgrid(dem_lons, dem_lats)
        pts_dem  = np.column_stack([DEM_LON.ravel(), DEM_LAT.ravel()])
        elev_dem = dem_data.ravel()
        valid    = ~np.isnan(elev_dem)
        pts_dem  = pts_dem[valid]
        elev_dem = elev_dem[valid]
        HAVE_DEM = True

    except ImportError:
        print("  ⚠️  rasterio not installed: pip install rasterio")
    except Exception as e:
        print(f"  ⚠️  rasterio error: {e}")

# ============================================================
# INTERPOLATION SUR GRILLE CIBLE 0.05°
# ============================================================
if HAVE_DEM:
    print("\nInterpolating DEM to 0.05° grid...")
    elev_grid = griddata(pts_dem, elev_dem, pts_target, method='linear')
    elev_grid = np.where(
        np.isnan(elev_grid),
        griddata(pts_dem, elev_dem, pts_target, method='nearest'),
        elev_grid
    )
    print(f"  ✅ Elevation: {np.nanmin(elev_grid):.0f}–{np.nanmax(elev_grid):.0f} m")
else:
    print("\n⚠️  No DEM data — using elevation proxy (not recommended for publication)")
    # Proxy basé sur la géographie connue de la Tunisie centrale
    lat_norm = (LAT_G.ravel() - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_norm = (LON_G.ravel() - LON_MIN) / (LON_MAX - LON_MIN)
    elev_grid = (400 * np.exp(-((lon_norm - 0.2)**2 / 0.1 +
                                 (lat_norm - 0.6)**2 / 0.15)) +
                 200 * (1 - lon_norm) + 50)
    print("  ⚠️  Using elevation proxy — install rasterio for real data!")

# ============================================================
# CALCUL SLOPE (degrés)
# ============================================================
print("\nComputing slope...")
elev_mat = elev_grid.reshape(len(lats), len(lons))

# Résolution en mètres à la latitude moyenne
dlat_m = RES * 111320
dlon_m = RES * 111320 * np.cos(np.radians(np.mean(lats)))

dz_dy, dz_dx = np.gradient(elev_mat, dlat_m, dlon_m)
slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
print(f"  ✅ Slope: {slope_deg.min():.1f}–{slope_deg.max():.1f}°")

# ============================================================
# CALCUL TWI (Topographic Wetness Index)
# TWI = ln(A / tan(slope))
# ============================================================
print("\nComputing TWI...")

def compute_twi(elev, res_m):
    dy, dx = np.gradient(elev, res_m, res_m)
    slope_r = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_r = np.where(slope_r < 1e-3, 1e-3, slope_r)
    # Flow accumulation simplifié (chaque cellule = res²)
    catchment = np.ones_like(elev) * (res_m ** 2)
    return np.clip(np.log(catchment / np.tan(slope_r)), 0, 25)

twi_mat = compute_twi(elev_mat, dlat_m)
print(f"  ✅ TWI: {twi_mat.min():.1f}–{twi_mat.max():.1f}")

# ============================================================
# ASSEMBLER ET SAUVEGARDER
# ============================================================
print("\nAssembling terrain DataFrame...")

df_terrain = pd.DataFrame({
    'lat':       LAT_G.ravel(),
    'lon':       LON_G.ravel(),
    'elevation': elev_grid,
    'slope':     slope_deg.ravel(),
    'TWI':       twi_mat.ravel(),
})

df_terrain = df_terrain[
    (df_terrain['lat'] >= LAT_MIN) & (df_terrain['lat'] <= LAT_MAX) &
    (df_terrain['lon'] >= LON_MIN) & (df_terrain['lon'] <= LON_MAX)
].copy()

print(f"\n{'='*55}")
print(f"TERRAIN DOWNLOAD SUMMARY")
print(f"{'='*55}")
print(f"  Points: {len(df_terrain)}")
print(f"\nStatistics:")
print(df_terrain[['elevation','slope','TWI']].describe().round(2).to_string())

out_path = os.path.join(OUTPUT_DIR, 'terrain_tunisia.csv')
df_terrain.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"✅ Terrain processing complete!")
print(f"   → Next: run src/data/merge_dataset.py")
