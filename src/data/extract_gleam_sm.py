"""
extract_gleam_sm.py
===================
Extraction de la surface soil moisture (SMs) depuis les fichiers NetCDF
GLEAM v4.2a pour la Tunisie centrale (5 gouvernorats)

SOURCE  : GLEAM v4.2a — https://www.gleam.eu/
ACCÈS   : SFTP hydras.ugent.be port 2225 (compte gratuit requis)
RÉSOL.  : 0.05° × 0.05° (natif GLEAM)
PÉRIODE : 1980-2022 (référence WMO + étude)
FICHIERS: SMs_YYYY_GLEAM_v4.2a_MO.nc  (mensuels)

Article : Spatiotemporal SHAP Analysis for Agricultural Drought Prediction:
          A Multi-Source Machine Learning Framework in Semi-Arid Tunisia
Author  : Majdi Argoubi, University of Sousse

PRÉREQUIS :
  1. Créer un compte sur https://www.gleam.eu/#downloads
  2. Recevoir les credentials SFTP par email
  3. Modifier HOST, USERNAME, PASSWORD ci-dessous
  4. pip install paramiko netCDF4 xarray

STRUCTURE FICHIERS GLEAM :
  gleam_data/
    SMs_2001_GLEAM_v4.2a_MO.nc   ← SM mensuel 2001
    SMs_2002_GLEAM_v4.2a_MO.nc
    ...
    SMs_2022_GLEAM_v4.2a_MO.nc
    (+ années référence 1980-2000 si téléchargées pour SSMI)

SORTIE :
  gleam_data/*.nc                 ← fichiers bruts NetCDF
  → Ensuite lancer compute_ssmi.py pour calculer SSMI
"""

import paramiko
import os
import numpy as np
import sys

# ============================================================
# CONFIGURATION SFTP GLEAM v4.2a
# Remplacer par vos credentials reçus par email de gleam.eu
# ============================================================
HOST     = 'hydras.ugent.be'
PORT     = 2225
USERNAME = 'YOUR_GLEAM_USERNAME'   # ← À REMPLACER
PASSWORD = 'YOUR_GLEAM_PASSWORD'   # ← À REMPLACER

LOCAL_DIR = 'gleam_data'
os.makedirs(LOCAL_DIR, exist_ok=True)

# Années à télécharger
# - Référence WMO 1980-2010 (pour calcul SSMI)
# - Étude 2001-2022
# Si vous avez déjà les années 2001-2022, télécharger aussi 1980-2000
# pour que la référence climatologique soit correcte
YEARS_REF   = list(range(1980, 2001))   # 21 ans référence
YEARS_STUDY = list(range(2001, 2023))   # 22 ans étude
ALL_YEARS   = sorted(set(YEARS_REF + YEARS_STUDY))

# Dossier GLEAM sur le serveur SFTP
# (peut varier selon version — explorer avec sftp.listdir('.'))
GLEAM_REMOTE_DIR = 'v4.2a/monthly'

# ============================================================
# CONNEXION SFTP
# ============================================================
print(f"Connecting to GLEAM SFTP server ({HOST}:{PORT})...")
print(f"  Username: {USERNAME}")

if USERNAME == 'YOUR_GLEAM_USERNAME':
    print("\n⚠️  ERROR: Please set your GLEAM credentials in the script.")
    print("   Register at: https://www.gleam.eu/#downloads")
    print("   Then replace USERNAME and PASSWORD in this script.")
    sys.exit(1)

try:
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print("✅ Connected!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check credentials at https://www.gleam.eu/#downloads")
    print("  2. Check firewall allows outbound port 2225")
    print("  3. Try: ssh -p 2225 YOUR_USERNAME@hydras.ugent.be")
    sys.exit(1)

# ============================================================
# EXPLORER LA STRUCTURE DU SERVEUR
# ============================================================
print("\nExploring server structure...")
try:
    top_dirs = sftp.listdir('.')
    print(f"  Root: {top_dirs}")

    for d in top_dirs:
        try:
            sub = sftp.listdir(d)
            print(f"  {d}/: {sub[:5]}...")
        except:
            pass
except Exception as e:
    print(f"  ⚠️  Cannot list root: {e}")

# ============================================================
# TÉLÉCHARGEMENT DES FICHIERS NetCDF MENSUELS
# ============================================================
print(f"\nDownloading SMs monthly files for {len(ALL_YEARS)} years...")
print(f"  Remote dir : {GLEAM_REMOTE_DIR}")
print(f"  Local dir  : {LOCAL_DIR}")
print(f"  Years      : {ALL_YEARS[0]}–{ALL_YEARS[-1]}\n")

downloaded = 0
skipped    = 0
errors     = 0

for year in ALL_YEARS:
    fname     = f'SMs_{year}_GLEAM_v4.2a_MO.nc'
    remote    = f'{GLEAM_REMOTE_DIR}/{fname}'
    local     = os.path.join(LOCAL_DIR, fname)
    local_tmp = local + '.tmp'

    # Vérifier si déjà téléchargé
    if os.path.exists(local):
        size = os.path.getsize(local)
        if size > 100_000:   # > 100 KB = fichier valide
            print(f"  ✅ {fname} already exists ({size/1e6:.1f} MB) — skipped")
            skipped += 1
            continue

    try:
        # Obtenir la taille distante
        remote_size = sftp.stat(remote).st_size
        print(f"  Downloading {fname} ({remote_size/1e6:.1f} MB)...",
              end='', flush=True)

        sftp.get(remote, local_tmp)
        os.rename(local_tmp, local)

        local_size = os.path.getsize(local)
        print(f" ✅ ({local_size/1e6:.1f} MB)")
        downloaded += 1

    except FileNotFoundError:
        # Essayer des variantes de nom
        alternatives = [
            f'v4.2a/monthly/SMs_{year}_GLEAM_v4.2a_MO.nc',
            f'GLEAM_v4.2a/SMs_{year}_GLEAM_v4.2a_MO.nc',
            f'monthly/SMs_{year}_GLEAM_v4.2a_MO.nc',
        ]
        found = False
        for alt in alternatives:
            try:
                sftp.stat(alt)
                sftp.get(alt, local_tmp)
                os.rename(local_tmp, local)
                print(f"  ✅ {fname} found at {alt}")
                downloaded += 1
                found = True
                break
            except:
                continue
        if not found:
            print(f"  ⚠️  {fname} not found on server")
            errors += 1

    except Exception as e:
        print(f"\n  ❌ Error downloading {fname}: {e}")
        if os.path.exists(local_tmp):
            os.remove(local_tmp)
        errors += 1

sftp.close()
transport.close()

# ============================================================
# VÉRIFICATION DES FICHIERS TÉLÉCHARGÉS
# ============================================================
print(f"\n{'='*55}")
print(f"DOWNLOAD SUMMARY")
print(f"{'='*55}")
print(f"  Downloaded : {downloaded}")
print(f"  Skipped    : {skipped} (already present)")
print(f"  Errors     : {errors}")

nc_files = [f for f in os.listdir(LOCAL_DIR) if f.endswith('.nc')]
print(f"\n  Files in gleam_data/: {len(nc_files)}")
for f in sorted(nc_files)[:5]:
    size = os.path.getsize(os.path.join(LOCAL_DIR, f))
    print(f"    {f}: {size/1e6:.1f} MB")
if len(nc_files) > 5:
    print(f"    ... and {len(nc_files)-5} more")

# ============================================================
# VÉRIFICATION RAPIDE D'UN FICHIER
# ============================================================
print("\nVerifying first NetCDF file...")
try:
    import xarray as xr
    sample = os.path.join(LOCAL_DIR, sorted(nc_files)[0])
    ds = xr.open_dataset(sample)
    print(f"  ✅ {os.path.basename(sample)}")
    print(f"     Variables : {list(ds.data_vars)}")
    print(f"     Dimensions: {dict(ds.dims)}")
    print(f"     Lat range : {float(ds.lat.min()):.2f}–{float(ds.lat.max()):.2f}")
    print(f"     Lon range : {float(ds.lon.min()):.2f}–{float(ds.lon.max()):.2f}")
    ds.close()
except ImportError:
    print("  ⚠️  xarray not installed: pip install xarray netCDF4")
except Exception as e:
    print(f"  ⚠️  Verification error: {e}")

print(f"\n✅ GLEAM extraction complete!")
print(f"   → Next: run src/data/compute_ssmi.py")
