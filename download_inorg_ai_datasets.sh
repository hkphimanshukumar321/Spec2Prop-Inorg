#!/usr/bin/env bash
set -euo pipefail

# download_inorg_ai_datasets.sh
# Purpose:
#   Download/open-cache the datasets for the Spec2Prop-Inorg project:
#   RRUFF Raman + chemistry, MLROD Raman, Matbench/Materials Project labels,
#   optional COD CIF files, tmQM, and tmQMg.
#
# Usage:
#   bash download_inorg_ai_datasets.sh [DATA_ROOT]
#
# Example:
#   bash download_inorg_ai_datasets.sh data
#
# Optional flags:
#   DOWNLOAD_COD=1 bash download_inorg_ai_datasets.sh data
#   DOWNLOAD_RRUFF_XRD=1 bash download_inorg_ai_datasets.sh data
#   DOWNLOAD_TMQM=0 DOWNLOAD_TMQMG=0 bash download_inorg_ai_datasets.sh data
#
# Notes:
#   - COD full CIF mirror is very large. It is skipped by default.
#   - Matbench is downloaded through matminer, the recommended programmatic loader.
#   - MLROD is obtained through the official companion GitHub package for the MLROD CNN work.
#   - Some datasets may have non-commercial / citation requirements. Check each source license before publishing.

DATA_ROOT="${1:-data}"

RAW_DIR="$DATA_ROOT/raw"
PROCESSED_DIR="$DATA_ROOT/processed"
SCRIPTS_DIR="$DATA_ROOT/scripts"
LOG_DIR="$DATA_ROOT/logs"

RRUFF_DIR="$RAW_DIR/rruff"
RRUFF_RAMAN_DIR="$RRUFF_DIR/raman"
RRUFF_CHEM_DIR="$RRUFF_DIR/chemistry"
RRUFF_XRD_DIR="$RRUFF_DIR/xrd"

MLROD_DIR="$RAW_DIR/mlrod"
MATBENCH_DIR="$RAW_DIR/matbench"
COD_DIR="$RAW_DIR/cod"
TMQM_DIR="$RAW_DIR/tmQM"
TMQMG_DIR="$RAW_DIR/tmQMg"

DOWNLOAD_RRUFF="${DOWNLOAD_RRUFF:-1}"
DOWNLOAD_RRUFF_XRD="${DOWNLOAD_RRUFF_XRD:-0}"
DOWNLOAD_MLROD="${DOWNLOAD_MLROD:-1}"
DOWNLOAD_MATBENCH="${DOWNLOAD_MATBENCH:-1}"
DOWNLOAD_COD="${DOWNLOAD_COD:-0}"
DOWNLOAD_TMQM="${DOWNLOAD_TMQM:-1}"
DOWNLOAD_TMQMG="${DOWNLOAD_TMQMG:-1}"
INSTALL_PY_DEPS="${INSTALL_PY_DEPS:-1}"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$SCRIPTS_DIR" "$LOG_DIR"
mkdir -p "$RRUFF_RAMAN_DIR" "$RRUFF_CHEM_DIR" "$RRUFF_XRD_DIR" "$MLROD_DIR" "$MATBENCH_DIR" "$COD_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Missing command: $1"
    echo "        Install it first and rerun this script."
    exit 1
  }
}

need_cmd curl
need_cmd unzip
need_cmd git
need_cmd python3

if [[ "$DOWNLOAD_COD" == "1" ]]; then
  need_cmd rsync
fi

download_file() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    echo "[SKIP] Already exists: $out"
  else
    echo "[GET ] $url"
    mkdir -p "$(dirname "$out")"
    curl -L --retry 3 --retry-delay 5 -C - -o "$out" "$url"
  fi
}

unzip_file() {
  local zipfile="$1"
  local outdir="$2"
  local marker="$outdir/.unzipped_$(basename "$zipfile").done"

  if [[ -f "$marker" ]]; then
    echo "[SKIP] Already unzipped: $zipfile"
  else
    echo "[UNZIP] $zipfile -> $outdir"
    mkdir -p "$outdir"
    unzip -q -o "$zipfile" -d "$outdir"
    touch "$marker"
  fi
}

clone_or_pull() {
  local repo="$1"
  local outdir="$2"

  if [[ -d "$outdir/.git" ]]; then
    echo "[PULL] $outdir"
    git -C "$outdir" pull --ff-only || echo "[WARN] Could not pull $outdir; keeping existing copy."
  else
    echo "[CLONE] $repo -> $outdir"
    git clone --depth 1 "$repo" "$outdir"
  fi
}

echo "============================================================"
echo "Spec2Prop-Inorg dataset downloader"
echo "DATA_ROOT=$DATA_ROOT"
echo "============================================================"

# --------------------------------------------------------------------
# 1) RRUFF Raman + chemistry data
# Direct ZIP folders from RRUFF:
#   https://www.rruff.net/zipped_data_files/raman/
#   https://www.rruff.net/zipped_data_files/chemistry/
# --------------------------------------------------------------------
if [[ "$DOWNLOAD_RRUFF" == "1" ]]; then
  echo
  echo "========== RRUFF Raman + chemistry =========="

  RRUFF_RAMAN_BASE="https://www.rruff.net/zipped_data_files/raman"
  RRUFF_CHEM_BASE="https://www.rruff.net/zipped_data_files/chemistry"

  # Main Raman spectral sets.
  RRUFF_RAMAN_ZIPS=(
    "LR-Raman.zip"
    "excellent_oriented.zip"
    "excellent_unoriented.zip"
    "fair_oriented.zip"
    "fair_unoriented.zip"
    "poor_unoriented.zip"
    "unrated_oriented.zip"
    "unrated_unoriented.zip"
  )

  for z in "${RRUFF_RAMAN_ZIPS[@]}"; do
    download_file "$RRUFF_RAMAN_BASE/$z" "$RRUFF_RAMAN_DIR/$z"
    unzip_file "$RRUFF_RAMAN_DIR/$z" "$RRUFF_RAMAN_DIR/extracted"
  done

  # Chemistry metadata / microprobe data.
  download_file "$RRUFF_CHEM_BASE/Microprobe_Data.zip" "$RRUFF_CHEM_DIR/Microprobe_Data.zip"
  unzip_file "$RRUFF_CHEM_DIR/Microprobe_Data.zip" "$RRUFF_CHEM_DIR/extracted"

  download_file "$RRUFF_CHEM_BASE/Reference_PDF.zip" "$RRUFF_CHEM_DIR/Reference_PDF.zip"
  unzip_file "$RRUFF_CHEM_DIR/Reference_PDF.zip" "$RRUFF_CHEM_DIR/extracted"
fi

# Optional RRUFF powder XRD data.
if [[ "$DOWNLOAD_RRUFF_XRD" == "1" ]]; then
  echo
  echo "========== RRUFF Powder XRD optional =========="

  RRUFF_POWDER_BASE="https://www.rruff.net/zipped_data_files/powder"
  RRUFF_POWDER_ZIPS=(
    "DIF.zip"
    "Refinement_Data.zip"
    "Refinement_Output_Data.zip"
    "XY_Processed.zip"
    "XY_RAW.zip"
  )

  for z in "${RRUFF_POWDER_ZIPS[@]}"; do
    download_file "$RRUFF_POWDER_BASE/$z" "$RRUFF_XRD_DIR/$z"
    unzip_file "$RRUFF_XRD_DIR/$z" "$RRUFF_XRD_DIR/extracted"
  done
else
  echo "[INFO] RRUFF XRD skipped. To include it: DOWNLOAD_RRUFF_XRD=1 bash $0 $DATA_ROOT"
fi

# --------------------------------------------------------------------
# 2) RamanSPy helper loader script
# RamanSPy can load RRUFF datasets using ramanspy.datasets.rruff("fair_oriented").
# We create a helper, but raw direct ZIP files above are the primary source.
# --------------------------------------------------------------------
cat > "$SCRIPTS_DIR/load_rruff_with_ramanspy.py" <<'PY'
from pathlib import Path
import pickle

try:
    import ramanspy
except ImportError as exc:
    raise SystemExit("Please install RamanSPy first: pip install ramanspy") from exc

out = Path("data/raw/rruff/ramanspy_cache")
out.mkdir(parents=True, exist_ok=True)

dataset_names = [
    "fair_oriented",
    "fair_unoriented",
    "excellent_oriented",
    "excellent_unoriented",
]

for name in dataset_names:
    print(f"[RamanSPy] Loading RRUFF subset: {name}")
    spectra = ramanspy.datasets.rruff(name)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(spectra, f)
    print(f"[OK] Saved: {out / f'{name}.pkl'}")
PY

# --------------------------------------------------------------------
# 3) MLROD Raman dataset companion package
# Official companion package:
#   https://github.com/GenTeML/Spec-CNN
# It contains code + MLROD example data folders used by the Raman CNN paper.
# --------------------------------------------------------------------
if [[ "$DOWNLOAD_MLROD" == "1" ]]; then
  echo
  echo "========== MLROD / Spec-CNN =========="
  clone_or_pull "https://github.com/GenTeML/Spec-CNN.git" "$MLROD_DIR/Spec-CNN"

  cat > "$MLROD_DIR/MLROD_SOURCE.txt" <<'TXT'
MLROD source notes:
- AHED page: https://ahed.nasa.gov/datasets/f5b6051bfeb18c5a7eaef6504582
- ODR landing page: https://www.odr.io/MLROD
- Companion GitHub package cloned here: https://github.com/GenTeML/Spec-CNN
- Zenodo software/data package: https://zenodo.org/records/7036374
TXT
fi

# --------------------------------------------------------------------
# 4) Matbench / Materials Project derived datasets
# Download through matminer.datasets.load_dataset.
# We cache both .pkl and lightweight formula-target CSV files.
# --------------------------------------------------------------------
if [[ "$DOWNLOAD_MATBENCH" == "1" ]]; then
  echo
  echo "========== Matbench / Materials Project derived datasets =========="

  if [[ "$INSTALL_PY_DEPS" == "1" ]]; then
    echo "[PIP] Installing Python packages needed for Matbench caching..."
    python3 -m pip install --upgrade pip >/dev/null
    python3 -m pip install -q matminer pandas numpy pymatgen
  fi

  cat > "$SCRIPTS_DIR/download_matbench.py" <<'PY'
from pathlib import Path
import pandas as pd
from matminer.datasets import load_dataset, get_all_dataset_info

out = Path("data/raw/matbench")
out.mkdir(parents=True, exist_ok=True)

datasets = [
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    "matbench_mp_e_form",
]

for name in datasets:
    print(f"[Matbench] Loading {name}")
    df = load_dataset(name)
    df.to_pickle(out / f"{name}.pkl")

    info = get_all_dataset_info(name)
    (out / f"{name}_info.txt").write_text(info, encoding="utf-8")

    # Create a lightweight merge-friendly table:
    # structure -> reduced formula + target columns.
    csv_df = pd.DataFrame()
    if "structure" in df.columns:
        csv_df["formula"] = df["structure"].map(lambda s: s.composition.formula)
        csv_df["reduced_formula"] = df["structure"].map(lambda s: s.composition.reduced_formula)
    elif "composition" in df.columns:
        csv_df["formula"] = df["composition"].astype(str)
        csv_df["reduced_formula"] = df["composition"].astype(str)

    for col in df.columns:
        if col not in {"structure", "composition"}:
            csv_df[col] = df[col]

    csv_df.to_csv(out / f"{name}_formula_targets.csv", index=False)
    print(f"[OK] {name}: rows={len(df)}")

print("[DONE] Matbench datasets cached.")
PY

  # Run from current directory so output path is DATA_ROOT-relative.
  # If DATA_ROOT is not "data", patch the script at runtime.
  if [[ "$DATA_ROOT" != "data" ]]; then
    sed -i.bak "s|Path(\"data/raw/matbench\")|Path(\"$MATBENCH_DIR\")|g" "$SCRIPTS_DIR/download_matbench.py"
  fi

  python3 "$SCRIPTS_DIR/download_matbench.py" | tee "$LOG_DIR/matbench_download.log"
fi

# --------------------------------------------------------------------
# 5) COD optional full CIF mirror
# COD full mirror can be very large.
# Use only if you have storage and time.
# --------------------------------------------------------------------
if [[ "$DOWNLOAD_COD" == "1" ]]; then
  echo
  echo "========== Crystallography Open Database optional CIF mirror =========="
  echo "[INFO] This can be large. Downloading via rsync..."
  mkdir -p "$COD_DIR/cif"
  rsync -av --progress rsync://www.crystallography.net/cif/ "$COD_DIR/cif/" | tee "$LOG_DIR/cod_rsync.log"
else
  echo "[INFO] COD skipped by default. To include it: DOWNLOAD_COD=1 bash $0 $DATA_ROOT"
fi

# --------------------------------------------------------------------
# 6) tmQM transition-metal complex dataset
# --------------------------------------------------------------------
if [[ "$DOWNLOAD_TMQM" == "1" ]]; then
  echo
  echo "========== tmQM transition-metal complexes =========="
  clone_or_pull "https://github.com/uiocompcat/tmQM.git" "$TMQM_DIR"

  cat > "$TMQM_DIR/TMQM_SOURCE.txt" <<'TXT'
tmQM source notes:
- GitHub: https://github.com/uiocompcat/tmQM
- Dataset contains transition-metal complexes, geometries, SMILES, electronic energy,
  dipole moment, metal charge, HOMO-LUMO gap, HOMO, LUMO, polarizability, and bond orders.
TXT
fi

# --------------------------------------------------------------------
# 7) tmQMg graph transition-metal complex dataset
# --------------------------------------------------------------------
if [[ "$DOWNLOAD_TMQMG" == "1" ]]; then
  echo
  echo "========== tmQMg graph transition-metal complexes =========="
  clone_or_pull "https://github.com/uiocompcat/tmQMg.git" "$TMQMG_DIR"

  cat > "$TMQMG_DIR/TMQMG_SOURCE.txt" <<'TXT'
tmQMg source notes:
- GitHub: https://github.com/uiocompcat/tmQMg
- Contains graph representations and properties for transition-metal complexes.
- For full graph archives, check data/tmQMg_graphs.md inside this repository.
TXT
fi

# --------------------------------------------------------------------
# 8) Create project manifest
# --------------------------------------------------------------------
cat > "$DATA_ROOT/DATASET_MANIFEST.md" <<EOF
# Spec2Prop-Inorg Dataset Manifest

Generated by: download_inorg_ai_datasets.sh

## Folder layout

- raw/rruff/raman/ : RRUFF Raman ZIP files + extracted spectra
- raw/rruff/chemistry/ : RRUFF microprobe/reference files
- raw/mlrod/Spec-CNN/ : MLROD companion CNN repository
- raw/matbench/ : cached Matbench datasets and formula-target CSVs
- raw/cod/ : optional COD CIF mirror, if DOWNLOAD_COD=1
- raw/tmQM/ : tmQM transition-metal complex dataset
- raw/tmQMg/ : tmQMg graph transition-metal complex dataset
- scripts/ : helper Python scripts

## Recommended next step

Build the processed Spec2Prop-Inorg table by:
1. Reading RRUFF/MLROD spectra.
2. Interpolating spectra to a fixed wavenumber grid.
3. Normalizing and baseline-correcting spectra.
4. Extracting mineral name and chemical formula.
5. Matching reduced_formula with Matbench formula-target CSVs.
6. Creating labels: mineral class, family class, mixture label, band-gap class, stability class.
EOF

echo
echo "============================================================"
echo "[DONE] Dataset download/cache script completed."
echo "Data root: $DATA_ROOT"
echo "Manifest: $DATA_ROOT/DATASET_MANIFEST.md"
echo "============================================================"
