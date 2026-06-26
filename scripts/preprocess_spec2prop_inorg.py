import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Spec2Prop-Inorg: Preprocessing Pipeline
========================================
Transforms raw RRUFF, Matbench, tmQM, and tmQMg datasets into a unified,
ML-ready dataset for inorganic compound spectra-to-property prediction.

Generates 4 distinct subsets (both .pkl with spectra and .csv metadata-only).
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FMT = "[%(asctime)s] %(levelname)-7s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("spec2prop")

# ---------------------------------------------------------------------------
# Constants & Dictionaries
# ---------------------------------------------------------------------------
RAMAN_INTERP_POINTS = 2048
RAMAN_WN_MIN = 100.0
RAMAN_WN_MAX = 4000.0

XRD_INTERP_POINTS = 2048
XRD_2THETA_MIN = 5.0
XRD_2THETA_MAX = 90.0

RAW_COMMON_MINERALS = {
    "Diamond": "C",
    "Graphite": "C",
    "Moissanite": "SiC",
    "Yttriumaluminumgarnet": "Y3Al5O12",
    "Yttrium Aluminum Garnet": "Y3Al5O12",
    "YAG": "Y3Al5O12",
    "Calaverite": "AuTe2",
    "Silver": "Ag",
    "Cubiczirconia": "ZrO2",
    "Cubic Zirconia": "ZrO2",
    "Awaruite": "Ni3Fe",
    "Osmium": "Os",
    "Silicon": "Si",
    "Corundum": "Al2O3",
    "Rutile": "TiO2",
    "Anatase": "TiO2",
    "Hematite": "Fe2O3",
    "Magnetite": "Fe3O4",
    "Quartz": "SiO2",
    "Tridymite": "SiO2",
    "Cristobalite": "SiO2",
    "Coesite": "SiO2",
    "Opal": "SiO2·nH2O",
    "Moganite": "SiO2",
    "Chalcedony": "SiO2",
    "Calcite": "CaCO3",
    "Gypsum": "CaSO4·2H2O",
    "Forsterite": "Mg2SiO4",
    "Fluorite": "CaF2",
    "Pyrite": "FeS2",
    "Apatite": "Ca5(PO4)3F",
    "Barite": "BaSO4",
    "Dolomite": "CaMg(CO3)2",
    "Goethite": "FeO(OH)",
    "Ilmenite": "FeTiO3",
    "Sphalerite": "ZnS",
    "Galena": "PbS",
    "Chalcopyrite": "CuFeS2",
    "Gold": "Au",
    "Copper": "Cu",
    "Sulfur": "S",
    "Platinum": "Pt",
    "Iron": "Fe",
    "Nickel": "Ni"
}

def robust_mineral_key(name: str) -> str:
    """Lowercase, remove spaces/punctuation."""
    return re.sub(r'[\s\-_]', '', name).lower()

COMMON_MINERALS = {robust_mineral_key(k): v for k, v in RAW_COMMON_MINERALS.items()}

# ============================================================================
# 0. Utility Functions
# ============================================================================

def clean_mineral_name(name: str) -> str:
    """Removes common noise from mineral filenames to get the clean name."""
    name = re.sub(r'__.*', '', name)
    name = re.sub(r'[-_]\d+.*', '', name)
    name = name.replace('_Raman', '').replace('Raman', '')
    name = name.replace('oriented', '').replace('unoriented', '')
    name = name.replace('excellent', '').replace('fair', '').replace('poor', '').replace('unrated', '')
    name = name.replace('Processed', '').replace('RAW', '').replace('Broad_Scan', '')
    return name.strip('_ \t\n-')


def get_formula_flags(formula: str) -> dict:
    """Checks the formula string for specific formatting issues."""
    if not isinstance(formula, str):
        return {}
    
    flags = {
        "has_vacancy_symbol": bool(re.search(r'\[box\]|□', formula, re.IGNORECASE)),
        "has_mixed_occupancy": bool(re.search(r'\([A-Za-z]+,[A-Za-z]+.*?\)', formula)),
        "has_variable_formula": bool(re.search(r'\d+-\d+', formula) or re.search(r'[a-z]H2O', formula) or "n" in formula and "H2O" in formula),
        "has_html_tag": bool(re.search(r'<[^>]+>', formula)),
        "has_hydrate": bool(re.search(r'·|\.|\bH2O\b', formula)),
    }
    return flags

def clean_display_formula(formula: str) -> str:
    """Removes HTML tags for display, but keeps commas/variables."""
    if not isinstance(formula, str): return ""
    return re.sub(r'<[^>]+>', '', formula)

def normalize_formula(formula: str) -> str:
    """Clean up formulas (unicode subscripts, dots, spaces, rruff formatting)."""
    if not isinstance(formula, str):
        return ""
    
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    formula = formula.translate(subscripts)
    formula = re.sub(r'\^.*?\^', '', formula)
    formula = formula.replace("_", "")
    formula = formula.replace(" ", "").replace("·", "").replace(".", "")
    return re.sub(r'<[^>]+>', '', formula)


def get_reduced_formula(formula: str) -> str:
    """Uses pymatgen to get reduced formula after stripping parentheses if needed."""
    try:
        from pymatgen.core import Composition
        clean_for_pymatgen = re.sub(r'\(.*?,.*?\)', '', formula)
        comp = Composition(clean_for_pymatgen)
        return comp.reduced_formula
    except Exception:
        return formula


def classify_family_by_formula(formula_str: str, elements: set, mineral_name: str) -> str:
    """Classify family using the strict 15-tier hierarchy."""
    f = formula_str.upper()
    name_clean = mineral_name.lower()
    
    # Check elements safely
    metals = {"Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U"}
    has_metal = any(e in metals for e in elements)
    all_metals = elements and all(e in metals for e in elements)
    
    # 1. Organic
    if "C" in elements and "H" in elements and "O" not in elements:
        return "Organic"
        
    # 2. Elemental/Native
    native_names = {"diamond", "graphite", "silver", "gold", "copper", "sulfur", "silicon", "osmium", "platinum", "iron", "nickel"}
    if len(elements) == 1 or name_clean in native_names:
        return "Elemental/Native"
        
    # 3. Silicate
    if ("SI" in f and "O" in f and has_metal) or "silicate" in name_clean:
        return "Silicate"
        
    # 4. Carbonate
    if "CO3" in f or "carbonate" in name_clean:
        return "Carbonate"
        
    # 5. Sulfate
    if "SO4" in f or "sulfate" in name_clean:
        return "Sulfate"
        
    # 6. Phosphate
    if "PO4" in f or "phosphate" in name_clean or "apatite" in name_clean:
        return "Phosphate"
        
    # 7. Borate
    if ("B" in elements and "O" in elements) or "borate" in name_clean:
        return "Borate"
        
    # 8. Halide
    if any(h in elements for h in ("F","Cl","Br","I")) and has_metal and not ("PO4" in f or "apatite" in name_clean):
        return "Halide"
        
    # 9. Sulfide
    if "S" in elements and "O" not in elements:
        return "Sulfide"
        
    # 10. Oxide
    if "O" in elements and has_metal:
        return "Oxide"
        
    # 11. Carbide
    if "C" in elements and ("SI" in f or has_metal) and not ("H" in elements and "O" not in elements):
        return "Carbide"
        
    # 12. Telluride/Selenide
    if ("TE" in f or "SE" in f) and has_metal and "O" not in elements:
        return "Telluride/Selenide"
        
    # 13. Intermetallic/Alloy
    if all_metals and len(elements) > 1:
        return "Intermetallic/Alloy"
        
    # 14. Mixed/Other
    if formula_str:
        return "Mixed/Other"
    
    # 15. Unknown
    return "Unknown"


# ============================================================================
# 1. RRUFF Raman Processing
# ============================================================================

def parse_rruff_filename(filepath: str) -> dict:
    basename = os.path.splitext(os.path.basename(filepath))[0]
    r_id_match = re.search(r'(R\d{5,7})', basename)
    rruff_id = r_id_match.group(1) if r_id_match else ""
    mineral_name = clean_mineral_name(basename)
    
    info = {
        "mineral_name": mineral_name,
        "rruff_id":     rruff_id,
        "data_type":    "",
        "laser_nm":     "",
        "raw_filename": os.path.basename(filepath),
    }

    if "Raman_Data_Processed" in basename: info["data_type"] = "Processed"
    elif "Raman_Data_RAW" in basename or "Raman_Data_Raw" in basename: info["data_type"] = "RAW"
    elif "Broad_Scan" in basename: info["data_type"] = "Broad_Scan"
        
    laser_match = re.search(r'__(\d{3}(?:-\d+)?)__', basename)
    if laser_match: info["laser_nm"] = laser_match.group(1).replace("-", ".")

    return info

def parse_rruff_spectrum(filepath: str) -> tuple:
    wavenumbers, intensities = [], []
    ideal_chem = ""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("##IDEAL CHEMISTRY="):
                    ideal_chem = line.split("=", 1)[1].strip()
                if not line or line.startswith("##") or line.startswith("#"): continue
                for sep in (",", "\t", None):
                    tokens = line.split(sep) if sep else line.split()
                    if len(tokens) >= 2:
                        try:
                            w1 = float(tokens[0].strip())
                            i1 = float(tokens[1].strip())
                            wavenumbers.append(w1)
                            intensities.append(i1)
                            break
                        except ValueError:
                            continue
    except Exception:
        return None, None, ""
    if len(wavenumbers) < 10: return None, None, ""
    return np.array(wavenumbers), np.array(intensities), ideal_chem

def interpolate_spectrum(wn, intensity, n_points, wn_min, wn_max):
    sort_idx = np.argsort(wn)
    wn, intensity = wn[sort_idx], intensity[sort_idx]
    mask = (wn >= wn_min) & (wn <= wn_max)
    wn, intensity = wn[mask], intensity[mask]
    if len(wn) < 5: return None
    _, unique_idx = np.unique(wn, return_index=True)
    wn, intensity = wn[unique_idx], intensity[unique_idx]
    if len(wn) < 5: return None

    f = interp1d(wn, intensity, kind="linear", bounds_error=False, fill_value=0.0)
    grid = np.linspace(wn_min, wn_max, n_points)
    resampled = f(grid)
    
    vmin, vmax = resampled.min(), resampled.max()
    if vmax - vmin > 1e-10:
        resampled = (resampled - vmin) / (vmax - vmin)
    else:
        resampled = np.zeros_like(resampled)
    return resampled

def process_rruff_raman(raman_dir: str, n_points: int, wn_min: float, wn_max: float) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Processing RRUFF Raman spectra from: %s", raman_dir)
    files = glob.glob(os.path.join(raman_dir, "*.txt"))
    
    # Group by (rruff_id, mineral_name) — select best variant per group
    groups = defaultdict(list)
    for fp in files:
        info = parse_rruff_filename(fp)
        key = (info["rruff_id"], info["mineral_name"])
        groups[key].append((info, fp))

    priority = {"Processed": 0, "RAW": 1, "Broad_Scan": 2, "": 3}

    # Count how many raw files exist per rruff_id (for duplicate flag)
    rruff_id_file_counts = defaultdict(int)
    for (r_id, _), entries in groups.items():
        if r_id:
            rruff_id_file_counts[r_id] += len(entries)

    # Select best variant per group
    selected = []
    for key, entries in groups.items():
        entries.sort(key=lambda x: priority.get(x[0]["data_type"], 99))
        selected.append((key, entries[0]))

    rows = []
    sample_id_counter = 0
    for (r_id, min_name), (info, fp) in selected:
        wn, intensity, ideal_chem = parse_rruff_spectrum(fp)
        if wn is None: continue
        resampled = interpolate_spectrum(wn, intensity, n_points, wn_min, wn_max)
        if resampled is None: continue

        sid = r_id if r_id else f"NO_ID_{sample_id_counter}"
        if not r_id:
            sample_id_counter += 1
        group_size = rruff_id_file_counts.get(r_id, 1) if r_id else 1

        row = {
            "sample_id":     sid,
            "rruff_id":      r_id,
            "mineral_name":  info["mineral_name"],
            "laser_nm":      info["laser_nm"],
            "data_type":     info["data_type"],
            "source_file":   info["raw_filename"],
            "ideal_chem":    ideal_chem,
            "duplicate_rruff_id":  group_size > 1,
            "duplicate_group_size": group_size,
            "measurement_variant_id": 0,
        }
        for j, val in enumerate(resampled):
            row[f"raman_{j}"] = val
        rows.append(row)

    # Deduplicate: if multiple groups produced the same sample_id (same rruff_id,
    # different mineral_name variants), keep first and rename subsequent.
    seen_ids = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in seen_ids:
            seen_ids[sid] += 1
            row["sample_id"] = f"{sid}_v{seen_ids[sid]}"
        else:
            seen_ids[sid] = 0

    log.info("Raman done: %d spectra selected (from %d raw files)", len(rows), len(files))
    return pd.DataFrame(rows)


# ============================================================================
# 2. RRUFF XRD Processing
# ============================================================================

def process_rruff_xrd(xrd_dir: str, n_points: int) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Processing RRUFF XRD patterns from: %s", xrd_dir)
    files = glob.glob(os.path.join(xrd_dir, "*Xray_Data_XY_Processed*.txt"))
    
    # Group by rruff_id, select first valid pattern per ID
    xrd_by_id = defaultdict(list)
    for fp in files:
        basename = os.path.basename(fp)
        r_match = re.search(r'(R\d{5,7})', basename)
        r_id = r_match.group(1) if r_match else ""
        if r_id:
            xrd_by_id[r_id].append(fp)

    rows = []
    total_raw = 0
    for r_id, fps in xrd_by_id.items():
        total_raw += len(fps)
        fps.sort()  # deterministic selection
        # Try each file until one parses successfully
        for fp in fps:
            wn, intensity, _ = parse_rruff_spectrum(fp)
            if wn is None: continue
            resampled = interpolate_spectrum(wn, intensity, n_points, XRD_2THETA_MIN, XRD_2THETA_MAX)
            if resampled is None: continue

            row = {
                "rruff_id": r_id,
                "xrd_variant_count": len(fps),
                "selected_xrd_file": os.path.basename(fp),
            }
            for j, val in enumerate(resampled):
                row[f"xrd_{j}"] = val
            rows.append(row)
            break  # one XRD pattern per rruff_id

    log.info("XRD done: %d unique rruff_ids selected (from %d raw files)", len(rows), total_raw)

    df_xrd = pd.DataFrame(rows)
    # Assert uniqueness: one row per rruff_id
    if not df_xrd.empty:
        assert df_xrd["rruff_id"].is_unique, \
            f"XRD table has duplicate rruff_ids: {df_xrd[df_xrd.duplicated('rruff_id')]['rruff_id'].tolist()}"
    return df_xrd


# ============================================================================
# 3. Chemistry Metadata
# ============================================================================

def process_rruff_chemistry(chem_dir: str) -> dict:
    log.info("=" * 60)
    log.info("Processing RRUFF Chemistry from: %s", chem_dir)
    files = glob.glob(os.path.join(chem_dir, "*.xls*"))
    
    chem_data = {}
    try:
        from pymatgen.core import Element
        valid_elements = {e.symbol for e in Element}
    except:
        valid_elements = set()

    for fp in files:
        basename = os.path.basename(fp)
        r_match = re.search(r'(R\d{5,7})', basename)
        if not r_match: continue
        r_id = r_match.group(1)
        
        try:
            engine = "openpyxl" if fp.endswith(".xlsx") else "xlrd"
            df = pd.read_excel(fp, engine=engine, header=None)
            elements_found = set()
            for col in df.columns:
                for val in df[col].dropna().astype(str):
                    matches = re.findall(r'\b([A-Z][a-z]?)\b', str(val))
                    if valid_elements:
                        matches = [m for m in matches if m in valid_elements]
                    elements_found.update(matches)
            chem_data[r_id] = ",".join(sorted(elements_found))
        except Exception:
            pass

    log.info("Chemistry mapping built for %d RRUFF IDs", len(chem_data))
    return chem_data


# ============================================================================
# 4. Matbench Property Matching
# ============================================================================

def load_matbench(matbench_dir: str) -> dict:
    log.info("=" * 60)
    log.info("Loading and aggregating Matbench tables from: %s", matbench_dir)
    lookup = defaultdict(dict)
    
    files_targets = {
        "gap":     "matbench_mp_gap_formula_targets.csv",
        "is_metal":"matbench_mp_is_metal_formula_targets.csv",
        "e_form":  "matbench_mp_e_form_formula_targets.csv",
    }

    for prop, fname in files_targets.items():
        fpath = os.path.join(matbench_dir, fname)
        if not os.path.isfile(fpath): continue
        df = pd.read_csv(fpath)
        target_col = [c for c in df.columns if c not in ("formula", "reduced_formula")][0]
        
        grouped = df.groupby("reduced_formula")[target_col]
        for rf, vals in grouped:
            if prop == "is_metal":
                lookup[rf]["is_metal"] = vals.mode().iloc[0] if not vals.mode().empty else vals.iloc[0]
                lookup[rf]["is_metal_match_count"] = int(vals.count())
            elif prop == "gap":
                lookup[rf]["band_gap_eV"] = float(vals.median())
                lookup[rf]["band_gap_min_eV"] = float(vals.min())
                lookup[rf]["band_gap_max_eV"] = float(vals.max())
                lookup[rf]["band_gap_match_count"] = int(vals.count())
            elif prop == "e_form":
                lookup[rf]["formation_energy"] = float(vals.median())
                lookup[rf]["formation_energy_min"] = float(vals.min())
                lookup[rf]["formation_energy_match_count"] = int(vals.count())

    log.info("Matbench lookup built: %d unique reduced formulas", len(lookup))
    return lookup


# ============================================================================
# 5. tmQM / tmQMg Auxiliary Data
# ============================================================================

def load_tmqm(tmqm_dir: str) -> pd.DataFrame:
    fpath = os.path.join(tmqm_dir, "tmQM", "tmQM_y.csv")
    if os.path.isfile(fpath): return pd.read_csv(fpath, sep=";")
    return pd.DataFrame()

def load_tmqmg(tmqmg_dir: str) -> pd.DataFrame:
    fpath = os.path.join(tmqmg_dir, "data", "tmQMg_properties_and_targets.csv")
    if os.path.isfile(fpath): return pd.read_csv(fpath)
    return pd.DataFrame()


# ============================================================================
# 6. Master Assembly
# ============================================================================

def build_master_table(df_raman: pd.DataFrame, df_xrd: pd.DataFrame, 
                       chem_map: dict, matbench_lookup: dict) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Building master metadata table")

    if df_raman.empty: return pd.DataFrame()
    
    master = df_raman[["sample_id", "rruff_id", "mineral_name", "laser_nm", "data_type", "source_file", "ideal_chem", "duplicate_rruff_id", "duplicate_group_size", "measurement_variant_id"]].copy()

    # Link XRD
    xrd_ids = set(df_xrd["rruff_id"]) if not df_xrd.empty else set()
    master["has_xrd"] = master["rruff_id"].isin(xrd_ids) & (master["rruff_id"] != "")
    master["xrd_file_path"] = master.apply(lambda row: row["source_file"].replace("Raman", "Xray_Data_XY") if row["has_xrd"] else "", axis=1)

    master["source_dataset"] = "RRUFF"
    master["raw_file_path"] = master["source_file"]
    master["compound_name"] = master["mineral_name"]
    master["is_mixture"] = False
    master["mixture_components"] = ""
    master["quality_flags"] = ""

    # -------------------------------------------------------------
    # Evaluate formulas and chemistry
    # -------------------------------------------------------------
    families, sub_families, reduced_forms, norm_forms = [], [], [], []
    disp_forms, orig_forms, n_elems, avg_ens, avg_masses = [], [], [], [], []
    
    # Flags
    f_vacancy, f_mixed, f_variable, f_html, f_hydrate = [], [], [], [], []
    f_success, f_eligible = [], []
    f_exclude_inorg = []
    
    elements_list = []
    c_tm, c_3d, c_4d, c_5d = [], [], [], []

    tm_3d = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"}
    tm_4d = {"Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd"}
    tm_5d = {"La","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
    all_tm = tm_3d | tm_4d | tm_5d
    
    obvious_organics = ["pyrimethamine", "polystyrene", "paracetamol", "magnesiumstearate", "organic", "polymer", "pharmaceutical"]

    for _, row in master.iterrows():
        min_name = row["mineral_name"]
        r_id = row["rruff_id"]
        ideal_chem = row.get("ideal_chem", "")
        
        elems_str = chem_map.get(r_id, "")
        elements = set(elems_str.split(",")) if elems_str else set()
        
        raw_formula = ideal_chem
        if not raw_formula:
            raw_formula = COMMON_MINERALS.get(robust_mineral_key(min_name), "")
            
        orig_forms.append(raw_formula)
        flags = get_formula_flags(raw_formula)
        
        f_vacancy.append(flags.get("has_vacancy_symbol", False))
        f_mixed.append(flags.get("has_mixed_occupancy", False))
        f_variable.append(flags.get("has_variable_formula", False))
        f_html.append(flags.get("has_html_tag", False))
        f_hydrate.append(flags.get("has_hydrate", False))
        
        disp_formula = clean_display_formula(raw_formula)
        disp_forms.append(disp_formula)
        
        norm_formula = normalize_formula(raw_formula)
        norm_forms.append(norm_formula)
        
        if norm_formula:
            matches = re.findall(r'([A-Z][a-z]?)', norm_formula)
            elements.update(matches)

        reduced_formula = get_reduced_formula(norm_formula)
        reduced_forms.append(reduced_formula)
        
        num_e, avg_en, avg_mass = np.nan, np.nan, np.nan
        parse_success = False
        try:
            clean_for_pymatgen = re.sub(r'\(.*?,.*?\)', '', norm_formula)
            from pymatgen.core import Composition
            comp = Composition(clean_for_pymatgen)
            comp_elems = [str(e) for e in comp.elements]
            if comp_elems:
                elements.update(comp_elems)
                num_e = len(comp_elems)
                en_vals = [e.X for e in comp.elements if hasattr(e, "X") and e.X is not None]
                if en_vals: avg_en = np.mean(en_vals)
                mass_vals = [float(e.atomic_mass) for e in comp.elements]
                if mass_vals: avg_mass = np.mean(mass_vals)
                parse_success = True
        except Exception:
            num_e = len(elements) if elements else np.nan
            
        f_success.append(parse_success)
        
        is_eligible = parse_success and not (
            flags.get("has_variable_formula") or 
            flags.get("has_vacancy_symbol") or 
            flags.get("has_mixed_occupancy") or 
            flags.get("has_html_tag")
        )
        
        family = classify_family_by_formula(norm_formula, elements, min_name)
        
        # Silica handling
        sub_family = ""
        is_silica = (reduced_formula == "SiO2" or min_name.lower() in ["quartz", "tridymite", "cristobalite", "coesite", "opal", "moganite", "chalcedony"])
        if is_silica:
            family = "Silicate"
            sub_family = "Silica"
            if min_name.lower() == "opal":
                flags["has_hydrate"] = True
                f_hydrate[-1] = True
                flags["has_variable_formula"] = True
                f_variable[-1] = True
                # Opal not eligible unless clean SiO2 fallback used
                if norm_formula != "SiO2":
                    is_eligible = False

        families.append(family)
        sub_families.append(sub_family)
        f_eligible.append(is_eligible)

        elements_list.append(",".join(sorted(elements)))
        c_tm.append(any(e in all_tm for e in elements))
        c_3d.append(any(e in tm_3d for e in elements))
        c_4d.append(any(e in tm_4d for e in elements))
        c_5d.append(any(e in tm_5d for e in elements))

        is_obvious_organic = any(o in min_name.lower() for o in obvious_organics)
        f_exclude_inorg.append(family == "Organic" or is_obvious_organic)
        n_elems.append(num_e)
        avg_ens.append(avg_en)
        avg_masses.append(avg_mass)

    master["chemical_family"] = families
    master["sub_family"] = sub_families
    master["original_formula"] = orig_forms
    master["display_formula_clean"] = disp_forms
    master["normalized_formula"] = norm_forms
    master["reduced_formula"] = reduced_forms
    master["elements"] = elements_list
    
    master["contains_transition_metal"] = c_tm
    master["contains_3d_transition_metal"] = c_3d
    master["contains_4d_transition_metal"] = c_4d
    master["contains_5d_transition_metal"] = c_5d
    
    master["has_vacancy_symbol"] = f_vacancy
    master["has_mixed_occupancy"] = f_mixed
    master["has_variable_formula"] = f_variable
    master["has_html_tag"] = f_html
    master["has_hydrate"] = f_hydrate
    master["formula_parse_success"] = f_success
    master["eligible_for_matbench_matching"] = f_eligible
    master["exclude_from_inorganic_main"] = f_exclude_inorg
    
    master["n_elements"] = n_elems
    master["avg_electronegativity"] = avg_ens
    master["avg_atomic_mass"] = avg_masses

    # -------------------------------------------------------------
    # Link Matbench
    # -------------------------------------------------------------
    log.info("Matching Matbench properties by reduced_formula...")
    mb_cols = defaultdict(list)
    mb_keys = [
        "band_gap_eV", "band_gap_min_eV", "band_gap_max_eV", "band_gap_match_count",
        "is_metal", "is_metal_match_count",
        "formation_energy", "formation_energy_min", "formation_energy_match_count"
    ]
    
    bg_class = []
    fe_class = []
    match_success = []
    
    for _, row in master.iterrows():
        rf = row["reduced_formula"]
        eligible = row["eligible_for_matbench_matching"]
        
        props = matbench_lookup.get(rf, {}) if eligible else {}
        for k in mb_keys:
            mb_cols[k].append(props.get(k, np.nan))
            
        gap = props.get("band_gap_eV", np.nan)
        if pd.isna(gap):
            bg_class.append(np.nan)
        elif gap <= 0.1: bg_class.append("metal_like")
        elif gap <= 1.5: bg_class.append("narrow_gap")
        elif gap <= 3.0: bg_class.append("semiconductor")
        else: bg_class.append("wide_gap_insulator")
            
        fe = props.get("formation_energy", np.nan)
        if pd.isna(fe):
            fe_class.append(np.nan)
        elif fe < -2.0: fe_class.append("highly_stable")
        elif fe < -0.5: fe_class.append("stable")
        elif fe < 0: fe_class.append("marginally_stable")
        else: fe_class.append("unstable")
        
        match_success.append(eligible and bool(props))
            
    for k in mb_keys:
        master[k] = mb_cols[k]
    master["band_gap_class"] = bg_class
    master["formation_energy_class"] = fe_class
    master["matbench_match_success"] = match_success

    master.drop(columns=["ideal_chem"], inplace=True)
    return master


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--wn-min", type=float, default=100.0)
    parser.add_argument("--wn-max", type=float, default=4000.0)
    parser.add_argument("--normalization", default="max")
    parser.add_argument("--baseline-correction", default="true")
    parser.add_argument("--smoothing", default="true")
    args = parser.parse_args()

    data_root, out_dir = Path(args.data_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    processing_log = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Process
    df_raman = process_rruff_raman(
        str(data_root / "raw" / "rruff" / "raman" / "extracted"),
        args.n_points, args.wn_min, args.wn_max
    )
    df_xrd = process_rruff_xrd(
        str(data_root / "raw" / "rruff" / "xrd" / "extracted"),
        args.n_points
    )
    chem_map = process_rruff_chemistry(str(data_root / "raw" / "rruff" / "chemistry" / "extracted"))
    matbench_lookup = load_matbench(str(data_root / "raw" / "matbench"))
    df_master = build_master_table(df_raman, df_xrd, chem_map, matbench_lookup)

    # tmQM / tmQMg
    df_tmqm = load_tmqm(str(data_root / "raw" / "tmQM"))
    if not df_tmqm.empty:
        df_tmqm[df_tmqm["SMILES"].notna()].to_csv(out_dir / "tmqm_with_smiles.csv", index=False)
        df_tmqm[df_tmqm["SMILES"].isna()].to_csv(out_dir / "tmqm_without_smiles_but_properties.csv", index=False)
        
    df_tmqmg = load_tmqmg(str(data_root / "raw" / "tmQMg"))
    if not df_tmqmg.empty:
        smiles_col = "smiles" if "smiles" in df_tmqmg.columns else df_tmqmg.columns[0]
        df_tmqmg[df_tmqmg[smiles_col].notna()].to_csv(out_dir / "tmqmg_with_smiles.csv", index=False)
        df_tmqmg[df_tmqmg[smiles_col].isna()].to_csv(out_dir / "tmqmg_without_smiles_but_properties.csv", index=False)

    # -------------------------------------------------------------
    # Subsets & Saving
    # -------------------------------------------------------------
    log.info("=" * 60)
    log.info("Generating and saving distinct subset versions...")
    
    spec_cols = [c for c in df_master.columns if c.startswith("raman_")]
    meta_cols = [
        "sample_id", "source_dataset", "raw_file_path", "rruff_id", "mineral_name", "compound_name",
        "original_formula", "display_formula_clean", "normalized_formula", "reduced_formula",
        "chemical_family", "sub_family", "elements", "contains_transition_metal",
        "contains_3d_transition_metal", "contains_4d_transition_metal", "contains_5d_transition_metal",
        "is_mixture", "mixture_components", "has_variable_formula", "has_vacancy_symbol",
        "has_mixed_occupancy", "has_html_tag", "has_hydrate", "formula_parse_success",
        "eligible_for_matbench_matching", "matbench_match_success", "band_gap_eV", "band_gap_class",
        "is_metal", "formation_energy", "formation_energy_class", "has_xrd", "xrd_file_path",
        "duplicate_rruff_id", "duplicate_group_size", "exclude_from_inorganic_main", "quality_flags"
    ]
    # Ensure all required meta_cols exist, otherwise log warning
    for col in meta_cols:
        if col not in df_master.columns:
            log.warning(f"Missing column {col} in master DataFrame.")
            df_master[col] = np.nan
            
    def save_subset(df_sub, name):
        if df_sub.empty: return
        df_sub.to_pickle(out_dir / f"{name}.pkl")
        df_sub[meta_cols].to_csv(out_dir / f"{name}_metadata.csv", index=False)
        log.info(f"Saved {name}: {len(df_sub)} rows")

    # ---- Sanity checks before saving ----
    n_rows = len(df_master)
    n_unique_sid = df_master["sample_id"].nunique()
    n_unique_rawfp = df_master["raw_file_path"].nunique()
    n_dup_sid = n_rows - n_unique_sid
    n_dup_rawfp = n_rows - n_unique_rawfp
    n_unique_rruff = df_master["rruff_id"].nunique()
    log.info("Master table: %d rows, %d unique sample_id, %d dup sample_id", n_rows, n_unique_sid, n_dup_sid)
    log.info("Master table: %d unique raw_file_path, %d dup raw_file_path", n_unique_rawfp, n_dup_rawfp)
    log.info("Master table: %d unique rruff_id", n_unique_rruff)
    if n_dup_sid > 0:
        log.error("CRITICAL: sample_id is not unique! %d duplicates found.", n_dup_sid)
        dup_sids = df_master[df_master.duplicated("sample_id", keep=False)]["sample_id"].unique()[:10]
        log.error("First duplicate sample_ids: %s", list(dup_sids))
    assert n_dup_sid == 0, f"sample_id must be unique, found {n_dup_sid} duplicates"
    if n_rows > 2 * n_unique_rawfp:
        log.error("POSSIBLE ROW EXPLOSION: rows (%d) > 2x unique raw_file_paths (%d)", n_rows, n_unique_rawfp)

    # 1. spec2prop_full_spectra
    save_subset(df_master, "spec2prop_full_spectra")
    
    # 2. Clean inorganic subset
    inorg_mask = (df_master["chemical_family"] != "Unknown") & (df_master["chemical_family"] != "Mixed/Other") & (~df_master["exclude_from_inorganic_main"])
    clean_inorg = df_master[inorg_mask].copy()
    save_subset(clean_inorg, "spec2prop_clean_inorganic")
    
    # 3. spec2prop_property_matched_inorganic
    prop_matched_inorg = clean_inorg[clean_inorg["matbench_match_success"] == True].copy()
    save_subset(prop_matched_inorg, "spec2prop_property_matched_inorganic")
    if len(prop_matched_inorg) > len(clean_inorg):
        log.error("ROW EXPLOSION: property_matched (%d) > clean_inorganic (%d)", len(prop_matched_inorg), len(clean_inorg))
    
    # 4. spec2prop_xrd_linked_inorganic — many_to_one merge
    if not df_xrd.empty:
        xrd_linked_inorg = clean_inorg[clean_inorg["has_xrd"] == True].copy()
        # XRD table already has one row per rruff_id, so this is many_to_one
        # (multiple Raman samples can share rruff_id, but XRD is unique per rruff_id)
        xrd_linked_inorg = xrd_linked_inorg.merge(
            df_xrd, on="rruff_id", how="left", validate="many_to_one"
        )
        if len(xrd_linked_inorg) > len(clean_inorg):
            log.error("ROW EXPLOSION: xrd_linked (%d) > clean_inorganic (%d)", len(xrd_linked_inorg), len(clean_inorg))
        
        xrd_linked_inorg.to_pickle(out_dir / "spec2prop_xrd_linked_inorganic.pkl")
        xrd_linked_inorg[meta_cols].to_csv(out_dir / "spec2prop_xrd_linked_inorganic_metadata.csv", index=False)
        log.info(f"Saved spec2prop_xrd_linked_inorganic: {len(xrd_linked_inorg)} rows")

    elapsed = time.time() - t0
    processing_log["elapsed_seconds"] = round(elapsed, 1)
    with open(out_dir / "processing_log.json", "w") as fh: json.dump(processing_log, fh, indent=2)
    log.info("Pipeline complete in %.1f seconds", elapsed)

if __name__ == "__main__":
    main()
