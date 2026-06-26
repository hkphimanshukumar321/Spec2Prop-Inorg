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
