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
