from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures"
OUTPUT = ROOT / "figures/reproduced"
OUTPUT.mkdir(parents=True, exist_ok=True)
