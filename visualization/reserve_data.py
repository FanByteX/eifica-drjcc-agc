import csv
import json
from collections import defaultdict
from pathlib import Path
def read_results(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    by_key = {(row["case"], int(row["N_WDR"]), int(row["seed"])): row
              for row in records}
    sizes = [int(n) for n in manifest["N_values"]]
    seeds = [int(seed) for seed in manifest["seeds"]]
    expected = {(case, n, seed) for case in ("direct", "reserve")
                for n in sizes for seed in seeds}
    if set(by_key) != expected:
        raise ValueError(f"Recorded keys differ from manifest: missing={expected - set(by_key)}, "
                         f"extra={set(by_key) - expected}")
    paired = defaultdict(list)
    status13 = defaultdict(list)
    for n in sizes:
        for seed in seeds:
            direct = by_key[("direct", n, seed)]
            reserve = by_key[("reserve", n, seed)]
            if direct["status"] == reserve["status"] == "OPTIMAL":
                paired[n].append((direct, reserve))
            if reserve["status"] == "GUROBI_13":
                status13[n].append(float(reserve["core_time_s"]))
    return manifest, sizes, seeds, paired, status13
