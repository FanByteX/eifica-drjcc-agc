"""Recreate the manuscript result figures from the bundled plotting data."""
import argparse
from pathlib import Path
import subprocess
import sys

SCRIPTS={
    "3": "plot_figure3.py",
    "4": "plot_figure4.py",
    "5,7": "plot_figures5_7.py",
    "6": "plot_figure6.py",
    "8,10,12": "plot_figures8_10_12.py",
    "9,11": "plot_figures9_11.py",
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", default="all", help="all or comma-separated figure numbers")
    args=parser.parse_args()
    requested=set(str(n) for n in range(3,13)) if args.figures=="all" else set(args.figures.split(","))
    if not requested <= set(str(n) for n in range(3,13)):
        parser.error("Choose figure numbers 3 through 12. Figure 1 uses TikZ and Figure 2 uses Illustrator.")
    for numbers,script in SCRIPTS.items():
        if requested.intersection(numbers.split(",")):
            subprocess.run([sys.executable,str(Path(__file__).with_name(script))],check=True)
    print("Saved figures to figures/reproduced/")

if __name__=="__main__":
    main()
