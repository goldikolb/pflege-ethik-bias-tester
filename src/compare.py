from __future__ import annotations
from pathlib import Path
from typing import Dict

from viz import plot_axis_comparison


DEFAULT_RUNS: Dict[str, str] = {
    "baseline": "outputs/baseline/results.csv",
    "deterministic": "outputs/deterministic/results.csv",
    "care_bias": "outputs/care_bias/results.csv",
    "autonomy_bias": "outputs/autonomy_bias/results.csv",
}


def main() -> None:
    # Nur vorhandene CSVs berücksichtigen
    run_csvs = {k: v for k, v in DEFAULT_RUNS.items() if Path(v).exists()}
    if not run_csvs:
        raise SystemExit("Keine results.csv-Dateien gefunden. Bitte zuerst Runs ausführen.")

    out_png = "docs/axis_comparison.png"
    plot_axis_comparison(run_csvs, out_png=out_png,
                         run_order=["baseline", "deterministic", "care_bias", "autonomy_bias"])
    print(f"Vergleichsgrafik gespeichert in: {Path(out_png).resolve()}")


if __name__ == "__main__":
    main()
