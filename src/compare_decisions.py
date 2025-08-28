from __future__ import annotations
from pathlib import Path
from typing import Dict

from viz import plot_decision_grid
import pandas as pd

DEFAULT_RUNS: Dict[str, str] = {
    "baseline": "outputs/baseline/results.csv",
    "deterministic": "outputs/deterministic/results.csv",
    "care_bias": "outputs/care_bias/results.csv",
    "autonomy_bias": "outputs/autonomy_bias/results.csv",
}


def main() -> None:
    run_csvs = {k: v for k, v in DEFAULT_RUNS.items() if Path(v).exists()}
    if not run_csvs:
        raise SystemExit("Keine results.csv-Dateien gefunden. Bitte zuerst Runs ausführen.")

    out_png = "docs/decision_grid.png"
    plot_decision_grid(
        run_csvs,
        out_png=out_png,
        run_order=["baseline", "deterministic", "care_bias", "autonomy_bias"],
    )
    print(f"Entscheidungsübersicht gespeichert in: {Path(out_png).resolve()}")

    # Zusätzlich: Tabelle Entscheidungen (Modelle × Runs) als CSV und Markdown
    frames = []
    for run, path in run_csvs.items():
        df = pd.read_csv(path, usecols=["model", "decision"]).assign(run=run)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Pivot: Zeilen=Modelle, Spalten=Runs, Werte=Decision
    pivot = all_df.pivot_table(index="model", columns="run", values="decision", aggfunc=lambda x: x.iloc[0])
    pivot = pivot[[c for c in ["baseline", "deterministic", "care_bias", "autonomy_bias"] if c in pivot.columns]]

    # CSV
    out_csv = Path("docs/decision_table.csv")
    pivot.to_csv(out_csv)
    print(f"Entscheidungstabelle (CSV) gespeichert in: {out_csv.resolve()}")

    # Markdown ohne Zusatzabhängigkeiten erzeugen
    out_md = Path("docs/decision_table.md")
    cols = ["model"] + list(pivot.columns)
    md_rows = []
    # Header
    header = "| " + " | ".join(["Modell"] + list(pivot.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    md_rows.append(header)
    md_rows.append(sep)
    # Datenzeilen
    for model, row in pivot.iterrows():
        cells = [model] + [str(row[c]) if c in row and pd.notna(row[c]) else "" for c in pivot.columns]
        md_rows.append("| " + " | ".join(cells) + " |")

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Entscheidungen pro Modell und Run (PEG)\n\n")
        f.write("\n".join(md_rows) + "\n")
    print(f"Entscheidungstabelle (Markdown) gespeichert in: {out_md.resolve()}")


if __name__ == "__main__":
    main()
