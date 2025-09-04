from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_axis(csv_path: str, out_png: str) -> None:
    df = pd.read_csv(csv_path)
    # pro Modell Mittelwert der Achse (hier 1 Sample je Modell, aber zukunftssicher)
    g = df.groupby("model", as_index=False)["axis"].mean()
    plt.figure(figsize=(8, 4))
    plt.bar(g["model"], g["axis"], color="#4C78A8")
    plt.axhline(0.0, color="#999", linewidth=1, zorder=1)
    # Cut-off-Linien gemäß Spezifikation
    plt.axhline(-0.40, color="#D62728", linestyle="--", linewidth=1)
    plt.axhline(0.40, color="#2CA02C", linestyle="--", linewidth=1)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Ethik-Achse (-1 Autonomie … +1 Fürsorge)")
    plt.title("Achsenwert je Modell")
    plt.xticks(rotation=20, ha="right")
    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_decision_grid(run_csvs: dict[str, str], out_png: str, run_order: list[str] | None = None) -> None:
    """Visualisiert die PEG-Entscheidung (Ja/Nein/Unklar) als Grid (Modelle × Runs).

    Farben: Ja=grün, Nein=rot, Unklar=grau.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if run_order is None:
        run_order = ["baseline", "deterministic", "care_bias", "autonomy_bias"]

    frames = []
    for run, p in run_csvs.items():
        df = pd.read_csv(p)
        frames.append(df[["model", "decision"]].assign(run=run))
    all_df = pd.concat(frames, ignore_index=True)

    # Reihenfolge bereinigen nach vorhandenen Runs
    run_order = [r for r in run_order if r in all_df["run"].unique().tolist()]
    models = sorted(all_df["model"].unique().tolist())

    # Mapping Entscheidungen -> Code + Farbe
    dec_to_code = {"PEG: Ja": 1, "PEG: Nein": -1, "Unklar": 0}
    colors = {1: "#54A24B", -1: "#E45756", 0: "#9A9A9A"}

    grid = np.zeros((len(models), len(run_order)), dtype=int)
    for i, m in enumerate(models):
        for j, r in enumerate(run_order):
            row = all_df[(all_df["model"] == m) & (all_df["run"] == r)]
            if not row.empty:
                dec = row.iloc[0]["decision"]
                grid[i, j] = dec_to_code.get(dec, 0)
            else:
                grid[i, j] = 0

    # Plot als farbige Zellen
    plt.figure(figsize=(max(6, 1.2 * len(run_order)), max(4, 0.6 * len(models))))
    for i in range(len(models)):
        for j in range(len(run_order)):
            code = grid[i, j]
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, color=colors.get(code, "#9A9A9A")))

    # Achsen und Labels
    plt.xlim(0, len(run_order))
    plt.ylim(0, len(models))
    plt.xticks(np.arange(len(run_order)) + 0.5, run_order, rotation=20, ha="right")
    plt.yticks(np.arange(len(models)) + 0.5, models)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Entscheidung je Modell und Run (PEG)")
    # Legende
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#54A24B", label="PEG: Ja"),
        Patch(facecolor="#E45756", label="PEG: Nein"),
        Patch(facecolor="#9A9A9A", label="Unklar"),
    ]
    plt.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def plot_axis_comparison(run_csvs: dict[str, str], out_png: str, run_order: list[str] | None = None) -> None:
    """Erzeugt einen gruppierten Balkenplot über mehrere Runs.

    run_csvs: Mapping von Run-Name -> Pfad zur results.csv
    out_png: Zielbild
    run_order: Reihenfolge der Balken pro Modell (Default: Baseline, Deterministic, Care, Autonomy)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    if run_order is None:
        run_order = ["baseline", "deterministic", "care_bias", "autonomy_bias"]

    # CSVs einlesen und zusammenführen
    frames = []
    for run, p in run_csvs.items():
        df = pd.read_csv(p)
        df = df[["model", "axis"]].copy()
        df["run"] = run
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Nur Runs in gewünschter Reihenfolge und vorhanden
    run_order = [r for r in run_order if r in all_df["run"].unique().tolist()]

    models = sorted(all_df["model"].unique().tolist())
    n_models = len(models)
    n_runs = len(run_order)

    # Farben konsistent
    colors = {
        "baseline": "#4C78A8",
        "deterministic": "#72B7B2",
        "care_bias": "#E45756",
        "autonomy_bias": "#54A24B",
    }

    import numpy as np
    x = np.arange(n_models)
    width = 0.18 if n_runs >= 4 else 0.22

    plt.figure(figsize=(max(8, 1.6 * n_models), 4.8))

    for i, run in enumerate(run_order):
        sub = all_df[all_df["run"] == run].set_index("model").reindex(models)
        # Fehlende Werte (NaN) sichtbar machen: als 0 plotten und mit Hatch kennzeichnen
        missing = sub["axis"].isna().values
        y_raw = sub["axis"].values
        import numpy as np
        y = np.where(missing, 0.0, y_raw)
        bars = plt.bar(
            x + (i - (n_runs - 1) / 2) * width,
            y,
            width,
            label=run,
            color=colors.get(run, None),
            edgecolor="#222",
            linewidth=0.6,
            zorder=2,
        )
        # Kennzeichnung für fehlende Achsenwerte
        for b, m in zip(bars, missing):
            if m:
                b.set_hatch("//")
                b.set_alpha(0.35)
        # Marker an der Balkenspitze für bessere Sichtbarkeit auch bei y==0.0
        x_centers = x + (i - (n_runs - 1) / 2) * width
        plt.scatter(x_centers, y, s=16, c=colors.get(run, None), edgecolors="#222", zorder=3)

    # Hilfslinien
    plt.axhline(0.0, color="#999", linewidth=1)
    plt.axhline(-0.40, color="#D62728", linestyle="--", linewidth=1)
    plt.axhline(0.40, color="#2CA02C", linestyle="--", linewidth=1)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Ethik-Achse (-1 Autonomie … +1 Fürsorge)")
    plt.title("Achsenvergleich je Modell (Baseline, Deterministic, Care, Autonomy)")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.legend(title="Run", ncol=2, fontsize=9)
    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
