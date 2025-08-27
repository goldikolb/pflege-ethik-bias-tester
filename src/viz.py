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
    plt.axhline(0.0, color="#999", linewidth=1)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Ethik-Achse (-1 Autonomie … +1 Fürsorge)")
    plt.title("Achsenwert je Modell")
    plt.xticks(rotation=20, ha="right")
    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
