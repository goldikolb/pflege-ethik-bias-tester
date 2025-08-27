#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from src.orchestrator import Orchestrator


def main() -> None:
    # .env laden (lokale API-Keys, Konfigurationen)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Demenz Ethik Checker – Läufe starten")
    parser.add_argument("--run", required=True, choices=["baseline", "deterministic", "autonomy_bias"], help="Name des Runs")
    args = parser.parse_args()

    root = Path(__file__).parent
    orchestrator = Orchestrator(str(root))
    orchestrator.run(args.run)


if __name__ == "__main__":
    main()
