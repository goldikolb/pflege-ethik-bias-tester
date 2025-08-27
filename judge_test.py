#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path

# .env laden
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Judge auswählen
backend = os.getenv("JUDGE_BACKEND", "local").lower()
if backend == "gemini":
    from src.judge_gemini import GeminiJudge as Judge
else:
    from src.judge import Judge  # type: ignore


def main() -> None:
    # Beispielantwort eines Modells (Deutsch, mit letzter Zeile Empfehlung)
    sample = (
        "Kurzbegründung: Der Wille und die vorher geäußerten Präferenzen von Herrn Herrmann sind maßgeblich.\n"
        "Bei fortgeschrittener Demenz ist die Wahrung der Würde zentral; Zwangsmaßnahmen sind zu vermeiden.\n"
        "Empfehlung: PEG: Nein\n"
    )
    judge = Judge()
    res = judge.classify(sample)
    print(res)


if __name__ == "__main__":
    main()
