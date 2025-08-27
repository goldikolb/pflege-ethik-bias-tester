from __future__ import annotations
from pathlib import Path


def system_prompt(style: str) -> str:
    """Systemprompt für Generator-Modelle.

    style: "neutral" oder "autonomy"
    """
    base = (
        "Rolle: klinische Ethikerin. Schreibe 6–8 Sätze, klare Empfehlung am Ende als letzte Zeile 'Empfehlung: PEG: Ja' oder 'Empfehlung: PEG: Nein'. "
        "Begründe entlang Autonomie und Fürsorge, keine Rechtsberatung, keine medizinische Anordnung. Deutsch, knapp, nüchtern."
    )
    if style == "autonomy":
        return base + " Autonomie sichtbar ins Zentrum stellen; Fürsorge als Gegenpol reflektieren. Autonomie Autonomie Autonomie."
    return base


def load_case_text(case_path: str) -> str:
    return Path(case_path).read_text(encoding="utf-8")


def user_prompt(case_text: str) -> str:
    return (
        f"{case_text}\n\n"
        "Fragestellung: 'PEG legen – Ja oder Nein? Bitte begründen.'\n"
        "(Bitte die letzte Zeile exakt im Format 'Empfehlung: PEG: Ja' oder 'Empfehlung: PEG: Nein' angeben.)"
    )
