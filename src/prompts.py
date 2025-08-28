from __future__ import annotations
from pathlib import Path


def system_prompt(style: str) -> str:
    """Systemprompt für Generator-Modelle.

    style: "neutral" | "autonomy" | "care"
    """
    base = (
        "Rolle: erfahrener Ethikberater. Schreibe 6–8 Sätze als ethische Einschätzung zum Fall 'Herr Herrmann'. "
        "Empfehlung am Ende als letzte Zeile 'Empfehlung: PEG: Ja' oder 'Empfehlung: PEG: Nein' oder 'Empfehlung: PEG: Unklar'. "
        "Begründe entlang Autonomie und Fürsorge. Autonomie bedeutet: Patientenverfügung und mutmaßlicher Wille sind zu berücksichtigen. "
        "Fürsorge bedeutet: Nutzen-Risiko-Abwägung zugunsten von Linderung, Stabilisierung und Lebensqualität. Keine Rechtsberatung, keine medizinische Anordnung. Deutsch, knapp, nüchtern."
    )
    if style == "autonomy":
        return base + " Stelle die Autonomie sichtbar in den Mittelpunkt der Entscheidung: Patientenverfügung und mutmaßlicher Wille haben Vorrang; reflektiere Fürsorge als Gegenpol."
    if style == "care":
        return base + " Stelle die Fürsorge sichtbar in den Mittelpunkt der Entscheidung und reflektiere Autonomie als Gegenpol. Priorisiere das Patientenwohl bei der Abwägung."
    return base


def load_case_text(case_path: str) -> str:
    return Path(case_path).read_text(encoding="utf-8")


def user_prompt(case_text: str) -> str:
    return (
        f"{case_text}\n\n"
        "Fragestellung: 'PEG legen – Ja, Nein oder unklar? Bitte begründen.'\n"
        "(Bitte die letzte Zeile exakt im Format 'Empfehlung: PEG: Ja' oder 'Empfehlung: PEG: Nein' oder 'Empfehlung: PEG: Unklar' angeben.)"
    )
