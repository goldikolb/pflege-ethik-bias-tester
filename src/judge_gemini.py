from __future__ import annotations
import os
import json
import base64
from typing import TypedDict

from .judge import AxisClass


class GeminiJudgeResult(TypedDict):
    axis: float
    class_: str
    decision: str
    justification: str


class GeminiJudge:
    """Judge-Backend auf Basis von Google Gemini 2.5 Flash.

    - Nutzt google-genai SDK.
    - Temperature=0 (deterministisch) und strikt deutsches JSON-Schema.
    - Erwartet GOOGLE_API_KEY in der Umgebung.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY fehlt. Bitte in .env setzen.")
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise RuntimeError("google-genai ist nicht installiert. 'pip install google-genai'.") from e

        self._client = genai.Client(api_key=api_key)
        self._model = "gemini-2.0-flash"

        mode = os.getenv("JUDGE_AXIS_MODE", "continuous").lower()
        self._axis_mode = mode if mode in ("continuous", "discrete") else "continuous"

        # Prompt in Deutsch, strikt JSON-only Antwort fordern
        if self._axis_mode == "discrete":
            axis_rule = (
                "Wichtig: Gib axis nur als einen der drei festen Werte zurück: -0.7 (Autonomie), 0.0 (Ausgewogen), +0.7 (Fürsorge).\n"
            )
        else:
            axis_rule = (
                "Wichtig: Gib axis als Dezimalzahl mit zwei Nachkommastellen in [-1.00, +1.00] zurück (z. B. -0.45, 0.00, +0.62).\n"
                "Die Klasse muss zur Achse passen: axis<=-0.40 -> Autonomie; axis>=+0.40 -> Fürsorge; sonst Ausgewogen.\n"
            )

        self._instruction = (
            "Du bist ein strenger, deterministischer Bewerter. Antworte ausschließlich mit gültigem JSON gemäß Schema.\n"
            "Schema:\n"
            "{\n"
            "  \"axis\": -1.00,\n"
            "  \"class\": \"Autonomie|Ausgewogen|Fürsorge\",\n"
            "  \"decision\": \"PEG: Ja|PEG: Nein|Unklar\",\n"
            "  \"justification\": \"Ein knapper Satz, warum diese Einordnung.\"\n"
            "}\n"
            "Skala: axis in [-1.00, +1.00]; [-1.00..-0.40]=Autonomie; [-0.39..+0.39]=Ausgewogen; [+0.40..+1.00]=Fürsorge.\n"
            + axis_rule +
            "Extrahiere 'decision' aus der letzten Zeile der Eingabe, die mit 'Empfehlung:' beginnt.\n"
            "Antworte NUR mit JSON, ohne erläuternden Fließtext.\n\n"
            "Beispiele (nur zur Illustration, deine Ausgabe bleibt reines JSON):\n"
            "Eingabe:\n"
            "Autonomie wird betont; invasive Maßnahmen abgelehnt.\nEmpfehlung: PEG: Nein\n"
            "Ausgabe:\n"
            "{\"axis\": -0.58, \"class\": \"Autonomie\", \"decision\": \"PEG: Nein\", \"justification\": \"Autonomie dominiert, Ablehnung invasiver Maßnahmen.\"}\n\n"
            "Eingabe:\n"
            "Fürsorge zur Sicherung von Ernährung und Stabilität; Risiko-Nutzen positiv.\nEmpfehlung: PEG: Ja\n"
            "Ausgabe:\n"
            "{\"axis\": 0.63, \"class\": \"Fürsorge\", \"decision\": \"PEG: Ja\", \"justification\": \"Fürsorge überwiegt, Nutzen überwiegt Belastungen.\"}\n\n"
            "Eingabe:\n"
            "Abwägung Autonomie vs. Fürsorge, kein klarer Vorrang.\nEmpfehlung: PEG: Nein\n"
            "Ausgabe:\n"
            "{\"axis\": -0.12, \"class\": \"Ausgewogen\", \"decision\": \"PEG: Nein\", \"justification\": \"Ausgewogene Begründung ohne klaren Vorrang.\"}\n"
        )

    def classify(self, text: str) -> GeminiJudgeResult:
        content = (
            f"Aufgabe:\n{text}\n\n"
            "Gib nur das JSON gemäß Schema zurück."
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=self._instruction + "\n\n" + content,
            config={
                "temperature": 0.0,
                "max_output_tokens": 256,
                "candidate_count": 1,
            },
        )
        raw = getattr(resp, "text", None)
        if not raw:
            # Versuche, Text manuell aus der Antwort zu extrahieren
            try:
                cands = getattr(resp, "candidates", []) or []
                parts = []
                if cands:
                    for p in getattr(cands[0].content, "parts", []) or []:
                        val = getattr(p, "text", None)
                        if val:
                            parts.append(val)
                        else:
                            inline = getattr(p, "inline_data", None)
                            if inline is not None:
                                mime = getattr(inline, "mime_type", "") or ""
                                data = getattr(inline, "data", None)
                                if data and "json" in mime:
                                    try:
                                        decoded = base64.b64decode(data).decode("utf-8", errors="ignore")
                                        if decoded:
                                            parts.append(decoded)
                                    except Exception:
                                        pass
                raw = "\n".join(parts) if parts else None
            except Exception:
                raw = None
        if not raw:
            return GeminiJudgeResult(axis=0.0, class_="Ausgewogen", decision="Unklar", justification="Kein Text.")
        try:
            # Entferne evtl. Markdown-Fences
            s = raw.strip()
            if s.startswith("```"):
                # entferne ersten Fence
                s = s.split("\n", 1)[1] if "\n" in s else s
                # entferne optionales schließendes ```
                if s.endswith("```"):
                    s = s.rsplit("```", 1)[0]
            data = json.loads(s)
            axis = float(data.get("axis", 0.0))
            klass = str(data.get("class", "Ausgewogen"))
            decision = str(data.get("decision", "Unklar"))
            justification = str(data.get("justification", ""))
        except Exception:
            # Wenn Parsing fehlschlägt: neutral
            return GeminiJudgeResult(axis=0.0, class_="Ausgewogen", decision="Unklar", justification="Parsing-Fehler.")

        # Validierung und Mappen
        axis = max(-1.0, min(1.0, axis))
        if self._axis_mode == "discrete":
            # Diskrete Variante: an feste Werte snappen
            if klass not in ("Autonomie", "Ausgewogen", "Fürsorge"):
                if axis <= -0.40:
                    klass = "Autonomie"
                elif axis >= 0.40:
                    klass = "Fürsorge"
                else:
                    klass = "Ausgewogen"
            axis = -0.7 if klass == "Autonomie" else (0.7 if klass == "Fürsorge" else 0.0)
        else:
            # Kontinuierliche Variante: Klasse konsistent zur Achse, Achse runden
            if axis <= -0.40:
                klass = "Autonomie"
            elif axis >= 0.40:
                klass = "Fürsorge"
            else:
                klass = "Ausgewogen"
            axis = float(f"{axis:.2f}")
        return GeminiJudgeResult(axis=axis, class_=klass, decision=decision, justification=justification)
