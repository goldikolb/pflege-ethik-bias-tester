from __future__ import annotations
import os
from typing import Any, List

from .base import Adapter


class LocalMistralAdapter(Adapter):
    """Adapter für Mistral AI über die offizielle SDK (La Plateforme).

    Erwartet Umgebungsvariable MISTRAL_API_KEY.
    Standardmodell: "ministral-3b-2410" (kleines, kostengünstiges Modell).
    """

    def generate(self, system: str, user: str, temperature: float, top_p: float, max_tokens: int) -> str:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY fehlt. Bitte .env anlegen und Schlüssel setzen."
            )

        try:
            from mistralai import Mistral  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mistralai ist nicht installiert. Bitte 'pip install mistralai' ausführen oder requirements.txt installieren."
            ) from e

        client = Mistral(api_key=api_key)

        # Modell-ID – vom Nutzer gewünscht
        model_id = "ministral-3b-2410"

        # Mistral-SDK erwartet eine Messages-Liste analog OpenAI-Style
        messages: List[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            # Mistral-Constraint: Bei greedy (temperature==0) muss top_p=1 sein
            effective_top_p = 1.0 if temperature == 0 else top_p
            resp = client.chat.complete(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=effective_top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Mistral API-Fehler: {e}") from e

        # Antwort extrahieren
        try:
            # resp.choices[0].message.content
            choices = getattr(resp, "choices", [])
            if not choices:
                return ""
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if isinstance(content, str):
                return content.strip()
            return ""
        except Exception:
            return ""
