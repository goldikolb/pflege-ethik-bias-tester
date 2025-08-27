from __future__ import annotations
import os
from typing import Any

from .base import Adapter


class OpenAIGPTAdapter(Adapter):
    """OpenAI-Adapter (Chat Completions).

    Erwartet Umgebungsvariable OPENAI_API_KEY.
    Modell-ID laut Vorgabe: "gpt-5" (kann in der OpenAI-Konsole variieren).
    """

    def generate(self, system: str, user: str, temperature: float, top_p: float, max_tokens: int) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY fehlt. Bitte .env erstellen und Schlüssel setzen (nicht ins Repo committen)."
            )

        # Import hier, damit das Modul auch ohne Abhängigkeit geladen werden kann
        from openai import OpenAI  # type: ignore
        from openai import BadRequestError  # type: ignore

        client = OpenAI(api_key=api_key)

        # Chat Completions mit System- und User-Prompt
        base_kwargs = dict(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
        )
        # Erster Versuch mit temperature/top_p laut Konfiguration
        try:
            resp = client.chat.completions.create(
                **base_kwargs, temperature=temperature, top_p=top_p
            )
        except BadRequestError as e:
            msg = str(e)
            # Fallback: ohne temperature/top_p erneut versuchen
            if "temperature" in msg or "top_p" in msg or "unsupported" in msg:
                resp = client.chat.completions.create(**base_kwargs)
            else:
                raise

        content = resp.choices[0].message.content or ""
        # OpenAI kann Listen/Nachrichten-Objekte liefern; sicherstellen, dass String entsteht
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return content.strip()
