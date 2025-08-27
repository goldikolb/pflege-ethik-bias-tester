from __future__ import annotations
import os
from typing import List

from .base import Adapter


class AnthropicClaudeAdapter(Adapter):
    """Anthropic-Adapter (Messages API) für "claude-sonnet-4"."""

    def generate(self, system: str, user: str, temperature: float, top_p: float, max_tokens: int) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY fehlt. Bitte .env anlegen und Schlüssel setzen."
            )

        # Lazy import, um Importfehler ohne Key/Installation zu vermeiden
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)

        # Primär gewünschtes Modell und Fallback-Liste
        primary_model = "claude-sonnet-4-20250514"
        fallbacks = [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20241022",
        ]

        def _call(model_id: str):
            return client.messages.create(
                model=model_id,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        try:
            resp = _call(primary_model)
        except anthropic.NotFoundError:
            last_exc = None
            for fb in fallbacks:
                try:
                    resp = _call(fb)
                    break
                except Exception as e:  # weiterhin versuchen
                    last_exc = e
            else:
                raise RuntimeError(
                    "Anthropic-Modell nicht verfügbar. Bitte in configs/models.yaml eine verfügbare Sonnet-Variante setzen (z. B. 'claude-3-7-sonnet-latest')."
                ) from last_exc

        # resp.content ist eine Liste von Content-Blocks; extrahiere Text-Inhalte
        parts: List[str] = []
        for block in getattr(resp, "content", []) or []:
            # neuere SDKs nutzen block.type == "text" und block.text
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts).strip()
