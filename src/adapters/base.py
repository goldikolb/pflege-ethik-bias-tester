from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class Adapter(Protocol):
    """Einheitliche Schnittstelle für alle Modelladapter."""

    def generate(self, system: str, user: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """Erzeugt einen Antworttext.

        Parameter:
        - system: Systemprompt (Rolle, Stilvorgaben)
        - user: Userprompt (Fallvignette + Frage)
        - temperature, top_p, max_tokens: Sampler-Parameter
        """
        ...


@dataclass
class AdapterConfig:
    name: str
    provider: str
    adapter: str
