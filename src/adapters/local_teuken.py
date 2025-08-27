from __future__ import annotations
from typing import Optional

from .base import Adapter

# Einfache Cache-Variablen, damit das Modell nur einmal geladen wird
_MODEL = None
_TOKENIZER = None
_DEVICE = None


class LocalTeukenAdapter(Adapter):
    """Lokaler Adapter für Teuken 7B (Hugging Face, Transformers).

    Nutzt die Chat-Template "DE" (siehe Model Card). Unterstützt Temperatur und top_p.
    Erwartet die Pakete: torch, transformers, sentencepiece, huggingface_hub.
    """

    def _ensure_model(self) -> None:
        global _MODEL, _TOKENIZER, _DEVICE
        if _MODEL is not None and _TOKENIZER is not None and _DEVICE is not None:
            return

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyTorch (torch) ist nicht installiert. Bitte 'pip install torch' ausführen (siehe requirements)."
            ) from e

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers ist nicht installiert. Bitte 'pip install transformers sentencepiece huggingface_hub' ausführen."
            ) from e

        # Gerätauswahl: bevorzugt MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.bfloat16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        model_name = "openGPT-X/Teuken-7B-instruct-v0.6"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            model = model.to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Fehler beim Laden des Teuken-Modells '{model_name}': {e}"
            ) from e

        _MODEL, _TOKENIZER, _DEVICE = model, tokenizer, device

    def _build_input_ids(self, system: str, user: str):
        """Erzeuge Eingabe-IDs via Chat-Template 'DE'.

        Hinweis: Die 'DE'-Vorlage erwartet Sequenzen User/Assistant. Daher nutzen wir
        nur eine User-Nachricht und betten den Systemtext vorne ein.
        """
        assert _TOKENIZER is not None and _MODEL is not None
        combined = (system.strip() + "\n\n" if system and system.strip() else "") + user.strip()
        messages = [{"role": "User", "content": combined}]

        prompt_ids = _TOKENIZER.apply_chat_template(
            messages,
            chat_template="DE",
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return prompt_ids

    def generate(self, system: str, user: str, temperature: float, top_p: float, max_tokens: int) -> str:
        self._ensure_model()
        assert _MODEL is not None and _TOKENIZER is not None and _DEVICE is not None

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("PyTorch fehlt zur Laufzeit.") from e

        input_ids = self._build_input_ids(system, user)

        # Sampling-Parameter
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "temperature": float(max(0.0, temperature)),
            "top_p": float(top_p),
        }
        if not do_sample:
            # Greedy: keine Sampling-Parameter, die Sampling erzwingen
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("temperature", None)

        try:
            with torch.no_grad():
                out = _MODEL.generate(
                    input_ids.to(_MODEL.device),
                    **gen_kwargs,
                )
        except Exception as e:
            raise RuntimeError(f"Fehler bei der lokalen Textgenerierung (Teuken): {e}") from e

        # Nur den neu erzeugten Teil dekodieren (ohne Prompt)
        try:
            gen_ids = out[0][input_ids.shape[-1] :]
            text = _TOKENIZER.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            # Fallback: vollständige Sequenz dekodieren
            text = _TOKENIZER.decode(out[0], skip_special_tokens=True)

        return text.strip()
