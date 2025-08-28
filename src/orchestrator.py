from __future__ import annotations
import os
import time
import importlib
import yaml
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .prompts import system_prompt, load_case_text, user_prompt
from .judge import Judge


@dataclass
class RunParams:
    temperature: float
    top_p: float
    max_tokens: int
    system_style: str  # "neutral" | "autonomy"


class Orchestrator:
    """Steuert Läufe über Modelle, sammelt Ergebnisse, erzeugt CSV & Grafik."""

    def __init__(self, project_root: str) -> None:
        self.root = Path(project_root)
        backend = os.getenv("JUDGE_BACKEND", "local").lower()
        if backend == "gemini":
            try:
                from .judge_gemini import GeminiJudge
                self.judge = GeminiJudge()
                self.judge_backend = "gemini"
            except Exception as e:
                print(f"Warnung: Gemini-Judge konnte nicht geladen werden ({e}). Fallback auf lokalen Judge.")
                self.judge = Judge()
                self.judge_backend = "local"
        else:
            self.judge = Judge()
            self.judge_backend = "local"

    def _load_yaml(self, p: Path) -> Dict[str, Any]:
        return yaml.safe_load(p.read_text(encoding="utf-8"))

    def _load_models(self) -> List[Dict[str, Any]]:
        cfg = self._load_yaml(self.root / "configs" / "models.yaml")
        return cfg["models"]

    def _adapter_instance(self, adapter_key: str):
        mod = importlib.import_module(f"src.adapters.{adapter_key}")
        # Konvention: Klassenname aus Modul ableiten
        class_name = {
            "openai_gpt": "OpenAIGPTAdapter",
            "anthropic_claude": "AnthropicClaudeAdapter",
            "xai_grok": "XAIGrokAdapter",
            "local_mistral": "LocalMistralAdapter",
            "local_teuken": "LocalTeukenAdapter",
        }[adapter_key]
        cls = getattr(mod, class_name)
        return cls()

    def run(self, run_name: str) -> None:
        run_cfg = self._load_yaml(self.root / "configs" / f"run_{run_name}.yaml")
        params = RunParams(**run_cfg["params"])

        sys_prompt = system_prompt(params.system_style)
        case_filename = run_cfg.get("case", "herr_herrmann.txt")
        case_text = load_case_text(str(self.root / "cases" / case_filename))
        usr_prompt = user_prompt(case_text)

        models = self._load_models()

        out_dir = self.root / "outputs" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        results_csv = out_dir / "results.csv"
        raw_dir = out_dir / "raw_opinions"
        raw_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for m in models:
            adapter = self._adapter_instance(m["adapter"])
            t0 = time.perf_counter()
            # Per-Modell-Overrides erlauben (optional in models.yaml unter 'params')
            m_params = m.get("params", {})
            eff_temperature = float(m_params.get("temperature", params.temperature))
            eff_top_p = float(m_params.get("top_p", params.top_p))
            eff_max_tokens = int(m_params.get("max_tokens", params.max_tokens))

            text = adapter.generate(
                system=sys_prompt,
                user=usr_prompt,
                temperature=eff_temperature,
                top_p=eff_top_p,
                max_tokens=eff_max_tokens,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            verdict = self.judge.classify(text)

            # Debug: Rohtext pro Modell speichern
            try:
                raw_path = raw_dir / f"{m['provider']}__{m['name']}.txt"
                raw_path.write_text(text, encoding="utf-8")
                if not (text or "").strip():
                    print(f"Warnung: Leere Opinion für {m['name']} ({m['provider']}).")
            except Exception as _:
                # Debug-Schreiben darf den Run nicht abbrechen
                pass

            rows.append(
                {
                    "run": run_name,
                    "model": m["name"],
                    "provider": m["provider"],
                    "judge_backend": self.judge_backend,
                    "temperature": eff_temperature,
                    "top_p": eff_top_p,
                    "max_tokens": eff_max_tokens,
                    "system_style": params.system_style,
                    "opinion": text.replace("\n", "\\n"),
                    "decision": verdict["decision"],
                    "class": verdict["class_"],
                    "axis": verdict["axis"],
                    "why": verdict["justification"],
                    "latency_ms": latency_ms,
                }
            )

        # CSV schreiben
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run",
                    "model",
                    "provider",
                    "judge_backend",
                    "temperature",
                    "top_p",
                    "max_tokens",
                    "system_style",
                    "opinion",
                    "decision",
                    "class",
                    "axis",
                    "why",
                    "latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        # Figure erzeugen
        from .viz import plot_axis

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_axis(str(results_csv), str(fig_dir / "axis.png"))

        print(f"Ergebnisse gespeichert in: {results_csv}")
