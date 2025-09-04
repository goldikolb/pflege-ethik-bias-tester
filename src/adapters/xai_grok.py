from __future__ import annotations
import os
from typing import Any, Dict

import httpx

from .base import Adapter


class XAIGrokAdapter(Adapter):
    """XAI Grok-Adapter über HTTP (Chat Completions API).

    Erwartet Umgebungsvariable XAI_API_KEY.
    Primärmodell: "grok-4-latest"; Fallback: "grok-4".
    """

    API_URL = "https://api.x.ai/v1/chat/completions"
    MSG_URL = "https://api.x.ai/v1/messages"

    def _build_payload(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        # Nur setzen, wenn Werte spezifiziert sind
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        # Einige Implementierungen nutzen max_tokens; optional setzen
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        return payload

    def generate(self, system: str, user: str, temperature: float, top_p: float, max_tokens: int) -> str:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY fehlt. Bitte .env anlegen und Schlüssel setzen.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        primary_model = "grok-4-0709"
        fallback_model = "grok-4"

        # 1) Versuch: Offizielle xAI SDK, falls vorhanden
        try:
            from xai_sdk import Client  # type: ignore
            from xai_sdk.chat import user as xai_user, system as xai_system  # type: ignore

            client = Client(api_key=api_key, timeout=60)
            # Einheitliche Reproduzierbarkeit: feste ID grok-4-0709
            chat = client.chat.create(model=primary_model)
            chat.append(xai_system(str(system)))
            chat.append(xai_user(str(user)))
            resp = chat.sample()
            content = getattr(resp, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            # Debug-Hinweis, falls leer
            print("xai_sdk lieferte leeren Content, wechsle zu HTTP-Fallback.")
        except ImportError:
            # SDK nicht installiert -> HTTP-Fallback nutzen
            pass
        except Exception as e:
            # SDK-Fehler -> HTTP-Fallback versuchen
            print(f"xai_sdk Fehler: {e}. HTTP-Fallback wird genutzt.")

        def _request(
            model_id: str,
            include_sampler: bool,
            include_max_tokens: bool,
        ) -> httpx.Response:
            payload = self._build_payload(
                model=model_id,
                system=system,
                user=user,
                temperature=temperature if include_sampler else None,
                top_p=top_p if include_sampler else None,
                max_tokens=max_tokens if include_max_tokens else None,
            )
            with httpx.Client(timeout=45.0) as client:
                return client.post(self.API_URL, headers=headers, json=payload)

        # Reihenfolge der Versuche (Primärmodell):
        # 1) mit Sampler + mit max_tokens
        # 2) ohne Sampler + mit max_tokens
        # 3) mit Sampler + ohne max_tokens
        # 4) ohne Sampler + ohne max_tokens
        attempts = [
            (primary_model, True, True),
            (primary_model, False, True),
            (primary_model, True, False),
            (primary_model, False, False),
        ]
        resp = None
        for model_id, incl_sampler, incl_max in attempts:
            r = _request(model_id, include_sampler=incl_sampler, include_max_tokens=incl_max)
            if r.status_code != 400 and r.status_code != 404:
                resp = r
                break
            last_r = r
        if resp is None:
            resp = last_r  # letzter 400/404, wird unten behandelt

        if resp.status_code == 404:
            # Fallback-Modell: gleiche Abfolge
            fb_attempts = [
                (fallback_model, True, True),
                (fallback_model, False, True),
                (fallback_model, True, False),
                (fallback_model, False, False),
            ]
            fb_resp = None
            for model_id, incl_sampler, incl_max in fb_attempts:
                r = _request(model_id, include_sampler=incl_sampler, include_max_tokens=incl_max)
                if r.status_code // 100 == 2:
                    fb_resp = r
                    break
            resp = fb_resp or resp

        if resp.status_code // 100 != 2:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"XAI API-Fehler: HTTP {resp.status_code}: {detail}")

        data = resp.json()
        # Erwartete Struktur: { choices: [ { message: { content: str|list } } ] }
        try:
            msg = data["choices"][0]["message"]
            content = msg.get("content")
            # 1) Direkter String
            if isinstance(content, str):
                return content.strip()
            # 2) Liste von Blöcken mit text/content
            if isinstance(content, list):
                parts: list[str] = []
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    # Häufige Varianten: {text: str} | {content: str} | {type: 'text', text: {value: str}} | {type:'output_text', text: str}
                    if isinstance(blk.get("text"), str):
                        parts.append(blk["text"])  # direkt
                    elif isinstance(blk.get("content"), str):
                        parts.append(blk["content"])  # alternativ
                    elif blk.get("type") in {"text", "output_text"}:
                        t = blk.get("text")
                        if isinstance(t, dict) and isinstance(t.get("value"), str):
                            parts.append(t["value"])  # OpenAI-ähnlich: text.value
                        elif isinstance(t, str):
                            parts.append(t)
                txt = "".join(parts).strip()
                if txt:
                    return txt
            # 3) Fallback: manchmal liegt Text direkt in choices[0]["text"]
            alt = data["choices"][0].get("text")
            if isinstance(alt, str) and alt.strip():
                return alt.strip()
            # 4) Weiterer Fallback: top-level output_text (manche Implementierungen)
            ot = data.get("output_text")
            if isinstance(ot, str) and ot.strip():
                return ot.strip()
            # Wenn Chat-Completions leer blieb: Fallback auf Messages-Endpoint
            # (Anthropic-kompatibel)
            try:
                with httpx.Client(timeout=45.0) as client:
                    msg_payload: Dict[str, Any] = {
                        "model": fallback_model,
                        "system": str(system),
                        "messages": [{"role": "user", "content": str(user)}],
                        # nur setzen, wenn sinnvoll
                        **({"temperature": float(temperature)} if temperature is not None else {}),
                        **({"top_p": float(top_p)} if top_p is not None else {}),
                        **({"max_tokens": int(max_tokens)} if isinstance(max_tokens, int) else {}),
                    }
                    r2 = client.post(self.MSG_URL, headers=headers, json=msg_payload)
                if r2.status_code // 100 != 2:
                    # Debug-Ausgabe mit kurzem Ausschnitt
                    try:
                        snippet = r2.text[:200]
                    except Exception:
                        snippet = "<no text>"
                    print(f"XAI /messages HTTP {r2.status_code}, snippet: {snippet}")
                    return "[xAI lieferte keinen Text]"
                d2 = r2.json()
                # Struktur laut Anthropic-kompatiblem Format: content ist Liste von Blocks
                try:
                    blocks = d2.get("content") or d2.get("message", {}).get("content")
                    parts: list[str] = []
                    if isinstance(blocks, list):
                        for blk in blocks:
                            if isinstance(blk, dict):
                                txt = blk.get("text") or blk.get("content")
                                if isinstance(txt, str):
                                    parts.append(txt)
                    txt2 = "".join(parts).strip()
                    if txt2:
                        return txt2
                except Exception:
                    pass
            except Exception as _e:
                # Fallback darf Run nicht abbrechen
                pass
            # Debug: kurzer Ausschnitt aus ursprünglicher Antwort loggen
            try:
                snippet = str(data)[:200]
            except Exception:
                snippet = "<unavailable>"
            print(f"XAI chat/completions lieferte leer. Debug-Snippet: {snippet}")
            return "[xAI lieferte keinen Text]"
        except Exception as e:
            raise RuntimeError(f"Unerwartetes XAI-Antwortformat: {data}") from e
