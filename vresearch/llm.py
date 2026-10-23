"""Single entry point for every LLM call. All calls go through the Mesh API gateway
(OpenAI-compatible), via the OpenAI SDK pointed at the Mesh base_url."""
import json
import threading
import time
from typing import Any, Callable

from openai import OpenAI

from .config import Settings, get_settings


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = sorted(i for i in (text.find("{"), text.find("[")) if i != -1)
    for start in starts:
        open_c = text[start]
        close_c = "}" if open_c == "{" else "]"
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    raise ValueError("no valid JSON found in model output")


class LLM:
    def __init__(self, settings: Settings | None = None, on_llm_call: Callable[[dict], None] | None = None):
        self.s = settings or get_settings()
        if not self.s.api_key:
            raise RuntimeError("MESH_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = OpenAI(base_url=self.s.base_url, api_key=self.s.api_key, timeout=90)
        self.calls = 0
        self.on_llm_call = on_llm_call
        self._limiter = threading.Semaphore(max(1, self.s.llm_concurrency))

    def chat(self, system: str, user: str, *, model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 2000, tag: str = "") -> str:
        last: Exception | None = None
        target_model = model or self.s.model
        for attempt in range(3):
            try:
                with self._limiter:
                    t0 = time.time()
                    try:
                        r = self.client.chat.completions.create(
                            model=target_model,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    except Exception as temp_err:
                        # Some reasoning models reject temperature parameter
                        if "temperature" in str(temp_err).lower():
                            r = self.client.chat.completions.create(
                                model=target_model,
                                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                max_tokens=max_tokens,
                            )
                        else:
                            raise temp_err

                    ms = int((time.time() - t0) * 1000)
                    self.calls += 1
                    tokens = 0
                    if hasattr(r, "usage") and r.usage:
                        tokens = getattr(r.usage, "total_tokens", 0) or 0
                    if self.on_llm_call:
                        self.on_llm_call({"tag": tag or "llm", "model": target_model, "ms": ms, "tokens": tokens})
                    return r.choices[0].message.content or ""
            except Exception as e:  # network, rate limit, gateway errors
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed [{tag}]: {last}")

    def json(self, system: str, user: str, *, model: str | None = None,
             tag: str = "", max_tokens: int = 2000) -> Any:
        sys_json = system + "\n\nRespond with valid JSON only: no prose, no code fences."
        raw = self.chat(sys_json, user, model=model, temperature=0.1, max_tokens=max_tokens, tag=tag)
        try:
            return extract_json(raw)
        except ValueError:
            raw = self.chat(sys_json, user + "\n\nYour previous reply was not valid JSON. Reply with JSON only.",
                            model=model, temperature=0.0, max_tokens=max_tokens, tag=tag + ":retry")
            return extract_json(raw)
