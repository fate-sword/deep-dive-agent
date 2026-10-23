"""Single entry point for every LLM call. Supports Mesh API gateway and dev-only OpenRouter."""
import json
import random
import re
import threading
import time
from typing import Any, Callable

from openai import OpenAI

from .config import Settings, get_settings
from .providers import Provider, get_provider


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from model output."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_json(text: str) -> Any:
    """Extract and parse JSON from model output, stripping any thinking blocks first."""
    text = strip_thinking(text or "").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting from markdown code block fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    # Strip leading/trailing code fences if present
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned.strip())
        except Exception:
            pass

    # Bracket matching search
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
    def __init__(self, settings: Settings | None = None, on_llm_call: Callable[[dict], None] | None = None,
                 provider: str | None = None):
        self.s = settings or get_settings()
        prov_name = provider or self.s.llm_provider
        self.provider: Provider = get_provider(prov_name)
        if not self.provider.api_key:
            raise RuntimeError(f"API key for provider '{self.provider.name}' is not set in environment.")

        self.client = OpenAI(
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
            default_headers=self.provider.default_headers if self.provider.default_headers else None,
            timeout=90,
        )
        self.calls = 0
        self.on_llm_call = on_llm_call
        self._limiter = threading.Semaphore(max(1, self.s.llm_concurrency))

    def chat(self, system: str, user: str, *, model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 2000, tag: str = "") -> str:
        target_model = model or self.s.get_model_for_tag(tag)
        if not target_model:
            raise ValueError(f"No model resolved for tag '{tag}' with provider '{self.provider.name}'")

        # Candidate models failover chain: primary model first, then fallbacks
        candidates = [target_model]
        if self.provider.name == "openrouter" and self.s.openrouter_fallback_models:
            for fb in self.s.openrouter_fallback_models:
                if fb not in candidates:
                    candidates.append(fb)

        last: Exception | None = None
        extra_body_base: dict[str, Any] = dict(self.s.llm_extra_body)

        for current_model in candidates:
            extra_body: dict[str, Any] = dict(extra_body_base)
            if self.provider.name == "openrouter":
                rem_fallbacks = [m for m in candidates if m != current_model]
                if rem_fallbacks:
                    # OpenRouter allows at most 3 models in the 'models' array
                    extra_body["models"] = ([current_model] + rem_fallbacks)[:3]

            for attempt in range(2):
                try:
                    with self._limiter:
                        t0 = time.time()
                        kwargs: dict[str, Any] = {
                            "model": current_model,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                        if extra_body:
                            kwargs["extra_body"] = extra_body

                        try:
                            r = self.client.chat.completions.create(**kwargs)
                        except Exception as temp_err:
                            if "temperature" in str(temp_err).lower():
                                kwargs.pop("temperature", None)
                                r = self.client.chat.completions.create(**kwargs)
                            else:
                                raise temp_err

                        ms = int((time.time() - t0) * 1000)
                        self.calls += 1
                        tokens = 0
                        if hasattr(r, "usage") and r.usage:
                            tokens = getattr(r.usage, "total_tokens", 0) or 0

                        actual_model = getattr(r, "model", None) or current_model
                        if self.on_llm_call:
                            self.on_llm_call({
                                "tag": tag or "llm",
                                "model": actual_model,
                                "ms": ms,
                                "tokens": tokens,
                                "provider": self.provider.name,
                            })

                        raw_content = r.choices[0].message.content or ""
                        return strip_thinking(raw_content)

                except Exception as e:
                    last = e
                    err_msg = str(e).lower()

                    # If rate-limited or daily limit, try the next fallback model in candidates chain
                    is_rate_limit = any(term in err_msg for term in [
                        "rate limit", "429", "daily request limit", "free-model", "quota", "per-day"
                    ])
                    if is_rate_limit:
                        # Try next candidate model
                        break

                    # Check for Retry-After
                    wait_sec = 0.0
                    if hasattr(e, "response") and e.response is not None:
                        hdr = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
                        if hdr:
                            try:
                                wait_sec = float(hdr)
                            except ValueError:
                                pass

                    if wait_sec > 0:
                        time.sleep(min(wait_sec, 15.0))
                    else:
                        time.sleep(1.0 + random.uniform(0.1, 0.5))

        # If all candidates in chain failed:
        err_msg = str(last).lower() if last else ""
        if any(phrase in err_msg for phrase in [
            "daily request limit",
            "free-model daily request limit",
            "free-models-per-day",
            "daily limit",
            "per-day limit",
            "quota exceeded",
            "credit_balance_exhausted",
            "spend_limit_exceeded",
            "rate limit exceeded: free",
        ]):
            raise RuntimeError(
                "free-model daily request limit reached; try later or switch provider"
            ) from last

        raise RuntimeError(f"LLM call failed [{tag}] via {self.provider.name}: {last}")

    def json(self, system: str, user: str, *, model: str | None = None,
             tag: str = "", max_tokens: int = 2000) -> Any:
        sys_json = system + "\n\nRespond with valid JSON only: no prose, no code fences, no reasoning tags."
        raw = self.chat(sys_json, user, model=model, temperature=0.1, max_tokens=max_tokens, tag=tag)
        try:
            return extract_json(raw)
        except ValueError:
            raw = self.chat(
                sys_json,
                user + "\n\nYour previous reply was not valid JSON. JSON only, no reasoning.",
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                tag=tag + ":retry",
            )
            return extract_json(raw)
