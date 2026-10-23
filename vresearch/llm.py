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


_DAILY_EXHAUSTED_MODELS: set[str] = set()


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

        # Candidate models failover chain: primary model first, then configured fallbacks
        all_candidates = [target_model]
        if self.provider.name == "openrouter":
            if self.s.openrouter_fallback_models:
                for fb in self.s.openrouter_fallback_models:
                    if fb not in all_candidates:
                        all_candidates.append(fb)
            # Ensure XingChen (Phala pool) is in failover chain as non-OpenRouter-pooled fallback
            emergency_model = "xingchen-agi/xing4.0-29b-a4b:free"
            if emergency_model not in all_candidates:
                all_candidates.append(emergency_model)

        # Prioritize candidates that have not exhausted their daily quota
        candidates = [m for m in all_candidates if m not in _DAILY_EXHAUSTED_MODELS]
        if not candidates:
            # If all candidates were marked exhausted, retry all in case quota reset
            candidates = list(all_candidates)

        last: Exception | None = None
        extra_body_base: dict[str, Any] = dict(self.s.llm_extra_body)

        for current_model in candidates:
            extra_body: dict[str, Any] = dict(extra_body_base)
            if self.provider.name == "openrouter":
                # Only pass non-exhausted fallbacks to OpenRouter, max 2 fallbacks (1 current + 2 fallbacks = 3 models max)
                valid_fallbacks = [m for m in candidates if m != current_model and m not in _DAILY_EXHAUSTED_MODELS][:2]
                if valid_fallbacks:
                    extra_body["models"] = [current_model] + valid_fallbacks
                else:
                    extra_body.pop("models", None)

            # Up to 4 attempts for transient upstream pool rate limits (e.g. Phala on XingChen)
            max_attempts = 4 if "xingchen" in current_model.lower() else 2

            for attempt in range(max_attempts):
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

                        msg = r.choices[0].message if (r.choices and len(r.choices) > 0) else None
                        raw_content = ""
                        if msg:
                            raw_content = getattr(msg, "content", "") or ""
                            if not raw_content and hasattr(msg, "reasoning_content"):
                                raw_content = getattr(msg, "reasoning_content") or ""

                        if not raw_content and attempt < max_attempts - 1:
                            time.sleep(1.0)
                            continue

                        return strip_thinking(raw_content)

                except Exception as e:
                    last = e
                    err_msg = str(e).lower()

                    # Check for account-level daily limit (e.g. 50 free requests per day reached on OpenRouter)
                    is_daily_limit = any(term in err_msg for term in [
                        "free-models-per-day",
                        "openrouter_free_tier_daily",
                        "daily request limit",
                        "daily limit",
                        "per-day limit",
                        "credit_balance_exhausted",
                    ])
                    if is_daily_limit:
                        _DAILY_EXHAUSTED_MODELS.add(current_model)
                        # Immediately break to next candidate model (do not sleep or retry this model)
                        break

                    # Check for upstream temporary shared pool limit (e.g. Phala shared pool for XingChen)
                    is_upstream_busy = any(term in err_msg for term in [
                        "temporarily rate-limited",
                        "upstream_provider_shared_pool",
                        "provider returned error",
                    ])
                    if is_upstream_busy and attempt < max_attempts - 1:
                        wait_sec = 2.0 * (attempt + 1) + random.uniform(0.1, 0.5)
                        time.sleep(wait_sec)
                        continue

                    # Retry-After header
                    wait_sec = 0.0
                    if hasattr(e, "response") and e.response is not None:
                        hdr = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
                        if hdr:
                            try:
                                wait_sec = float(hdr)
                            except ValueError:
                                pass

                    if wait_sec > 0:
                        time.sleep(min(wait_sec, 10.0))
                    elif attempt < max_attempts - 1:
                        time.sleep(1.0 + random.uniform(0.1, 0.4))

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
