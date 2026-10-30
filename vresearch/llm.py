"""Single entry point for every LLM call via Mesh API gateway."""
import json
import os
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


def check_outer_unclosed(text: str) -> bool:
    """Check if the text begins an outer JSON structure ('{' or '[') that was never closed."""
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace == -1 and first_bracket == -1:
        return False
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        open_c, close_c, start = "{", "}", first_brace
    else:
        open_c, close_c, start = "[", "]", first_bracket

    depth = 0
    in_str = False
    esc = False
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
                return False
    return depth > 0


def extract_json(text: str) -> Any:
    """Extract and parse JSON from model output, stripping any thinking blocks first."""
    text = strip_thinking(text or "").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find all code fences ```json ... ``` and try to parse each, prioritizing dicts with known keys or largest JSON
    fence_matches = re.findall(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for fm in reversed(fence_matches):
        try:
            val = json.loads(fm.strip())
            if isinstance(val, dict):
                return val
        except Exception:
            pass
    for fm in fence_matches:
        try:
            return json.loads(fm.strip())
        except Exception:
            pass

    # Strip code fences if whole text is fenced
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned.strip())
        except Exception:
            pass

    # If the text has an unclosed outer structure, it was truncated mid-generation.
    # Do not return partial inner sub-objects from truncated text.
    if check_outer_unclosed(cleaned):
        raise ValueError("truncated JSON: unclosed outer structure in model output")

    # Search for JSON blocks in text: collect all valid JSON candidate structures
    candidates = []
    for open_c, close_c in (("{", "}"), ("[", "]")):
        idx = 0
        while True:
            start = text.find(open_c, idx)
            if start == -1:
                break
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
                        candidate_str = text[start : i + 1]
                        try:
                            parsed = json.loads(candidate_str)
                            candidates.append((parsed, len(candidate_str)))
                        except Exception:
                            try:
                                fixed = re.sub(r",\s*([\}\]])", r"\1", candidate_str)
                                parsed = json.loads(fixed)
                                candidates.append((parsed, len(fixed)))
                            except Exception:
                                pass
                        break
            idx = start + 1

    if candidates:
        dict_candidates = [c for c in candidates if isinstance(c[0], dict)]

        # 1. If any dict candidate has 'sub_questions', 'claims', or 'evaluations', return the largest one
        key_matches = [c for c in dict_candidates if any(k in c[0] for k in ("sub_questions", "claims", "evaluations", "contradictions"))]
        if key_matches:
            return max(key_matches, key=lambda x: x[1])[0]

        # 2. Check list candidates for evaluations or claims
        list_candidates = [c for c in candidates if isinstance(c[0], list)]
        for lc in list_candidates:
            items = lc[0]
            if items and isinstance(items, list) and isinstance(items[0], dict):
                if any("cid" in it and "relevance" in it for it in items):
                    return {"evaluations": items}
                if any("evidence_ids" in it for it in items):
                    return {"claims": items}

        # 3. If multiple separate sub-question dicts were generated across blocks, aggregate them
        q_dicts = [c[0] for c in dict_candidates if "question" in c[0] and ("tag" in c[0] or "label" in c[0] or "id" in c[0])]
        if len(q_dicts) >= 2:
            seen = set()
            deduped = []
            for qd in q_dicts:
                key = qd.get("id") or qd.get("question")
                if key not in seen:
                    seen.add(key)
                    deduped.append(qd)
            return {"sub_questions": deduped}

        # 4. Otherwise return the largest dict candidate
        if dict_candidates:
            return max(dict_candidates, key=lambda x: x[1])[0]

        # 5. Otherwise return the largest candidate
        return max(candidates, key=lambda x: x[1])[0]

    # 6. Regex fallback for evaluations if model used markdown list instead of JSON
    eval_matches = re.findall(
        r"\[?(temp_[a-zA-Z0-9_-]+)\]?[\s:\-]+(direct|off_topic|relevant|irrelevant)\b(?:[\s:\-]+([^\n\r]+))?",
        text,
        re.IGNORECASE
    )
    if eval_matches:
        evals = []
        for cid, rel, rsn in eval_matches:
            v = "direct" if "direct" in rel.lower() or "relevant" in rel.lower() else "off_topic"
            evals.append({"cid": cid, "relevance": v, "reason": (rsn or v).strip()})
        return {"evaluations": evals}

    raise ValueError("no valid JSON found in model output")


_DAILY_EXHAUSTED_MODELS: set[str] = set()


class LLM:
    def __init__(self, settings: Settings | None = None, on_llm_call: Callable[[dict], None] | None = None,
                 provider: str | None = None):
        self.s = settings or get_settings()
        prov_name = provider or self.s.llm_provider
        self.provider: Provider = get_provider(prov_name)
        if self.s and self.s.api_key and not self.provider.api_key:
            self.provider.api_key = self.s.api_key
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
             temperature: float = 0.2, max_tokens: int = 2000, tag: str = "",
             retry_reason: str = "") -> str:
        target_model = model or (self.s.get_model_for_tag(tag) if hasattr(self.s, "get_model_for_tag") else None) or getattr(self.s, "model", None) or os.getenv("MESH_MODEL", "z-ai/glm-5.3-flash")
        if not target_model:
            raise ValueError(f"No model resolved for tag '{tag}' with provider '{self.provider.name}'")

        # Candidate models failover chain: primary model first
        all_candidates = [target_model]

        # Prioritize candidates that have not exhausted their daily quota
        candidates = [m for m in all_candidates if m not in _DAILY_EXHAUSTED_MODELS]
        if not candidates:
            # If all candidates were marked exhausted, retry all in case quota reset
            candidates = list(all_candidates)

        last: Exception | None = None
        extra_body_base: dict[str, Any] = dict(self.s.llm_extra_body)

        current_max_tokens = max_tokens
        current_tag = tag
        current_retry_reason = retry_reason

        for current_model in candidates:
            extra_body: dict[str, Any] = dict(extra_body_base)
            # Candidate failover is handled cleanly in Python across the candidate chain
            extra_body.pop("models", None)

            # Ensure no invalid reasoning config causes 422 on Mesh API
            if self.provider.name == "mesh":
                if "reasoning" in extra_body:
                    r_conf = extra_body.get("reasoning")
                    if isinstance(r_conf, dict) and r_conf.get("max_tokens", 0) < 1024:
                        extra_body.pop("reasoning", None)

            # Per-call thinking policy: disable or minimize reasoning overhead for GLM on Mesh
            if self.provider.name == "mesh" and "glm" in current_model.lower():
                if any(current_tag.startswith(prefix) for prefix in ("planner", "researcher_refine", "relevance")):
                    extra_body["reasoning_effort"] = "none"
                elif any(current_tag.startswith(prefix) for prefix in ("drafter", "verifier", "synthesizer")):
                    extra_body["reasoning_effort"] = "low"

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
                            "max_tokens": current_max_tokens,
                        }
                        if extra_body:
                            kwargs["extra_body"] = extra_body

                        try:
                            r = self.client.chat.completions.create(**kwargs)
                        except Exception as temp_err:
                            err_str = str(temp_err).lower()
                            if "temperature" in err_str:
                                kwargs.pop("temperature", None)
                                r = self.client.chat.completions.create(**kwargs)
                            elif "reasoning" in err_str or "extra_body" in err_str:
                                kwargs.pop("extra_body", None)
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
                            call_meta = {
                                "tag": current_tag or "llm",
                                "model": actual_model,
                                "ms": ms,
                                "tokens": tokens,
                                "provider": self.provider.name,
                            }
                            if current_retry_reason:
                                call_meta["reason"] = current_retry_reason
                            self.on_llm_call(call_meta)

                        choice = r.choices[0] if (r.choices and len(r.choices) > 0) else None
                        msg = getattr(choice, "message", None) if choice else None
                        finish_reason = getattr(choice, "finish_reason", None) if choice else None
                        self.last_finish_reason = finish_reason
                        raw_content = ""
                        if msg:
                            raw_content = getattr(msg, "content", "") or ""
                            r_cnt = getattr(msg, "reasoning_content", None)
                            if not raw_content and isinstance(r_cnt, str):
                                raw_content = r_cnt

                        stripped = strip_thinking(raw_content)

                        if not stripped and attempt < max_attempts - 1:
                            if finish_reason == "length":
                                # Output is empty because token limit was hit: retry must use higher max_tokens
                                current_max_tokens = min(8192, max(int(current_max_tokens * 1.5), current_max_tokens + 2000))
                            else:
                                current_max_tokens = min(8192, current_max_tokens + 500)
                            current_tag = f"{tag}:retry" if not (tag and tag.endswith(":retry")) else tag
                            current_retry_reason = "empty output"
                            time.sleep(1.0)
                            continue

                        return stripped

                except Exception as e:
                    last = e
                    err_msg = str(e).lower()

                    # Check for account-level daily limit
                    is_daily_limit = any(term in err_msg for term in [
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
            if getattr(self, "last_finish_reason", None) == "length":
                raise ValueError("model response truncated by token limit (finish_reason='length')")
            return extract_json(raw)
        except ValueError:
            retry_max_tokens = min(8192, max(max_tokens * 2, 4000))
            retry_tag = tag + ":retry" if not (tag and tag.endswith(":retry")) else tag
            retry_reason = "length limit reached" if getattr(self, "last_finish_reason", None) == "length" else ("empty output" if not (raw and raw.strip()) else "invalid JSON")
            raw = self.chat(
                sys_json,
                user + "\n\nYour previous reply was not valid JSON. JSON only, no reasoning.",
                model=model,
                temperature=0.0,
                max_tokens=retry_max_tokens,
                tag=retry_tag,
                retry_reason=retry_reason,
            )
            if getattr(self, "last_finish_reason", None) == "length":
                raise ValueError("retry model response truncated by token limit (finish_reason='length')")
            return extract_json(raw)
