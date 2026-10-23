import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROLES = ("planner", "drafter", "verifier", "synthesizer")

TAG_TO_ROLE: dict[str, str] = {
    "planner": "planner",
    "researcher_refine": "planner",
    "drafter": "drafter",
    "verifier": "verifier",
    "contradictor": "verifier",
    "synthesizer": "synthesizer",
    "followup": "synthesizer",
}


def map_tag_to_role(tag: str) -> str:
    """Map agent tag to standard role."""
    t = (tag or "").lower().strip()
    return TAG_TO_ROLE.get(t, "planner")


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    verifier_model: str
    max_loops: int
    min_verified_per_subq: int
    pages_per_subq: int
    results_per_query: int
    db_path: str
    tavily_api_key: str = ""
    tavily_monthly_cap: int = 0
    research_workers: int = 3
    llm_concurrency: int = 4
    llm_provider: str = "mesh"  # "mesh" | "openrouter"
    llm_extra_body: dict[str, Any] = field(default_factory=dict)
    openrouter_fallback_models: list[str] = field(default_factory=list)
    role_models: dict[str, str] = field(default_factory=dict)

    def get_model_for_role(self, role: str) -> str:
        role = role.lower().strip()
        m = self.role_models.get(role) or self.model
        if not m and self.llm_provider == "openrouter":
            raise ValueError(
                f"No model configured for OpenRouter role '{role}'. "
                f"Set OPENROUTER_MODEL or OPENROUTER_MODEL_{role.upper()} in environment."
            )
        return m

    def get_model_for_tag(self, tag: str) -> str:
        return self.get_model_for_role(map_tag_to_role(tag))


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "mesh").lower().strip()
    if provider not in ("mesh", "openrouter"):
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Deeptrace registry strictly supports 'mesh' and 'openrouter'."
        )

    # Validate LLM_EXTRA_BODY_JSON if provided
    extra_body_raw = os.getenv("LLM_EXTRA_BODY_JSON", "").strip()
    extra_body: dict[str, Any] = {}
    if extra_body_raw:
        try:
            parsed = json.loads(extra_body_raw)
            if not isinstance(parsed, dict):
                raise ValueError("LLM_EXTRA_BODY_JSON must be a valid JSON object")
            extra_body = parsed
        except Exception as e:
            raise ValueError(f"Invalid LLM_EXTRA_BODY_JSON: {e}")

    # Fallback models for OpenRouter
    fallback_raw = os.getenv("OPENROUTER_FALLBACK_MODELS", "").strip()
    openrouter_fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()] if fallback_raw else []

    # Concurrency limit
    default_concurrency = 2 if provider == "openrouter" else 4
    concurrency_env = os.getenv("LLM_CONCURRENCY")
    llm_concurrency = int(concurrency_env) if concurrency_env else default_concurrency

    # Model resolution
    p_upper = provider.upper()
    default_model = os.getenv(f"{p_upper}_MODEL", "")
    if provider == "mesh" and not default_model:
        default_model = os.getenv("MESH_MODEL", "openai/gpt-4o-mini")

    role_models: dict[str, str] = {}
    for role in ROLES:
        role_override = os.getenv(f"{p_upper}_MODEL_{role.upper()}", "")
        if not role_override and provider == "mesh" and role == "verifier":
            role_override = os.getenv("MESH_VERIFIER_MODEL", "")
        role_models[role] = role_override or default_model

    # Base URL & API key
    if provider == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    else:
        base_url = os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1")
        api_key = os.getenv("MESH_API_KEY", "")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=default_model,
        verifier_model=role_models.get("verifier", default_model),
        max_loops=int(os.getenv("MAX_LOOPS", "2")),
        min_verified_per_subq=int(os.getenv("MIN_VERIFIED_PER_SUBQ", "2")),
        pages_per_subq=int(os.getenv("PAGES_PER_SUBQ", "4")),
        results_per_query=int(os.getenv("RESULTS_PER_QUERY", "6")),
        db_path=os.getenv("DB_PATH", "data/runs.db"),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        tavily_monthly_cap=int(os.getenv("TAVILY_MONTHLY_CAP", "0")),
        research_workers=int(os.getenv("RESEARCH_WORKERS", "3")),
        llm_concurrency=llm_concurrency,
        llm_provider=provider,
        llm_extra_body=extra_body,
        openrouter_fallback_models=openrouter_fallback_models,
        role_models=role_models,
    )
