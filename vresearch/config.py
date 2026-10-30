"""Pipeline configuration and runtime settings."""
import json
import os
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

load_dotenv()

ROLES = ("planner", "researcher", "drafter", "verifier", "synthesizer")

TAG_TO_ROLE: dict[str, str] = {
    "planner": "planner",
    "researcher": "researcher",
    "drafter": "drafter",
    "verifier": "verifier",
    "contradictor": "verifier",
    "synthesizer": "synthesizer",
    "followup": "synthesizer",
}


def map_tag_to_role(tag: str) -> str:
    cleaned = tag.split(":")[0].lower().strip()
    return TAG_TO_ROLE.get(cleaned, "planner")


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
    llm_provider: str = "mesh"
    relevance_model: str = ""
    show_filtered_in_trail: bool = True
    depth: str = "std"  # "quick" | "std" | "deep"
    llm_extra_body: dict[str, Any] = field(default_factory=dict)
    role_models: dict[str, str] = field(default_factory=dict)

    def get_model_for_role(self, role: str) -> str:
        role = role.lower().strip()
        return self.role_models.get(role) or self.model

    def get_model_for_tag(self, tag: str) -> str:
        return self.get_model_for_role(map_tag_to_role(tag))


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "mesh").lower().strip()
    if provider != "mesh":
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Deeptrace registry strictly supports 'mesh'."
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

    # Concurrency limit
    concurrency_env = os.getenv("LLM_CONCURRENCY")
    llm_concurrency = int(concurrency_env) if concurrency_env else 4

    # Model resolution
    default_model = os.getenv("MESH_MODEL", "z-ai/glm-5.3-flash")

    role_models: dict[str, str] = {}
    for role in ROLES:
        role_override = os.getenv(f"MESH_MODEL_{role.upper()}", "")
        if not role_override and role == "verifier":
            role_override = os.getenv("MESH_VERIFIER_MODEL", "")
        role_models[role] = role_override or default_model

    # Base URL & API key
    base_url = os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1")
    api_key = os.getenv("MESH_API_KEY", "")

    relevance_model = os.getenv("MESH_RELEVANCE_MODEL") or os.getenv("MESH_MODEL_RELEVANCE") or role_models.get("verifier", default_model)
    show_filtered = os.getenv("SHOW_FILTERED_IN_TRAIL", "1").lower() in ("1", "true", "yes")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=default_model,
        verifier_model=role_models.get("verifier", default_model),
        relevance_model=relevance_model,
        show_filtered_in_trail=show_filtered,
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
        role_models=role_models,
    )
