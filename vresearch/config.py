import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
    llm_provider: str = "mesh"  # "mesh" | "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    def get_provider_config(self, provider: str | None = None) -> tuple[str, str, str, str]:
        """Returns (api_key, base_url, model, verifier_model) based on selected provider."""
        p = (provider or self.llm_provider or "mesh").lower()
        if p == "openai":
            key = self.openai_api_key or self.api_key
            url = self.openai_base_url
            model = self.model.replace("openai/", "") if self.model.startswith("openai/") else self.model
            v_model = self.verifier_model.replace("openai/", "") if self.verifier_model.startswith("openai/") else self.verifier_model
            return key, url, model, v_model
        else:
            # Default to Mesh API
            key = self.api_key
            url = self.base_url
            model = self.model if "/" in self.model else f"openai/{self.model}"
            v_model = self.verifier_model if "/" in self.verifier_model else f"openai/{self.verifier_model}"
            return key, url, model, v_model


def get_settings() -> Settings:
    model = os.getenv("MESH_MODEL", "openai/gpt-4o-mini")
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") and not os.getenv("MESH_API_KEY") else "mesh"

    return Settings(
        api_key=os.getenv("MESH_API_KEY", ""),
        base_url=os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1"),
        model=model,
        verifier_model=os.getenv("MESH_VERIFIER_MODEL", model),
        max_loops=int(os.getenv("MAX_LOOPS", "2")),
        min_verified_per_subq=int(os.getenv("MIN_VERIFIED_PER_SUBQ", "2")),
        pages_per_subq=int(os.getenv("PAGES_PER_SUBQ", "4")),
        results_per_query=int(os.getenv("RESULTS_PER_QUERY", "6")),
        db_path=os.getenv("DB_PATH", "data/runs.db"),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        tavily_monthly_cap=int(os.getenv("TAVILY_MONTHLY_CAP", "0")),
        research_workers=int(os.getenv("RESEARCH_WORKERS", "3")),
        llm_concurrency=int(os.getenv("LLM_CONCURRENCY", "4")),
        llm_provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
