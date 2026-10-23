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


def get_settings() -> Settings:
    model = os.getenv("MESH_MODEL", "openai/gpt-4o-mini")
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
    )
