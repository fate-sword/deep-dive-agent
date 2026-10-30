"""LLM Provider abstraction and registry for Deeptrace.
Mesh API is the production provider.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    default_headers: dict[str, str] = field(default_factory=dict)
    dev_only: bool = False


ALLOWED_PROVIDERS = ("mesh",)


def get_provider(name: str | None = None) -> Provider:
    """Resolve and validate the requested provider according to guardrails."""
    prov = (name or os.getenv("LLM_PROVIDER", "mesh")).lower().strip()
    if prov not in ALLOWED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{prov}'. Deeptrace registry strictly supports 'mesh'."
        )

    # Mesh provider (default)
    return Provider(
        name="mesh",
        base_url=os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1"),
        api_key=os.getenv("MESH_API_KEY", ""),
        default_headers={},
        dev_only=False,
    )
