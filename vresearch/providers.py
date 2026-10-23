"""LLM Provider abstraction and registry for Deeptrace.
Registry contains exactly two providers: 'mesh' (production default) and 'openrouter' (dev-only).
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


ALLOWED_PROVIDERS = ("mesh", "openrouter")


def get_provider(name: str | None = None) -> Provider:
    """Resolve and validate the requested provider according to guardrails."""
    prov = (name or os.getenv("LLM_PROVIDER", "mesh")).lower().strip()
    if prov not in ALLOWED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{prov}'. Deeptrace registry strictly supports 'mesh' and 'openrouter'."
        )

    deploy_env = os.getenv("DEPLOY_ENV", "").lower().strip()
    if deploy_env == "production" and prov != "mesh":
        raise RuntimeError(
            f"Deploy environment is production (DEPLOY_ENV=production). Refusing non-mesh provider '{prov}'. "
            "All production calls must route through Mesh API."
        )

    if prov == "openrouter":
        allow_dev = os.getenv("ALLOW_DEV_PROVIDER", "").lower().strip()
        if allow_dev not in ("1", "true", "yes"):
            raise RuntimeError(
                "Provider 'openrouter' is dev-only and requires ALLOW_DEV_PROVIDER=1 in environment."
            )
        headers: dict[str, str] = {}
        referer = os.getenv("OPENROUTER_REFERER", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_TITLE", "Deeptrace").strip()
        if title:
            headers["X-Title"] = title

        return Provider(
            name="openrouter",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            default_headers=headers,
            dev_only=True,
        )

    # Mesh provider (default)
    return Provider(
        name="mesh",
        base_url=os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1"),
        api_key=os.getenv("MESH_API_KEY", ""),
        default_headers={},
        dev_only=False,
    )
