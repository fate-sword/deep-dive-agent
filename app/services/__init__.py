"""ProofLens services — external integrations.

    search.py       pluggable retrieval providers
    mesh_client.py  client for the distributed research mesh
"""
from .mesh_client import MeshClient, MeshError
from .search import NullProvider, SearchService

__all__ = ["MeshClient", "MeshError", "NullProvider", "SearchService"]