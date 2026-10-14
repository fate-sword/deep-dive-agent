"""ProofLens application package.

Layout (keep this structure when adding new utilities):

    app/
      main.py          web entry point (serves the DEEP DIVE landing screen)
      schemas.py       shared dataclasses for the research pipeline
      storage.py       JSON persistence under data/
      agents/          planner / researcher / writer / verifier
      services/        search + mesh client integrations
      static/          landing screen (index.html, css, js, assets)
"""
__all__ = ["main", "schemas", "storage"]