# ProofLens — DEEP DIVE Landing Screen

**ProofLens** is a research-agent application ("go beyond the surface").
Right now the project contains the interactive **DEEP DIVE landing screen**
(booting a retro research terminal) plus the backend **skeleton** from
`struccurelayout.jpg` — agents/services are documented placeholders you will
fill in later. No research backend is implemented or pretended.

## Run it

From the project root (needs Python ≥ 3.10; the landing page itself is
plain HTML/CSS/JS and has zero runtime dependencies):

```bash
# with the web stack (see requirements.txt)
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
# → http://127.0.0.1:8000/          (landing screen)
# → http://127.0.0.1:8000/workspace.html

# zero-dependency fallback (stdlib only)
python -m app.main
```

You can also open `app/static/index.html` directly in a browser — the
landing page works from `file://` as well.

### Can't access the website?

- The page is served by a **local server** — it only answers while a server
  process is running. If the browser says "unable to connect", the server is
  not running: double-click **`start.bat`** (or run `python -m app.main`
  from the project root) and keep the server window open while you browse.
- The address is `http://127.0.0.1:8000/` or `http://localhost:8000/` —
  this is a local address, not a public website.
- Port already taken by something else? Set `PORT=8001` in your `.env`
  (see `.env.example`) and open `http://127.0.0.1:8001/`.
- A server started from an assistant/tool session dies when that session's
  terminal closes — always use `start.bat` (or your own terminal) for a
  server you want to keep using.

## Project structure

```
prooflens/
├── app/
│   ├── main.py            web entry point: serves the landing screen,
│   │                      /api/health, and static files (FastAPI, with a
│   │                      stdlib fallback server so it runs anywhere)
│   ├── agents/            planner / researcher / writer / verifier
│   │                      (documented placeholder agents — integration
│   │                      points are marked in each file's docstring)
│   ├── services/          search + mesh_client (pluggable providers)
│   ├── storage.py         JSON persistence under data/
│   ├── schemas.py         shared dataclasses for the pipeline
│   ├── __init__.py
│   └── static/            ★ the DEEP DIVE landing screen
│       ├── index.html     landing page
│       ├── workspace.html placeholder hand-off page (integration point)
│       ├── css/landing.css
│       ├── js/landing.js
│       └── assets/deep-dive-landing.png   (the supplied artwork)
├── tests/                 pytest tests for the skeleton (run: python -m pytest)
├── data/                  runtime JSON sessions (empty; git-kept)
├── .env.example           copy to .env, fill in keys later
├── .gitignore
├── requirements.txt
├── start.bat              double-click: starts the server + opens the browser
└── README.md
```

## The landing screen

`app/static/index.html` displays the supplied artwork **as the complete
visual design** — nothing is redrawn, no extra imagery, no panels. All
motion is CSS/JS around that single image.

### Entrance & atmosphere

- **Entrance** (`.artwork`): fade from black over ~700 ms with a subtle
  1.02 → 1.00 scale; the complete design is readable as soon as it lands.
- **CRT scanlines** (`.fx--scanlines`): 1px repeating gradient at ~35 %
  opacity over the whole stage; never intercepts pointer events.
- **Faint flicker**: the scanline layer's opacity drifts ±0.04 over 7 s —
  restrained, no strobing.
- **Ambient brightness**: the image's brightness breathes 1.00 → 1.02 over
  9 s.
- **Blinking block cursor** (`.cursor-blink`): a small navy block in the
  blank space right after "…INITIALIZE" (percentage-positioned, scales with
  the artwork).

All of the above are disabled by `prefers-reduced-motion: reduce` (static
image, static cursor, quick plain fade on initialization instead of the
sweep).

### Button alignment (how the invisible button tracks the artwork)

The "PRESS ENTER TO INITIALIZE" control is **part of the image**, so a real
`<button aria-label="Initialize Deep Dive">` is layered invisibly over it:

- The control strip was measured from the artwork (2752×1536):
  `x 1040–1686, y 1294–1356`; the padded hit area is
  `x 1020–1700, y 1281–1369`.
- `.artwork` is an `aspect-ratio: 2752/1536` box sized to fit both viewport
  axes, so **the artwork box is exactly the displayed image bounds** —
  there is no letterboxing to compensate for.
- The hotspot therefore uses plain percentages of the image:
  `left 37.06 % / top 83.40 % / width 24.71 % / height 5.73 %` and stays
  aligned at every viewport size and orientation.
- `js/landing.js::fitArtwork()` clamps the box to the stage content area
  (safe-area padding, odd viewports) on `resize` / `orientationchange` /
  `visualViewport` changes.

### Artwork text cleanup (subtitle + description)

The supplied artwork had muddy anti-aliasing halos around two text areas
("GO BEYOND THE SURFACE" and the "INVESTIGATE COMPLEX QUESTIONS…"
description lines), which made them look blurry. They were cleaned
**in the image itself** (letters, font and layout unchanged):

- Only pixels within a 3 px collar around the letter strokes were touched.
- Each halo pixel was snapped to either the band's ink tone or the true
  local background tone (sampled just outside the halo), which makes every
  edge crisp and high-contrast.
- Background gradients, clouds, waves and all other artwork are untouched;
  no seams were introduced (verified row-by-row against the original).
- The untouched master is the original file in the user's Pictures folder
  (`Camera Roll/Gemini_Generated_Image_cv2dsycv2dsycv2d.png`).

Note: after replacing the artwork, do a hard refresh (**Ctrl+F5**) in the

### High-readability typography layer (below DEEP DIVE)

To ensure the fonts below "DEEP DIVE" are razor-sharp, accessible, and legible at any resolution or viewport size:

- **Vector typography**: Rather than relying purely on rasterized text downscaled by the browser, `app/static/index.html` renders vector typography elements (`.artwork__subtitle`, `.artwork__desc-box`, `.hotspot`) layered with precision over the artwork bounds.
- **Fonts**: Loaded via Google Fonts (`Press Start 2P` for retro pixel headers and `Plus Jakarta Sans` for crisp text clarity) with robust system fallbacks (`ui-monospace`, `Consolas`, `system-ui`).
- **Contrast HUD plate**: The description ("Investigate complex questions, discover credible sources, and turn scattered information into structured insight.") sits in a semi-translucent terminal glass plate (`rgba(3, 14, 40, 0.84)`) with backdrop blur and cyan border, providing a 14:1+ WCAG AAA contrast ratio and eliminating visual overlap with the artwork background.
- **CRT scanlines balance**: Scanlines run at `z-index: 20` with tuned opacity (`0.18`), maintaining atmospheric retro CRT texture across the background while keeping the typography layer (`z-index: 35`) 100% legible.
- **Scalable sizing**: Uses CSS container queries (`cqw`) and `clamp()` minimums, guaranteeing text remains comfortably readable across 4K displays, laptops, tablets, and phones without breaking layout bounds.

### What pressing Enter / clicking does

1. Mouse/touch: `click` on the hotspot. Keyboard: native `Space`/`Enter` on
   the focused button, or **Enter anywhere** on the page (ignored inside
   `input`/`textarea`/`select`/`contenteditable`, ignored on key
   auto-repeat).
2. Duplicate activations are impossible: an `initializing` flag, a
   `disabled` button, and an `event.repeat` guard.
3. Press feedback (~150 ms dip) → CRT-style sweep + fade (~620 ms) →
   `navigateToWorkspace()` → `workspace.html`.
4. `navigateToWorkspace()` in `app/static/js/landing.js` is the **single
   integration point**: point it at the real workspace route when one
   exists. `workspace.html` is an honest placeholder — it does not claim
   any backend was initialized.
5. On failure the button is restored and a polite `role="status"` message
   appears — the screen is never dead-ended.

## Integration points for later work

- `app/agents/*` — implement real planning/retrieval/writing/verification;
  `app/agents/__init__.py::run_pipeline(...)` is the pipeline entry.
- `app/services/search.py` — wire a real search provider
  (`SEARCH_PROVIDER`, `SEARCH_API_KEY`).
- `app/services/mesh_client.py` — point at the research mesh
  (`MESH_SERVICE_URL`).
- `app/main.py` — add API routes; the pipeline + storage are ready to be
  exposed.
- `app/static/js/landing.js::navigateToWorkspace()` — final hand-off of
  the landing screen.

## Checks performed

- `python -m compileall` over `app/` and `tests/` (syntax clean).
- `python -m pytest` — tests pass (verifier heuristics, pipeline with
  stub/null services, storage roundtrip).
- HTTP smoke test: `/`, `/workspace.html`, `/css/landing.css`,
  `/js/landing.js`, `/assets/deep-dive-landing.png`, `/api/health` all
  served correctly.

## Known limitations

- `app/static/assets/deep-dive-landing.png` is the supplied 6.3 MB PNG,
  used as-is (one image, no optimization/derivative assets, per spec).
- The verifier/planner heuristics are intentionally simple placeholders —
  not a functional research pipeline.
- `uvicorn[standard]` extras are omitted from requirements (some lack
  wheels on brand-new Python versions); plain `uvicorn` is sufficient.