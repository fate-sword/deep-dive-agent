/* ============================================================
   DEEP DIVE — landing interactions (ProofLens)
   Plain JavaScript, no dependencies, one listener set for the
   lifetime of the page. Handles: artwork fit/clamping, the
   transparent initialization control (mouse + touch + keyboard),
   duplicate-activation guards, the brief startup transition, and
   the navigation hand-off to the workspace.
   ============================================================ */
(function () {
  "use strict";

  var IMG_W = 2752;
  var IMG_H = 1536;
  var RATIO = IMG_W / IMG_H;

  var stage = document.getElementById("stage");
  var artwork = document.getElementById("artwork");
  var button = document.getElementById("init-button");
  var status = document.getElementById("status");
  var imgEl = null;

  var initializing = false;

  /* ---------- Layout: CRT screen aperture fit with 0.5cm bezel framing ---------- */

  function fitArtwork() {
    var screenEl = document.getElementById("crt-screen") || stage;
    if (!screenEl || !artwork) return;
    var availW = screenEl.clientWidth;
    var availH = screenEl.clientHeight;
    if (!(availW > 0) || !(availH > 0)) return;

    // Artwork spans full width of the CRT screen aperture (framed by the 0.5cm bezel)
    var width = availW;
    var height = width / RATIO;

    // Smart vertical positioning to ensure button & title remain fully visible inside the CRT screen:
    // Button bottom is at ~89.13% of artwork height. Title top is at ~22.8%.
    if (height > availH) {
      // Widescreen: artwork is taller than CRT screen
      var buttonBottom = height * 0.8913;
      var maxTopForButton = availH - 18 - buttonBottom; // keep button >= 18px above screen bottom
      var centerTop = (availH - height) / 2;
      var top = Math.min(0, Math.min(centerTop, maxTopForButton));
      artwork.style.top = top + "px";
      artwork.style.left = "0px";
    } else {
      // Tall / mobile screen: center vertically inside the CRT screen
      artwork.style.top = ((availH - height) / 2) + "px";
      artwork.style.left = "0px";
    }

    artwork.style.width = width + "px";
    artwork.style.height = height + "px";
  }

  /* ---------- Rendering mode: crisp upscaled, smooth downscaled ---------- */

  function updateRendering() {
    if (!imgEl || !artwork) return;
    var naturalW = imgEl.naturalWidth || IMG_W;
    var displayedW = artwork.getBoundingClientRect().width;
    if (displayedW >= naturalW * 0.99) {
      imgEl.classList.add("is-pixelated");     // upscaled/native: crisp pixel art
    } else {
      imgEl.classList.remove("is-pixelated");  // downscaled: smooth = readable text
    }
  }

  function onViewportChange() {
    fitArtwork();
    updateRendering();
  }

  /* ---------- Status messaging ---------- */

  function setStatus(message) {
    if (status) status.textContent = message || "";
  }

  /* ---------- Initialization ---------- */

  function initialize() {
    if (initializing) return;            // no duplicate activations
    if (!button) return;
    initializing = true;
    button.disabled = true;
    button.classList.add("is-pressed");
    setStatus("");
    document.body.classList.add("is-initializing");

    // Press feedback (~150ms) + CRT sweep/shutdown (~620ms), then hand off.
    window.setTimeout(function () {
      try {
        navigateToWorkspace();
      } catch (err) {
        recover(err);
      }
    }, 640);
  }

  function navigateToWorkspace() {
    // INTEGRATION POINT ------------------------------------------------
    // This is the single navigation hand-off of the landing screen.
    // It currently opens the placeholder workspace page served by
    // app/main.py. When the real research workspace route exists,
    // point this at it (e.g. "/workspace" or "/app") — nothing else in
    // this file needs to change. The research pipeline entry point is
    // app.agents.run_pipeline(...).
    window.location.assign("workspace.html");
  }

  function recover(err) {
    // Restore an interactive state and surface a small, accessible error.
    initializing = false;
    if (button) {
      button.disabled = false;
      button.classList.remove("is-pressed");
    }
    document.body.classList.remove("is-initializing");
    setStatus("Initialization failed — press Enter or click the button to try again.");
    if (window.console && console.error) console.error("Deep Dive initialization failed:", err);
  }

  /* ---------- Input wiring ---------- */

  function onActivate() {
    initialize();
  }

  function onKeyDown(event) {
    if (event.key !== "Enter" || event.repeat || initializing) return;

    var target = event.target;
    if (target) {
      // Never hijack Enter inside editable controls.
      if (target.isContentEditable) return;
      var tag = target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      // When the button itself is focused, native activation fires the
      // click event — do not also run the global handler.
      if (target === button) return;
    }

    event.preventDefault();
    initialize();
  }

  function onCleanup() {
    document.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("resize", onViewportChange);
    window.removeEventListener("orientationchange", onViewportChange);
    window.removeEventListener("pagehide", onCleanup);
    if (window.visualViewport) {
      window.visualViewport.removeEventListener("resize", onViewportChange);
    }
  }

  /* ---------- Boot ---------- */

  function boot() {
    if (!stage || !artwork || !button) return;

    button.addEventListener("click", onActivate);      // mouse, touch, pen
    document.addEventListener("keydown", onKeyDown);   // Enter anywhere
    window.addEventListener("resize", onViewportChange);
    window.addEventListener("orientationchange", onViewportChange);
    window.addEventListener("pagehide", onCleanup);
    if (window.visualViewport) {
      window.visualViewport.addEventListener("resize", onViewportChange);
    }

    imgEl = artwork.querySelector(".artwork__img");
    if (imgEl) {
      imgEl.addEventListener("error", function () {
        setStatus("Warning: artwork failed to load — initialization is still available.");
      });
      imgEl.addEventListener("load", updateRendering);
    }

    fitArtwork();
    updateRendering();
  }

  boot();
})();