# Vendored: @yogesharc/thinking-orbs

| Field    | Value                                                    |
|----------|----------------------------------------------------------|
| Package  | @yogesharc/thinking-orbs                                 |
| Version  | 0.1.2                                                    |
| License  | MIT (see thinking-orbs.LICENSE.txt)                      |
| Source   | https://github.com/yogesharc/thinking-orbs               |
| File     | thinking-orbs.js  (= dist/orb-core.js from the package) |
| Fetched  | 2026-10-27                                               |

## Why vendored
The task spec (rule 6) forbids runtime CDN loads and npm/build steps.
`orb-core.js` is the zero-dependency plain-JS build.  It exposes
`mountOrb(svgEl, opts)` and returns `{ pause, play, destroy }`.
The React wrapper `orb.js` is NOT loaded; only `thinking-orbs.js` is.

## Usage in index.html
```html
<script type="module" src="/web/vendor/thinking-orbs.js"></script>
```
In practice the file is inlined via a module shim so no module loader
is required at runtime.
