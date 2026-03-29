# Splatt2 Changelog

## v1.1.0 (2026-03-29)

### Tracking improvements
- **Software gain normalisation** — before CLAHE, the greyscale frame is scaled so its mean brightness hits a configurable target (default 128). Compensates automatically for clouds, time of day, and lighting changes with no driver cooperation needed. Configurable in Settings → Camera → Performance → Brightness target (quick buttons: 64 / 96 / 128 / 160 / 192).
- **Velocity spike filter** — rejects bad homography readings caused by a marker being briefly lost. Detects the characteristic sharp reversal of a false position (teleports out, immediately snaps back) without rejecting genuine fast movement such as recoil or pistol swing. Two parameters in Settings → Camera → Spike Filter:
  - **Spike velocity (mm/frame)** — minimum speed to flag as a spike candidate (default 25mm/frame; quick buttons 15/20/25/35/50)
  - **Reversal ratio (0–1)** — how sharply the direction must reverse to confirm a spike (default 0.7; quick buttons 0.5/0.6/0.7/0.8/0.9)
  - Filter operates on raw positions before the smoother — the smoother is never contaminated, eliminating the slow relaxation artefact after a spike.
- **CLAHE clip limit** — now user-configurable in Settings → Camera → Performance (quick buttons: 2 / 4 / 6 / 8 / 12; default raised from 2.0 to 4.0). Apply rebuilds the tracker immediately.

### Multi-marker support (4 / 6 / 8 markers)
- **Marker count dropdown** in Settings → Camera → ArUco: choose 4, 6, or 8 markers. More markers improve homography robustness — tracking continues if one or two are briefly occluded.
  - 4 = corner markers only (original behaviour)
  - 6 = corners + left/right edge midpoints (IDs 4 & 5)
  - 8 = corners + left/right + top/bottom edge midpoints (IDs 6 & 7)
- **Marker count dropdown** in Marker Sheet dialog — generate sheets matching your tracker setting. IDs are fixed so existing printed sheets remain compatible.
- **Dynamic tracking quality** — quality bar now shows percentage of configured markers detected (was always out of 4 regardless of marker count).

### Camera
- **Pixel format dropdown** — Settings → Camera → ArUco → Pixel format: Auto / MJPEG / YUY2. MJPEG prevents frame rate throttling on static scenes (some cameras halve fps when the scene isn't changing). Console confirms whether the camera accepted the format.
- **🎛 Cam Props button** — opens the Windows native DirectShow camera properties dialog (brightness, contrast, saturation, sharpness, hue, gamma, white balance, exposure). Start the camera first. Adjusting contrast, sharpness, and saturation in this dialog can significantly improve ArUco detection.

### Focus assist
- **◎ Focus assist toggle** in the camera panel — reveals a live sharpness bar (Laplacian variance) with peak-hold indicator. Turn the focus ring until the bar peaks and shows green. Zero CPU cost when off. Designed for manual-focus lenses.

### Performance
- **Trace colour caching** — trace point colours are pre-computed at append time and cached, rather than recalculated on every frame. Eliminates FPS degradation during long holds (previously O(n) colour calculations per frame; now O(1) per frame regardless of trace length). Scales to 120fps × 30-second holds with no slowdown. Cache is invalidated automatically if colour settings change.

### Scoring fix
- Shot score and shot hole now always use the same retroactively interpolated position — where the crosshair was at the exact moment of audio detection. Previously, scoring used the live frame position while drawing used the interpolated position, causing a visible mismatch on some shots.

---

## v1.0.0 — GitHub initial release (2026-03-25)
- ArUco marker tracking with homography (4-marker sheet, DICT_4X4_50 default)
- CLAHE adaptive contrast enhancement for ArUco detection
- Retroactive shot timing: audio timestamp interpolated against trace history
- Geometry-based scoring: R = card_radius + calibre_radius, independent of visual ring boundaries
  - Integer: 10 equal bands (10→1), miss = 0
  - Decimal: 99 equal bands (10.9→1.0), miss = 0
- Pellet calibre user-configurable (affects all scoring bands dynamically)
- 5-tab Settings dialog: Camera, Audio, Target, Colours, Advanced
- Target editor: create/edit/delete targets from within the app, live hot-reload
- Three built-in targets: 10m Air Rifle (ISSF), 10m Air Pistol (ISSF), 6 Yard Air Rifle (NSRA)
- 5-mark target (NSRA competition card): Option C scoring — each shot assigned to nearest mark
- Daily session subfolders, companion JSON for full trace data
- Series Review window: two-dropdown day/series picker, per-shot checkboxes, statistics
- Zoom slider on target canvas (0.30× to 1.50×)
- Marker sheet generator with per-target calibre and ArUco dictionary selection
- Camera rotation (0/90/180/270°), flip, CLAHE, EMA/Savitzky-Golay smoothing
- Zero offset (persistent) and fine zero (click-on-canvas)
- Approach zone rejection of false positives
- Trace colour zones: approach / hold / pre-shot / final
- ACP (Aim Centre Point), MPI, group circle, bounding boxes
- Statistics: MR, ES, FOM, CEP, Std X/Y, MPI X/Y, best/worst
