"""
core/marker_sheet.py
Generates a printable A4 marker sheet with:
  - Four ArUco corner markers
  - A distance-scaled aiming mark (black) in the centre
  - Optional scoring ring guides

The aiming mark (the black) is scaled as a percentage of its reference size
at the nominal distance. E.g. 10m air rifle black = 30.5mm dia at 10m;
at 5m it should be 15.25mm dia (50% scale).

Scaling formula:
    aiming_mark_dia_mm = reference_dia_mm * (print_distance_m / reference_distance_m)
"""

import cv2
import numpy as np
import os

# A4 at 300 DPI
A4_W_MM  = 210.0
A4_H_MM  = 297.0
DPI      = 300
A4_W_PX  = 2480
A4_H_PX  = 3508
MM_TO_PX = DPI / 25.4

MARGIN_MM  = 8.0
MARKER_MM  = 40.0


# ── Target aiming mark definitions ───────────────────────────────────────────
# Each entry:
#   name              : display name
#   reference_dist_m  : nominal distance this target is designed for
#   aiming_mark_dia_mm: diameter of the black aiming mark at reference distance
#   outer_ring_dia_mm : diameter of the full scoring card / outer ring
#   rings_dia_mm      : list of ring diameters (innermost first) for guide circles
#   ring_labels       : score labels for each ring

def _get_aiming_marks():
    """Build the aiming marks dict from the loaded TARGETS."""
    from core.config import TARGETS
    marks = {}
    for key, t in TARGETS.items():
        marks[key] = {
            "name":               t["name"],
            "reference_dist_m":   t.get("reference_dist_m", 10.0),
            "aiming_mark_dia_mm": t.get("aiming_mark_dia_mm", t["diameter_mm"] * 0.67),
            "outer_ring_dia_mm":  t["diameter_mm"],
            "rings_dia_mm":       t.get("rings_dia_mm", [d * 2 for d in t["rings_mm"]]),
            "ring_labels":        t.get("ring_labels",
                                        [str(int(s)) if s == int(s) else str(s)
                                         for s in t["ring_scores"]]),
            "mark_offsets":       t.get("mark_offsets"),   # None for single-mark
        }
    return marks

# Computed once at import — will include any user-added CSVs
AIMING_MARKS = _get_aiming_marks()


def mm(v: float) -> int:
    return int(v * MM_TO_PX)


def generate_marker_sheet(
    output_path: str = "aruco_sheet.png",
    target_key: str = "10m_air_rifle",
    print_distance_m: float = None,
    show_ring_guides: bool = True,
    aruco_dict_name: str = "DICT_4X4_50",
    marker_size_mm: float = None,         # None = use MARKER_MM default (40mm)
    margin_mm: float = None,              # None = use MARGIN_MM default (8mm)
    marker_count: int = 4,               # 4 / 6 / 8 markers on the sheet
) -> str:
    """
    Generate and save the ArUco marker sheet.

    The aiming mark is scaled proportionally:
        scale = print_distance_m / reference_distance_m

    So at half the reference distance, everything is half the size.
    """
    mark_cfg = AIMING_MARKS.get(target_key, AIMING_MARKS["10m_air_rifle"])
    ref_dist  = mark_cfg["reference_dist_m"]
    dist      = print_distance_m if print_distance_m else ref_dist
    scale_pct = dist / ref_dist          # e.g. 0.5 for 5m with 10m reference

    aiming_dia_scaled = mark_cfg["aiming_mark_dia_mm"] * scale_pct
    outer_dia_scaled  = mark_cfg["outer_ring_dia_mm"]  * scale_pct

    img = np.full((A4_H_PX, A4_W_PX), 255, dtype=np.uint8)

    dict_id    = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    _marker = marker_size_mm if marker_size_mm else MARKER_MM
    _margin = margin_mm if margin_mm else MARGIN_MM
    marker_px  = mm(_marker)
    margin_px  = mm(_margin)

    # ── Draw ArUco markers ────────────────────────────────────────────────────
    positions = {
        0: (margin_px,                        margin_px),
        1: (A4_W_PX - margin_px - marker_px,  margin_px),
        2: (A4_W_PX - margin_px - marker_px,  A4_H_PX - margin_px - marker_px),
        3: (margin_px,                        A4_H_PX - margin_px - marker_px),
    }
    # Extended positions for 6 and 8 marker layouts
    if marker_count >= 6:
        positions[4] = (margin_px, A4_H_PX // 2 - marker_px // 2)
        positions[5] = (A4_W_PX - margin_px - marker_px, A4_H_PX // 2 - marker_px // 2)
    if marker_count >= 8:
        positions[6] = (A4_W_PX // 2 - marker_px // 2, margin_px)
        positions[7] = (A4_W_PX // 2 - marker_px // 2, A4_H_PX - margin_px - marker_px)

    labels = {0: "TL(0)", 1: "TR(1)", 2: "BR(2)", 3: "BL(3)",
              4: "LM(4)", 5: "RM(5)", 6: "TM(6)", 7: "BM(7)"}
    for mid, (x, y) in positions.items():
        mimg = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)
        b = 4
        bordered = np.full((marker_px + 2*b, marker_px + 2*b), 255, dtype=np.uint8)
        bordered[b:b+marker_px, b:b+marker_px] = mimg
        mh, mw = bordered.shape
        img[y-b:y-b+mh, x-b:x-b+mw] = bordered
        lx = x if mid in (0, 3) else x - mm(8)
        ly = y + marker_px + mm(5) if mid in (0, 1) else y - mm(2)
        cv2.putText(img, labels[mid], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    # ── Draw aiming mark(s) ───────────────────────────────────────────────────
    cx, cy = A4_W_PX // 2, A4_H_PX // 2

    # For multi-mark targets, compute each mark's pixel position.
    # mark_offsets are in mm from sheet centre; scale_pct shrinks them for
    # closer distances just like the rings.
    mark_offsets = mark_cfg.get("mark_offsets")   # None for single-mark targets
    if mark_offsets:
        centres_px = [
            (cx + mm(mx * scale_pct), cy + mm(my * scale_pct))
            for mx, my in mark_offsets
        ]
    else:
        centres_px = [(cx, cy)]

    def _draw_one_mark(ocx, ocy, show_labels):
        """Draw rings, outer boundary, black bull, and cross at (ocx, ocy)."""
        if show_ring_guides:
            for i, ring_dia in enumerate(mark_cfg["rings_dia_mm"]):
                r_px = mm(ring_dia / 2 * scale_pct)
                if r_px < 4:
                    continue
                _draw_dashed_circle(img, ocx, ocy, r_px, shade=160, n_dashes=48)
                if show_labels:
                    lbl = mark_cfg["ring_labels"][i] if i < len(mark_cfg["ring_labels"]) else ""
                    if lbl:
                        cv2.putText(img, lbl, (ocx + r_px + mm(1), ocy + mm(1)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 1)
        outer_r_px = mm(outer_dia_scaled / 2)
        if outer_r_px > 4:
            cv2.circle(img, (ocx, ocy), outer_r_px, 100, 2)
        aim_r_px = mm(aiming_dia_scaled / 2)
        if aim_r_px > 2:
            cv2.circle(img, (ocx, ocy), aim_r_px, 0, -1)
        else:
            cv2.circle(img, (ocx, ocy), max(3, mm(0.5)), 0, -1)
        cross = max(mm(2), aim_r_px // 4)
        cv2.line(img, (ocx - cross, ocy), (ocx + cross, ocy), 255, 2)
        cv2.line(img, (ocx, ocy - cross), (ocx, ocy + cross), 255, 2)

    for i, (ocx, ocy) in enumerate(centres_px):
        _draw_one_mark(ocx, ocy, show_labels=(i == 0))

    # ── Print instructions ────────────────────────────────────────────────────
    instr_lines = [
        f"SPLATT2  —  {mark_cfg['name']}",
        f"Print distance: {dist:.1f}m  |  Scale: {scale_pct*100:.0f}%  |  Markers: {marker_count}  |  "
        f"Aiming mark: {aiming_dia_scaled:.1f}mm  |  Card: {outer_dia_scaled:.1f}mm  |  Markers: {_marker:.0f}mm",
        "Print at 100% on A4 — NO fit-to-page scaling.",
        "Verify printed aiming mark diameter with ruler before use.",
    ]
    for i, line in enumerate(instr_lines):
        cv2.putText(img, line, (mm(10), A4_H_PX - mm(36) + i * mm(9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, 60, 1)

    cv2.imwrite(output_path, img)
    print(f"[MarkerSheet] Saved: {output_path}  (scale={scale_pct*100:.0f}%,"
          f" aiming mark={aiming_dia_scaled:.1f}mm)")
    return output_path


def _draw_dashed_circle(img, cx, cy, r, shade=150, n_dashes=48):
    for i in range(n_dashes):
        if i % 2 == 0:
            a1 = 2 * np.pi * i / n_dashes
            a2 = 2 * np.pi * (i + 1) / n_dashes
            pts = [(int(cx + r * np.cos(a)), int(cy + r * np.sin(a)))
                   for a in np.linspace(a1, a2, 6)]
            for k in range(len(pts) - 1):
                cv2.line(img, pts[k], pts[k+1], shade, 2)


if __name__ == "__main__":
    generate_marker_sheet(print_distance_m=10.0)
    generate_marker_sheet("aruco_sheet_5m.png", print_distance_m=5.0)
