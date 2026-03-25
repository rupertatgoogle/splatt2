"""
core/tracker.py
ArUco marker tracking — computes where the camera is aimed relative to the
target centre, expressed in millimetres on the target face.

Layout expected on the printed A4 sheet:
    Marker IDs:
        0 ── top-left
        1 ── top-right
        2 ── bottom-right
        3 ── bottom-left

The four markers define the corners of a known rectangle (the "board").
We use a homography from detected marker corners → known board coordinates
to map the image centre (i.e. where the camera is pointing) into real-world
mm coordinates relative to the target centre.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TrackFrame:
    """Result of processing one video frame."""
    aim_mm: Optional[Tuple[float, float]] = None   # (x, y) mm from target centre
    aim_px: Optional[Tuple[int, int]] = None        # pixel location of aim on DISPLAY image
    markers_found: int = 0
    frame_display: Optional[np.ndarray] = None      # annotated frame for display
    homography: Optional[np.ndarray] = None
    quality: float = 0.0                            # 0-1 tracking quality


# ── Constants ─────────────────────────────────────────────────────────────────

MARKER_IDS = [0, 1, 2, 3]   # TL, TR, BR, BL


class ArucoTracker:
    """
    Tracks the aim point using four ArUco markers arranged around a target.

    Parameters
    ----------
    board_width_mm  : real-world width of the marker board (outer edges of markers), mm
    board_height_mm : real-world height of the marker board, mm
    marker_size_mm  : printed size of each individual marker, mm
    aruco_dict_name : cv2.aruco dictionary constant name string
    """

    def __init__(
        self,
        board_width_mm: float = 210.0,
        board_height_mm: float = 297.0,
        marker_size_mm: float = 40.0,
        aruco_dict_name: str = "DICT_4X4_50",
        margin_mm: float = 8.0,
        use_clahe: bool = True,
    ):
        self.board_width_mm = board_width_mm
        self.board_height_mm = board_height_mm
        self.marker_size_mm = marker_size_mm
        self.margin_mm = margin_mm

        # Build ArUco detector
        dict_id = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        # CLAHE — Contrast Limited Adaptive Histogram Equalisation.
        # Dramatically improves marker detection under uneven indoor lighting
        # with negligible performance cost (~2ms per frame at 480p).
        self.use_clahe = use_clahe
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Known corners of each marker in board-space (mm, origin = board TL)
        # Order within each marker: TL, TR, BR, BL
        m = marker_size_mm
        mg = margin_mm
        bw = board_width_mm
        bh = board_height_mm

        self._board_corners = {
            0: np.array([[mg,    mg   ], [mg+m,  mg   ], [mg+m,  mg+m ], [mg,    mg+m ]], dtype=np.float32),
            1: np.array([[bw-mg-m, mg ], [bw-mg,  mg  ], [bw-mg, mg+m ], [bw-mg-m, mg+m]], dtype=np.float32),
            2: np.array([[bw-mg-m, bh-mg-m], [bw-mg, bh-mg-m], [bw-mg, bh-mg], [bw-mg-m, bh-mg]], dtype=np.float32),
            3: np.array([[mg, bh-mg-m], [mg+m, bh-mg-m], [mg+m, bh-mg], [mg, bh-mg]], dtype=np.float32),
        }

        # Target centre in board-space (mm) — exactly the centre of the board
        self.target_centre_mm = np.array([bw / 2, bh / 2], dtype=np.float32)

        self._last_homography: Optional[np.ndarray] = None
        self._homography_age: int = 0
        self.MAX_HOMOGRAPHY_AGE = 5   # frames we'll reuse a stale homography

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> TrackFrame:
        result = TrackFrame()
        result.frame_display = frame.copy()

        # Pre-process: convert to greyscale and optionally apply CLAHE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            gray = self._clahe.apply(gray)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            result.markers_found = 0
            self._homography_age += 1
            self._try_reuse_homography(result, frame)
            return result

        # Flatten id array
        ids_flat = ids.flatten()
        result.markers_found = len(ids_flat)

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(result.frame_display, corners, ids)

        # Build point correspondences: image pixel → board mm
        img_pts: List[np.ndarray] = []
        brd_pts: List[np.ndarray] = []

        for i, mid in enumerate(ids_flat):
            if mid in self._board_corners:
                img_pts.append(corners[i][0])          # shape (4,2)
                brd_pts.append(self._board_corners[mid])

        if len(img_pts) < 1:
            self._homography_age += 1
            self._try_reuse_homography(result, frame)
            return result

        img_pts_all = np.concatenate(img_pts, axis=0)   # (N*4, 2)
        brd_pts_all = np.concatenate(brd_pts, axis=0)

        H, mask = cv2.findHomography(img_pts_all, brd_pts_all, cv2.RANSAC, 5.0)

        if H is None:
            self._homography_age += 1
            self._try_reuse_homography(result, frame)
            return result

        self._last_homography = H
        self._homography_age = 0
        result.homography = H
        result.quality = min(1.0, len(img_pts) / 4.0)

        self._compute_aim(result, frame)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _try_reuse_homography(self, result: TrackFrame, frame: np.ndarray):
        if self._last_homography is not None and self._homography_age <= self.MAX_HOMOGRAPHY_AGE:
            result.homography = self._last_homography
            result.quality = max(0.1, 0.5 - self._homography_age * 0.1)
            self._compute_aim(result, frame)

    def _compute_aim(self, result: TrackFrame, frame: np.ndarray):
        H = result.homography
        if H is None:
            return

        h, w = frame.shape[:2]
        img_centre = np.array([[[w / 2, h / 2]]], dtype=np.float32)

        # Map image centre → board coordinates (mm)
        board_pt = cv2.perspectiveTransform(img_centre, H)[0][0]

        # Convert to mm offset from target centre
        aim_mm = (
            float(board_pt[0] - self.target_centre_mm[0]),
            float(board_pt[1] - self.target_centre_mm[1]),
        )
        result.aim_mm = aim_mm

        # Also store pixel coordinates of aim (image centre)
        result.aim_px = (int(w / 2), int(h / 2))

        # Draw crosshair on display frame
        cx, cy = int(w / 2), int(h / 2)
        color = (0, 255, 0) if result.quality > 0.5 else (0, 165, 255)
        cv2.line(result.frame_display, (cx - 20, cy), (cx + 20, cy), color, 2)
        cv2.line(result.frame_display, (cx, cy - 20), (cx, cy + 20), color, 2)
        cv2.circle(result.frame_display, (cx, cy), 8, color, 1)

        # Annotate aim coords
        txt = f"Aim: ({aim_mm[0]:+.1f}, {aim_mm[1]:+.1f}) mm"
        cv2.putText(result.frame_display, txt, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_shot(
    aim_mm: tuple,
    scoring_radius_mm: float,
    decimal: bool = False,
    mark_offsets: list = None,
) -> tuple:
    """
    Score a shot purely from geometry.

    aim_mm            : (x, y) in mm from sheet centre.
    scoring_radius_mm : R = card_radius + calibre_radius.
    decimal           : 99 bands (10.9->1.0) if True; 10 bands (10->1) if False.
    mark_offsets      : list of (x_mm, y_mm) mark centres for multi-mark targets.
                        If None, scores relative to (0,0) — single-mark behaviour
                        unchanged. If multiple marks, assigns shot to nearest mark
                        (Option C) and scores relative to that mark centre.

    Returns (score, band_index, mark_index).
    mark_index is always 0 for single-mark targets.
    score=0.0, band=-1, mark_index=0 for a complete miss.
    """
    import math as _math

    # Multi-mark: find nearest mark centre
    if mark_offsets and len(mark_offsets) > 1:
        best_dist  = float("inf")
        best_mark  = 0
        best_local = aim_mm
        for idx, (mx, my) in enumerate(mark_offsets):
            dx = aim_mm[0] - mx
            dy = aim_mm[1] - my
            d  = _math.sqrt(dx*dx + dy*dy)
            if d < best_dist:
                best_dist  = d
                best_mark  = idx
                best_local = (dx, dy)
        aim_local = best_local
        mark_idx  = best_mark
    else:
        aim_local = aim_mm
        mark_idx  = 0

    aim_r = _math.sqrt(aim_local[0]**2 + aim_local[1]**2)

    if aim_r > scoring_radius_mm:
        return 0.0, -1, mark_idx

    if decimal:
        n_bands = 99
        step    = 9.9 / 98
        band_w  = scoring_radius_mm / n_bands
        band_n  = min(int(aim_r / band_w), n_bands - 1)
        score   = round(10.9 - band_n * step, 1)
        return score, band_n, mark_idx
    else:
        n_bands = 10
        band_w  = scoring_radius_mm / n_bands
        band_n  = min(int(aim_r / band_w), n_bands - 1)
        score   = float(10 - band_n)
        return score, band_n, mark_idx

def aim_to_display(aim_mm, target_cfg, display_size_px):
    """
    Convert aim_mm (x,y) offset from target centre into pixel coordinates
    on the target display canvas.

    display_size_px: (width, height) of the target display area
    """
    dw, dh = display_size_px
    scale = min(dw, dh) / target_cfg["diameter_mm"]  # px per mm
    cx, cy = dw / 2, dh / 2
    px = int(cx + aim_mm[0] * scale)
    py = int(cy + aim_mm[1] * scale)
    return px, py
