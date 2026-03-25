"""
core/smoother.py
Real-time trace smoothing for aim-point data.

Two modes:
  EMA  — Exponential Moving Average. Zero dependencies. Very low lag.
          Good for general use. alpha controls responsiveness:
          low alpha (0.2) = very smooth but laggy
          high alpha (0.6) = less smooth but more responsive
          Recommended: 0.35

  SAVGOL — Savitzky-Golay filter over a rolling window. Requires scipy.
            Preserves the shape of genuine movement (hold, trigger pull)
            while removing high-frequency tremor and camera noise.
            window=11, poly=2 is a good starting point for 30fps.
            Falls back to EMA if scipy is not available.

Both operate on (x, y) float tuples and are designed to be called
once per camera frame with the latest zeroed aim position.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np


# ── EMA smoother ─────────────────────────────────────────────────────────────

class EMASmoother:
    """
    Single-pole IIR low-pass filter applied independently to x and y.

    s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    alpha=1.0 means no smoothing (pass-through).
    alpha=0.2 means heavy smoothing.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = max(0.05, min(1.0, alpha))
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def reset(self):
        self._sx = None
        self._sy = None

    def update(self, aim: Tuple[float, float]) -> Tuple[float, float]:
        x, y = aim
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            self._sx = self.alpha * x + (1.0 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1.0 - self.alpha) * self._sy
        return (self._sx, self._sy)


# ── Savitzky-Golay smoother ───────────────────────────────────────────────────

class SavGolSmoother:
    """
    Rolling Savitzky-Golay filter.

    Keeps a deque of the last `window` points and fits a polynomial of
    degree `poly` to them. Returns the smoothed value at the centre of
    the window — this introduces a lag of window//2 frames (~5–8 frames
    at 30fps with window=11, i.e. ~170ms).

    Falls back to EMA if scipy is unavailable.

    window must be odd and > poly.
    Recommended: window=11, poly=2 for 30fps; window=7, poly=2 for 60fps.
    """

    def __init__(self, window: int = 11, poly: int = 2,
                 fallback_alpha: float = 0.35):
        if window % 2 == 0:
            window += 1   # must be odd
        self.window = max(poly + 2, window)
        self.poly = poly
        self._buf_x: deque = deque(maxlen=self.window)
        self._buf_y: deque = deque(maxlen=self.window)
        self._fallback = EMASmoother(fallback_alpha)
        self._savgol_available = self._check_savgol()

    @staticmethod
    def _check_savgol() -> bool:
        try:
            from scipy.signal import savgol_filter  # noqa
            return True
        except ImportError:
            return False

    def reset(self):
        self._buf_x.clear()
        self._buf_y.clear()
        self._fallback.reset()

    def update(self, aim: Tuple[float, float]) -> Tuple[float, float]:
        x, y = aim
        self._buf_x.append(x)
        self._buf_y.append(y)

        if not self._savgol_available:
            return self._fallback.update(aim)

        n = len(self._buf_x)
        if n < self.poly + 2:
            # Not enough points yet — return EMA estimate
            return self._fallback.update(aim)

        from scipy.signal import savgol_filter
        # Use however many points we have, ensure window is odd
        w = n if n % 2 == 1 else n - 1
        w = max(self.poly + 2 if (self.poly + 2) % 2 == 1 else self.poly + 3, w)
        w = min(w, self.window if self.window % 2 == 1 else self.window - 1)

        xs = np.array(self._buf_x)
        ys = np.array(self._buf_y)
        try:
            sx = savgol_filter(xs, w, self.poly)
            sy = savgol_filter(ys, w, self.poly)
            # Return the most recent smoothed value
            return (float(sx[-1]), float(sy[-1]))
        except Exception:
            return self._fallback.update(aim)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_smoother(mode: str, **kwargs):
    """
    mode: 'none'   — pass-through, no smoothing
          'ema'    — Exponential Moving Average
          'savgol' — Savitzky-Golay (falls back to EMA if scipy absent)
    """
    if mode == 'ema':
        return EMASmoother(alpha=kwargs.get('alpha', 0.35))
    elif mode == 'savgol':
        return SavGolSmoother(
            window=kwargs.get('window', 11),
            poly=kwargs.get('poly', 2),
            fallback_alpha=kwargs.get('alpha', 0.35),
        )
    else:
        # 'none' — identity smoother
        class PassThrough:
            def update(self, aim): return aim
            def reset(self): pass
        return PassThrough()
