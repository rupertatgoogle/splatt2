"""
core/audio.py
Shot detection via transient (percussive click) detection.

For dry-firing, a click is a very short-duration, high-amplitude transient —
not a sustained loud noise. The detector works by:
  1. Tracking a rolling RMS baseline (ambient noise floor)
  2. Computing a short-window peak
  3. Firing when peak / baseline > transient_ratio AND peak > abs_threshold
  4. Enforcing a cooldown to prevent double-triggers

This approach rejects: sustained talking, chair noise, HVAC hum.
It accepts: a sharp click/snap that is N× louder than the room baseline.
"""

import threading
import time
import collections
import numpy as np
from typing import Callable, Optional, Deque

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

WAVEFORM_SAMPLES = 300   # history length for waveform display (frames)


class AudioDetector:
    """
    Parameters
    ----------
    threshold        : absolute peak threshold (0–1). Acts as a floor —
                       quiet rooms can use 0.05, louder rooms 0.15–0.3.
    transient_ratio  : peak must be this many times the rolling baseline RMS.
                       Typical: 4–8. Higher = only sharp spikes trigger.
    cooldown_ms      : min ms between triggers
    baseline_window  : number of chunks used for rolling baseline
    """

    def __init__(
        self,
        threshold: float = 0.15,
        transient_ratio: float = 6.0,
        cooldown_ms: int = 800,
        sample_rate: int = 44100,
        chunk_size: int = 2205,           # ~50ms chunks for responsiveness
        device_index: Optional[int] = None,
        on_shot: Optional[Callable] = None,
    ):
        self.threshold = threshold
        self.transient_ratio = transient_ratio
        self.cooldown_s = cooldown_ms / 1000.0
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.on_shot = on_shot

        self._stream = None
        self._last_trigger_time: float = 0.0
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Live display data (thread-safe via deque)
        self.current_level: float = 0.0       # latest RMS (for bar meter)
        self.current_peak: float = 0.0        # latest peak
        self.current_baseline: float = 0.0    # rolling baseline
        self.last_trigger_level: float = 0.0  # peak at last trigger
        self._waveform: Deque[float] = collections.deque(
            [0.0] * WAVEFORM_SAMPLES, maxlen=WAVEFORM_SAMPLES)

        # Rolling baseline: median of recent chunk RMSes
        self._baseline_buf: Deque[float] = collections.deque(
            [0.001] * 40, maxlen=40)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if not SD_AVAILABLE:
            print("[Audio] sounddevice not available.")
            return
        if self._running:
            return
        self._running = True
        self._paused = False
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[Audio] Stream error: {e}")
            self._running = False

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def pause(self, paused: bool):
        with self._lock:
            self._paused = paused

    def set_threshold(self, value: float):
        self.threshold = max(0.005, min(1.0, value))

    def set_transient_ratio(self, value: float):
        self.transient_ratio = max(1.5, min(20.0, value))

    def set_cooldown(self, ms: int):
        self.cooldown_s = ms / 1000.0

    def get_waveform(self) -> list:
        """Return recent normalised amplitude history for display."""
        return list(self._waveform)

    @staticmethod
    def list_devices():
        if not SD_AVAILABLE:
            return []
        devs = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    devs.append((i, d["name"]))
        except Exception:
            pass
        return devs

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if not any(indata):
            return

        audio = indata[:, 0].astype(np.float32)

        rms  = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        # Update waveform ring buffer (downsampled to one value per chunk)
        self._waveform.append(rms)

        # Update rolling baseline with slow-reacting median
        self._baseline_buf.append(rms)
        baseline = float(np.percentile(list(self._baseline_buf), 60))

        self.current_level    = rms
        self.current_peak     = peak
        self.current_baseline = baseline

        with self._lock:
            paused = self._paused
        if paused:
            return

        # Trigger condition:
        #   1. Absolute peak exceeds threshold floor
        #   2. Peak-to-baseline ratio exceeds transient_ratio (percussive test)
        ratio = peak / max(baseline, 1e-6)
        if peak >= self.threshold and ratio >= self.transient_ratio:
            now = time.time()
            if now - self._last_trigger_time >= self.cooldown_s:
                self._last_trigger_time = now
                self.last_trigger_level = peak
                if self.on_shot:
                    try:
                        self.on_shot(now)
                    except Exception as e:
                        print(f"[Audio] callback error: {e}")
