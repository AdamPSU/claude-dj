"""Vibe detection: head-pitch oscillation vs beat grid via DSP.

Computes a vibe_score (0..1) measuring rhythmic entrainment. Two sub-scores:

1. Phase-Locking Value (PLV): Hilbert transform of the bandpass-filtered
   head-pitch signal -> instantaneous phase -> sample at beat times ->
   PLV = |mean(exp(i * phase_at_beats))|. Consistent phase = real sync.

2. Period match: FFT dominant frequency of the pitch signal vs the beat
   frequency, allowing half-time (people bob on every other beat) and
   double-time. Scored by closeness to 1x/0.5x/2x beat freq.

CRITICAL INVARIANT: The input pitch series is RAW and UNSMOOTHED.
The vibe lives in the oscillation -- smoothing destroys it.
Responsiveness is tuned by the window length, not by smoothing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from vibe_dj import config


class VibeDetector:
    """Stateless vibe scorer. Call compute() each fusion tick."""

    def compute(
        self,
        pitch_window: list[tuple[float, float]],
        beat_times_in_window: list[float],
        bpm: float,
    ) -> tuple[float, float, float]:
        """Compute vibe from raw pitch samples and beat times.

        Args:
            pitch_window: [(timestamp, pitch_degrees), ...] -- RAW, not smoothed.
            beat_times_in_window: beat timestamps falling within the window.
            bpm: current beats per minute.

        Returns:
            (vibe_score, plv, period_match) -- all in [0, 1].
        """
        if len(pitch_window) < 10 or not beat_times_in_window or bpm <= 0:
            return 0.0, 0.0, 0.0

        timestamps = np.array([t for t, _ in pitch_window])
        pitches = np.array([p for _, p in pitch_window])

        # Compute effective sample rate from median interval
        diffs = np.diff(timestamps)
        dt = float(np.median(diffs))
        if dt <= 0:
            return 0.0, 0.0, 0.0
        fs = 1.0 / dt

        # Need at least one full cycle of the lowest bandpass frequency
        duration = timestamps[-1] - timestamps[0]
        if duration < 1.0 / config.BOB_FREQ_LO:
            return 0.0, 0.0, 0.0

        # Resample to uniform time grid (FFT and Hilbert need uniform spacing)
        n_samples = len(timestamps)
        t_uniform = np.linspace(timestamps[0], timestamps[-1], n_samples)
        pitch_uniform = np.interp(t_uniform, timestamps, pitches)

        # Remove DC offset
        pitch_uniform -= np.mean(pitch_uniform)

        # --- Bandpass filter (0.5-3 Hz) ---
        nyq = fs / 2.0
        if nyq <= config.BOB_FREQ_LO:
            return 0.0, 0.0, 0.0

        hi = min(config.BOB_FREQ_HI, nyq * 0.9)
        lo = config.BOB_FREQ_LO
        if lo >= hi:
            return 0.0, 0.0, 0.0

        try:
            b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
            filtered = filtfilt(b, a, pitch_uniform)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0, 0.0, 0.0

        # Amplitude gate: if RMS of filtered signal is tiny, the head
        # isn't really moving — no vibe regardless of phase statistics.
        rms = float(np.sqrt(np.mean(filtered ** 2)))
        if rms < 0.5:  # less than 0.5 degrees RMS = noise floor
            return 0.0, 0.0, 0.0

        # --- PLV (Phase-Locking Value) ---
        analytic = hilbert(filtered)
        inst_phase = np.angle(analytic)

        # Sample instantaneous phase at each beat time
        beat_arr = np.array(beat_times_in_window)
        beat_phases = np.interp(beat_arr, t_uniform, inst_phase)

        plv = float(abs(np.mean(np.exp(1j * beat_phases))))

        # --- Period match (FFT dominant freq vs beat freq) ---
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        fft_mag = np.abs(np.fft.rfft(filtered))

        # Mask to the bob frequency band
        mask = (freqs >= config.BOB_FREQ_LO) & (freqs <= config.BOB_FREQ_HI)
        if not np.any(mask):
            vibe = config.VIBE_PLV_WEIGHT * plv
            return max(0.0, min(1.0, vibe)), plv, 0.0

        fft_masked = fft_mag.copy()
        fft_masked[~mask] = 0
        dominant_freq = float(freqs[np.argmax(fft_masked)])

        beat_freq = bpm / 60.0

        # Check 1x, 0.5x, 2x beat frequency
        best_match = 0.0
        for candidate in [beat_freq, beat_freq / 2.0, beat_freq * 2.0]:
            if candidate <= 0:
                continue
            ratio = dominant_freq / candidate
            match = max(0.0, 1.0 - abs(ratio - 1.0) / config.PERIOD_TOLERANCE)
            best_match = max(best_match, match)

        period_match = min(1.0, best_match)

        # --- Combine ---
        vibe_score = config.VIBE_PLV_WEIGHT * plv + config.VIBE_PERIOD_WEIGHT * period_match
        vibe_score = max(0.0, min(1.0, vibe_score))

        return vibe_score, plv, period_match
