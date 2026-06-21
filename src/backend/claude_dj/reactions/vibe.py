from __future__ import annotations

import numpy as np

from . import config


class VibeDetector:
    def compute(
        self,
        pitch_window: list[tuple[float, float]],
        beat_times_in_window: list[float],
        bpm: float,
    ) -> tuple[float, float, float]:
        if len(pitch_window) < 10 or not beat_times_in_window or bpm <= 0:
            return 0.0, 0.0, 0.0

        from scipy.signal import butter, filtfilt, hilbert

        timestamps = np.array([timestamp for timestamp, _ in pitch_window])
        pitches = np.array([pitch for _, pitch in pitch_window])
        diffs = np.diff(timestamps)
        dt = float(np.median(diffs))
        if dt <= 0:
            return 0.0, 0.0, 0.0
        fs = 1.0 / dt

        duration = timestamps[-1] - timestamps[0]
        if duration < 1.0 / config.BOB_FREQ_LO:
            return 0.0, 0.0, 0.0

        sample_count = len(timestamps)
        uniform_timestamps = np.linspace(timestamps[0], timestamps[-1], sample_count)
        uniform_pitch = np.interp(uniform_timestamps, timestamps, pitches)
        uniform_pitch -= np.mean(uniform_pitch)

        nyquist = fs / 2.0
        if nyquist <= config.BOB_FREQ_LO:
            return 0.0, 0.0, 0.0
        high = min(config.BOB_FREQ_HI, nyquist * 0.9)
        low = config.BOB_FREQ_LO
        if low >= high:
            return 0.0, 0.0, 0.0

        try:
            b_coeff, a_coeff = butter(2, [low / nyquist, high / nyquist], btype="band")
            filtered = filtfilt(b_coeff, a_coeff, uniform_pitch)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0, 0.0, 0.0

        rms = float(np.sqrt(np.mean(filtered**2)))
        if rms < 0.5:
            return 0.0, 0.0, 0.0

        phase = np.angle(hilbert(filtered))
        beat_phases = np.interp(np.array(beat_times_in_window), uniform_timestamps, phase)
        plv = float(abs(np.mean(np.exp(1j * beat_phases))))

        freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
        magnitudes = np.abs(np.fft.rfft(filtered))
        mask = (freqs >= config.BOB_FREQ_LO) & (freqs <= config.BOB_FREQ_HI)
        if not np.any(mask):
            vibe = config.VIBE_PLV_WEIGHT * plv
            return max(0.0, min(1.0, vibe)), plv, 0.0

        masked_magnitudes = magnitudes.copy()
        masked_magnitudes[~mask] = 0
        dominant_freq = float(freqs[np.argmax(masked_magnitudes)])
        beat_freq = bpm / 60.0
        period_match = 0.0
        for candidate in (beat_freq, beat_freq / 2.0, beat_freq * 2.0):
            if candidate <= 0:
                continue
            ratio = dominant_freq / candidate
            match = max(0.0, 1.0 - abs(ratio - 1.0) / config.PERIOD_TOLERANCE)
            period_match = max(period_match, match)
        period_match = min(1.0, period_match)
        vibe_score = (config.VIBE_PLV_WEIGHT * plv) + (config.VIBE_PERIOD_WEIGHT * period_match)
        return max(0.0, min(1.0, vibe_score)), plv, period_match


def generated_beat_times(start: float, end: float, bpm: float) -> list[float]:
    if bpm <= 0 or end <= start:
        return []
    interval = 60.0 / bpm
    first = start - (start % interval)
    beats: list[float] = []
    current = first
    while current <= end:
        if current >= start:
            beats.append(current)
        current += interval
    return beats
