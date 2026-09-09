from __future__ import annotations

from fractions import Fraction

import numpy as np


def polyphase_resample(
    values: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float = 200.0,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Anti-aliased rational resampling; never use this to invent bandwidth."""
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise RuntimeError("scipy is required for polyphase resampling") from exc
    ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(10_000)
    return resample_poly(np.asarray(values), ratio.numerator, ratio.denominator, axis=axis)


def line_noise_ratio_db(emg: np.ndarray, sample_rate_hz: float = 200.0) -> float:
    """Compare the 50 Hz bin with neighboring 5 Hz bands."""
    signal = np.asarray(emg, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 8:
        raise ValueError("emg must have shape [samples,8]")
    if len(signal) < int(sample_rate_hz):
        return float("-inf")
    centered = signal - np.mean(signal, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate_hz)
    target = spectrum[(freqs >= 49.0) & (freqs <= 51.0)].mean()
    neighbor_mask = ((freqs >= 44.0) & (freqs <= 47.0)) | ((freqs >= 53.0) & (freqs <= 56.0))
    neighbor = spectrum[neighbor_mask].mean()
    return float(10.0 * np.log10((target + 1e-12) / (neighbor + 1e-12)))


class CausalEMGFilter:
    def __init__(
        self,
        sample_rate_hz: float = 200.0,
        *,
        low_hz: float = 20.0,
        high_hz: float = 90.0,
        notch_50hz: bool = False,
    ) -> None:
        try:
            from scipy.signal import butter, iirnotch, sosfilt, tf2sos
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scipy is required for causal EMG filtering") from exc
        if not 0 < low_hz < high_hz < sample_rate_hz / 2:
            raise ValueError("EMG band must lie inside Nyquist")
        sections = [butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")]
        if notch_50hz:
            b, a = iirnotch(50.0, 30.0, fs=sample_rate_hz)
            sections.append(tf2sos(b, a))
        self.sos = np.concatenate(sections, axis=0)
        self._sosfilt = sosfilt
        self.zi = np.zeros((len(self.sos), 2, 8), dtype=np.float64)

    def reset(self) -> None:
        self.zi.fill(0.0)

    def process(self, samples: np.ndarray) -> np.ndarray:
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 8:
            raise ValueError("EMG filter input must have shape [samples,8]")
        output, self.zi = self._sosfilt(self.sos, array, axis=0, zi=self.zi)
        return output


def _threshold_crossings(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    products = x[:-1] * x[1:]
    changes = np.abs(x[1:] - x[:-1])
    return np.sum((products < 0) & (changes >= threshold), axis=0)


def extract_emg_features(window: np.ndarray) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 8 or len(x) < 3:
        raise ValueError("EMG window must have shape [samples>=3,8]")
    rms = np.sqrt(np.mean(x * x, axis=0))
    mav = np.mean(np.abs(x), axis=0)
    wl = np.sum(np.abs(np.diff(x, axis=0)), axis=0) / (len(x) - 1)
    threshold = np.maximum(np.median(np.abs(x), axis=0) * 0.1, 1e-6)
    zc = _threshold_crossings(x, threshold) / (len(x) - 1)
    slopes = np.diff(x, axis=0)
    ssc = np.sum(
        ((slopes[:-1] * slopes[1:]) < 0)
        & (np.abs(slopes[1:] - slopes[:-1]) >= threshold),
        axis=0,
    ) / max(len(x) - 2, 1)
    energy = rms * rms
    relative = energy / max(float(energy.sum()), 1e-12)
    return np.concatenate([rms, mav, wl, zc, ssc, relative]).astype(np.float32)


def extract_imu_features(window: np.ndarray) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6 or len(x) < 2:
        raise ValueError("IMU window must have shape [samples>=2,6]")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    rms = np.sqrt(np.mean(x * x, axis=0))
    peak = np.max(np.abs(x), axis=0)
    t = np.arange(len(x), dtype=np.float64)
    t -= t.mean()
    slope = (t[:, None] * (x - mean)).sum(axis=0) / max(float((t * t).sum()), 1e-12)
    accel_norm = np.linalg.norm(x[:, :3], axis=1)
    gyro_norm = np.linalg.norm(x[:, 3:], axis=1)
    norms = np.array([
        accel_norm.mean(), accel_norm.std(), accel_norm.max(),
        gyro_norm.mean(), gyro_norm.std(), gyro_norm.max(),
    ])
    return np.concatenate([mean, std, rms, peak, slope, norms]).astype(np.float32)


def activation_intensity(normalized_emg: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.asarray(normalized_emg, dtype=np.float64) ** 2, axis=0))
    return float(np.clip(np.median(rms), 0.0, 1.0))


def motion_intensity(normalized_imu: np.ndarray) -> float:
    x = np.asarray(normalized_imu, dtype=np.float64)
    accel = np.sqrt(np.mean(np.sum(x[:, :3] ** 2, axis=1)))
    gyro = np.sqrt(np.mean(np.sum(x[:, 3:] ** 2, axis=1)))
    return float(np.clip(0.55 * accel + 0.45 * gyro, 0.0, 1.0))


def estimate_signal_quality(
    raw_emg: np.ndarray,
    raw_imu: np.ndarray,
    *,
    expected_samples: int,
) -> tuple[float, float]:
    emg = np.asarray(raw_emg, dtype=np.float64)
    imu = np.asarray(raw_imu, dtype=np.float64)
    emg_finite = np.isfinite(emg).mean(axis=0)
    imu_finite = np.isfinite(imu).mean(axis=0)
    emg_dynamic = np.std(np.nan_to_num(emg), axis=0) > 1e-10
    imu_dynamic_or_stable = np.all(np.isfinite(np.nan_to_num(imu)), axis=0)
    length_factor = min(len(emg), len(imu)) / max(expected_samples, 1)
    emg_score = float(np.clip(np.mean(emg_finite * emg_dynamic) * length_factor, 0.0, 1.0))
    imu_score = float(np.clip(np.mean(imu_finite * imu_dynamic_or_stable) * length_factor, 0.0, 1.0))
    return emg_score, imu_score
