"""Bounded, non-diagnostic EEG signal feature screening.

This module reports descriptive signal features and review candidates only.
It does not diagnose epilepsy, identify seizures, or classify epileptiform discharges.
"""
from __future__ import annotations

from typing import Any

import numpy as np


BANDS = (
    ("delta_0_4_hz", 0.5, 4.0),
    ("theta_4_8_hz", 4.0, 8.0),
    ("alpha_8_13_hz", 8.0, 13.0),
    ("beta_13_30_hz", 13.0, 30.0),
    ("low_gamma_30_45_hz", 30.0, 45.0),
)


def _area(power: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float | None:
    mask = (freqs >= low) & (freqs < high)
    if int(mask.sum()) < 2:
        return None
    from scipy.integrate import trapezoid
    return float(trapezoid(power[mask], freqs[mask]))


def _channel_features(signal_volts: np.ndarray, sfreq: float) -> dict[str, Any]:
    from scipy.signal import butter, find_peaks, sosfiltfilt, welch

    x = np.asarray(signal_volts, dtype=np.float64) * 1_000_000.0  # MNE EEG data are in volts; report microvolts.
    x = x[np.isfinite(x)]
    if x.size < max(32, int(sfreq * 2)):
        return {"usable": False, "reason": "Fewer than two seconds of finite samples."}

    median = float(np.median(x))
    centered = x - median
    std = float(np.std(x))
    robust_sigma = float(1.4826 * np.median(np.abs(centered)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak_abs = float(np.max(np.abs(centered)))
    peak_to_peak = float(np.ptp(x))
    flat_fraction = float(np.mean(np.abs(np.diff(x)) < 0.01)) if x.size > 1 else 0.0
    high_amplitude_fraction = float(np.mean(np.abs(centered) > 500.0))

    nperseg = min(x.size, max(32, int(round(sfreq * 4))))
    freqs, psd = welch(centered, fs=sfreq, nperseg=nperseg, detrend="constant")
    band_absolute: dict[str, float | None] = {}
    for name, low, high in BANDS:
        band_absolute[name] = _area(psd, freqs, low, min(high, sfreq / 2.0))
    usable_power = [v for v in band_absolute.values() if v is not None and np.isfinite(v)]
    total_band_power = float(sum(usable_power)) if usable_power else 0.0
    band_relative = {
        name: (round(100.0 * value / total_band_power, 2) if value is not None and total_band_power > 0 else None)
        for name, value in band_absolute.items()
    }
    dominant_mask = (freqs >= 0.5) & (freqs <= min(30.0, sfreq / 2.0))
    dominant_frequency = None
    if int(dominant_mask.sum()) and np.any(psd[dominant_mask] > 0):
        dominant_frequency = float(freqs[dominant_mask][int(np.argmax(psd[dominant_mask]))])

    line_noise_ratio = None
    if sfreq / 2.0 > 52.0:
        line = _area(psd, freqs, 49.5, 50.5)
        broad = _area(psd, freqs, 1.0, min(45.0, sfreq / 2.0))
        if line is not None and broad is not None and broad > 0:
            line_noise_ratio = round(100.0 * line / broad, 2)

    flags: list[str] = []
    if std < 0.5 and flat_fraction > 0.95:
        flags.append("possible_flat_or_low_variance_channel")
    if high_amplitude_fraction > 0.01 or peak_abs > 1000.0:
        flags.append("large_amplitude_excursion_candidate")
    if flat_fraction > 0.98:
        flags.append("many_near_flat_samples")
    if line_noise_ratio is not None and line_noise_ratio > 25.0:
        flags.append("possible_50_hz_line_noise")
    
    # A generic sharp-transient candidate screen. These are not epileptiform-spike detections.
    transient_peaks: list[dict[str, float]] = []
    if sfreq >= 80.0 and x.size >= int(sfreq * 3.0):
        high_cut = min(35.0, sfreq * 0.40)
        if high_cut > 2.0:
            try:
                sos = butter(3, [1.0, high_cut], btype="bandpass", fs=sfreq, output="sos")
                filtered = sosfiltfilt(sos, centered)
                filtered_sigma = float(1.4826 * np.median(np.abs(filtered - np.median(filtered))))
                if filtered_sigma > 0:
                    threshold = max(6.0 * filtered_sigma, 30.0)
                    peaks, _ = find_peaks(
                        np.abs(filtered),
                        height=threshold,
                        distance=max(1, int(round(0.20 * sfreq))),
                        prominence=max(4.0 * filtered_sigma, 20.0),
                    )
                    for peak in peaks[:100]:
                        transient_peaks.append({
                            "time_seconds": round(float(peak / sfreq), 3),
                            "amplitude_uV": round(float(abs(filtered[peak])), 1),
                        })
            except Exception:
                # Preserve descriptive features if a short or unusual recording cannot be filtered.
                pass

    return {
        "usable": True,
        "samples_analysed": int(x.size),
        "mean_uV": round(float(np.mean(x)), 3),
        "std_uV": round(std, 3),
        "rms_uV": round(rms, 3),
        "peak_to_peak_uV": round(peak_to_peak, 1),
        "peak_absolute_deviation_uV": round(peak_abs, 1),
        "near_flat_sample_fraction_pct": round(100.0 * flat_fraction, 2),
        "samples_over_500uV_fraction_pct": round(100.0 * high_amplitude_fraction, 3),
        "absolute_band_power": {k: (round(v, 6) if v is not None else None) for k, v in band_absolute.items()},
        "relative_band_power_pct": band_relative,
        "dominant_frequency_hz": (round(dominant_frequency, 2) if dominant_frequency is not None else None),
        "line_noise_50hz_ratio_pct": line_noise_ratio,
        "quality_flags": flags,
        "transient_candidate_count": len(transient_peaks),
        "_transient_candidates": transient_peaks,
    }


def build_signal_review(
    data_volts: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    recording_duration_seconds: float,
    window_start_seconds: float = 0.0,
) -> dict[str, Any]:
    """Create compact per-channel features from (channels, samples) EEG data in volts."""
    data = np.asarray(data_volts)
    if data.ndim != 2:
        raise ValueError("EEG samples must be a 2D channels-by-samples array.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a positive finite number.")
    if data.shape[0] == 0 or data.shape[1] < 32:
        raise ValueError("No usable EEG channels or too few samples were supplied.")

    channel_results = []
    candidates = []
    total_candidates = 0
    for index in range(min(data.shape[0], len(channel_names))):
        name = str(channel_names[index])
        features = _channel_features(data[index], float(sfreq))
        local_candidates = features.pop("_transient_candidates", [])
        if not features.get("usable"):
            channel_results.append({"channel": name, **features})
            continue
        for candidate in local_candidates:
            candidates.append({
                "channel": name,
                "time_seconds": round(window_start_seconds + candidate["time_seconds"], 3),
                "amplitude_uV": candidate["amplitude_uV"],
                "type": "sharp_transient_candidate",
            })
        total_candidates += len(local_candidates)
        channel_results.append({"channel": name, **features})

    candidates.sort(key=lambda item: item["amplitude_uV"], reverse=True)
    flags_by_channel = [
        {"channel": item["channel"], "flags": item.get("quality_flags", [])}
        for item in channel_results if item.get("quality_flags")
    ]
    usable_count = sum(1 for item in channel_results if item.get("usable"))
    return {
        "status": "research_screening_only",
        "method": "Descriptive amplitude, spectral-band, signal-quality and generic transient-candidate screening",
        "recording_duration_seconds": round(float(recording_duration_seconds), 2),
        "analysis_window_start_seconds": round(float(window_start_seconds), 2),
        "analysis_window_seconds": round(float(data.shape[1] / sfreq), 2),
        "sampling_rate_hz": round(float(sfreq), 3),
        "channels_processed": len(channel_results),
        "usable_channels": usable_count,
        "channels_with_quality_flags": len(flags_by_channel),
        "transient_candidate_count": total_candidates,
        "channel_quality_flags": flags_by_channel,
        "channels": channel_results,
        "transient_candidates_top25": candidates[:25],
        "interpretation": (
            "These are descriptive signal features and heuristic review candidates, not diagnoses. "
            "A transient candidate is not an epileptiform discharge or seizure. Review the raw EEG, "
            "montage, artifacts, clinical context and expert-labelled reference annotations."
        ),
        "coverage_note": (
            "Only the initial bounded analysis window shown above was processed for this report; "
            "the remainder of a longer recording was not screened."
        ),
        "thresholds_note": (
            "Generic engineering thresholds are used for low variance, large amplitude and transient "
            "candidates. They are not clinically validated EEG abnormality thresholds."
        ),
    }
