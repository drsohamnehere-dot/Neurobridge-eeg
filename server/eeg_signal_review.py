"""Bounded, non-diagnostic EEG signal feature screening.

This module reports descriptive signal features and review candidates only.
It does not diagnose epilepsy, identify seizures, or classify epileptiform discharges.
"""
from __future__ import annotations

from typing import Any
import re

import numpy as np


BANDS = (
    ("delta_0_4_hz", 0.5, 4.0),
    ("theta_4_8_hz", 4.0, 8.0),
    ("alpha_8_13_hz", 8.0, 13.0),
    ("beta_13_30_hz", 13.0, 30.0),
    ("low_gamma_30_45_hz", 30.0, 45.0),
)

# Supported processing / display capacities. A file may contain fewer actual scalp EEG channels.
SUPPORTED_CHANNEL_LIMITS = (24, 32, 64)


def apply_channel_limit(channel_indices: list[int], channel_limit: int) -> list[int]:
    """Return up to 24, 32 or 64 selected scalp-channel indices without inventing channels."""
    if channel_limit not in SUPPORTED_CHANNEL_LIMITS:
        raise ValueError(f"Channel limit must be one of {SUPPORTED_CHANNEL_LIMITS}.")
    return list(channel_indices[:channel_limit])


SCALP_ELECTRODE_LABELS = {
    "FP1","FP2","FPZ","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "AF7","AF3","AFZ","AF4","AF8","FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T3","T7","C5","C3","C1","CZ","C2","C4","C6","T4","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "T5","P7","P5","P3","P1","PZ","P2","P4","P6","P8","T6","PO7","PO5","PO3","POZ","PO4","PO6","PO8","O1","OZ","O2",
}

def scalp_eeg_channel_indices(channel_names: list[str]) -> list[int]:
    """Identify standard scalp electrodes; exclude auxiliary ECG/BP/DC/PG leads."""
    result = []
    for index, name in enumerate(channel_names):
        text = re.sub(r"\\b(?:EEG|REF|LE|RE|AVG|AV)\\b", " ", str(name).upper())
        tokens = [token for token in re.split(r"[^A-Z0-9]+", text) if token]
        if any(token in SCALP_ELECTRODE_LABELS for token in tokens):
            result.append(index)
    return result


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


# Heuristic artefact screening for selecting candidate low-artifact epochs.
# These thresholds are engineering review prompts, not validated clinical cut-offs.
ARTIFACT_HARD_FLAGS = {
    "possible_flat_or_disconnected_channel",
    "possible_large_amplitude_artifact",
    "possible_non_finite_samples",
}
ARTIFACT_SOFT_FLAGS = {
    "possible_abrupt_step_or_electrode_pop",
    "possible_high_frequency_muscle_emg",
    "possible_50_hz_line_noise",
    "possible_sweat_or_baseline_drift",
    "possible_frontal_eye_movement_or_slow_activity",
}


def assess_eeg_epoch_artifacts(
    data_volts: np.ndarray,
    sfreq: float,
    channel_names: list[str],
) -> dict[str, Any]:
    """Score one short EEG epoch for engineering artefact-review candidates.

    Inputs are volts in channels x samples. The score is not a probability and
    "low_artifact_candidate" does not mean proven artifact-free.
    """
    data = np.asarray(data_volts, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("Epoch data must be a non-empty channels-by-samples array.")
    if not channel_names:
        raise ValueError("At least one scalp EEG channel name is required.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be positive and finite.")
    if data.shape[1] < max(32, int(round(sfreq * 1.5))):
        raise ValueError("Epoch is too short; use at least 1.5 seconds.")

    channel_results: list[dict[str, Any]] = []
    band_rows: list[dict[str, float]] = []
    hard_channels = 0
    soft_channels = 0
    for index in range(min(data.shape[0], len(channel_names))):
        name = str(channel_names[index])
        raw = data[index]
        valid = np.isfinite(raw)
        flags: list[str] = []
        if float(np.mean(~valid)) > 0.005:
            flags.append("possible_non_finite_samples")
        if int(valid.sum()) < max(32, int(round(sfreq * 1.5))):
            channel_results.append({
                "channel": name, "usable": False,
                "flags": flags or ["insufficient_finite_samples"],
            })
            hard_channels += 1
            continue

        x = raw[valid] * 1_000_000.0
        centered = x - float(np.median(x))
        ac_rms = float(np.sqrt(np.mean(np.square(centered))))
        peak_abs = float(np.max(np.abs(centered)))
        peak_to_peak = float(np.ptp(x))
        differences = np.abs(np.diff(x))
        flat_fraction = float(np.mean(differences < 0.01)) if differences.size else 0.0

        if ac_rms < 0.5 and flat_fraction > 0.95:
            flags.append("possible_flat_or_disconnected_channel")
        if peak_to_peak > 500.0 or peak_abs > 500.0:
            flags.append("possible_large_amplitude_artifact")
        if differences.size:
            robust_diff = float(1.4826 * np.median(np.abs(differences - np.median(differences))))
            step_limit = max(100.0, 25.0 * robust_diff)
            if float(np.max(differences)) > step_limit:
                flags.append("possible_abrupt_step_or_electrode_pop")

        window = np.hanning(x.size)
        spectrum = np.square(np.abs(np.fft.rfft(centered * window)))
        frequencies = np.fft.rfftfreq(x.size, d=1.0 / float(sfreq))

        def band_sum(low: float, high: float) -> float:
            mask = (frequencies >= low) & (frequencies < high)
            return float(np.sum(spectrum[mask])) if np.any(mask) else 0.0

        broad = band_sum(0.5, min(45.0, sfreq / 2.0))
        delta = band_sum(0.5, min(4.0, sfreq / 2.0))
        theta = band_sum(4.0, min(8.0, sfreq / 2.0))
        alpha = band_sum(8.0, min(13.0, sfreq / 2.0))
        beta = band_sum(13.0, min(30.0, sfreq / 2.0))
        gamma = band_sum(30.0, min(45.0, sfreq / 2.0))
        denom = max(broad, 1e-20)
        band_percent = {
            "delta": round(100.0 * delta / denom, 2),
            "theta": round(100.0 * theta / denom, 2),
            "alpha": round(100.0 * alpha / denom, 2),
            "beta": round(100.0 * beta / denom, 2),
            "low_gamma": round(100.0 * gamma / denom, 2),
        }

        muscle_ratio = band_sum(20.0, min(45.0, sfreq / 2.0)) / denom
        if muscle_ratio > 0.45:
            flags.append("possible_high_frequency_muscle_emg")
        if sfreq / 2.0 >= 52.0:
            mains_ratio = band_sum(49.0, 51.0) / denom
            if mains_ratio > 0.20:
                flags.append("possible_50_hz_line_noise")
        drift_ratio = band_sum(0.1, min(0.5, sfreq / 2.0)) / max(band_sum(0.1, min(45.0, sfreq / 2.0)), 1e-20)
        if drift_ratio > 0.70:
            flags.append("possible_sweat_or_baseline_drift")
        if name.upper().replace("EEG ", "").startswith(("FP1", "FP2", "FPZ")):
            if band_percent["delta"] > 70.0 and peak_to_peak > 100.0:
                flags.append("possible_frontal_eye_movement_or_slow_activity")

        hard = any(flag in ARTIFACT_HARD_FLAGS for flag in flags)
        soft = any(flag in ARTIFACT_SOFT_FLAGS for flag in flags)
        hard_channels += int(hard)
        soft_channels += int(soft)
        band_rows.append(band_percent)
        channel_results.append({
            "channel": name,
            "usable": True,
            "ac_rms_uv": round(ac_rms, 3),
            "peak_to_peak_uv": round(peak_to_peak, 3),
            "peak_absolute_deviation_uv": round(peak_abs, 3),
            "flat_sample_fraction_pct": round(100.0 * flat_fraction, 2),
            "band_power_pct": band_percent,
            "flags": flags,
        })

    channel_count = max(1, len(channel_results))
    score = max(0.0, 100.0 - (55.0 * hard_channels / channel_count) - (20.0 * soft_channels / channel_count))
    status = "low_artifact_candidate" if hard_channels == 0 and soft_channels == 0 and all(x.get("usable") for x in channel_results) else "review_required"
    band_medians: dict[str, float] = {}
    if band_rows:
        for band in ("delta", "theta", "alpha", "beta", "low_gamma"):
            band_medians[band] = round(float(np.median([row[band] for row in band_rows])), 2)
    return {
        "status": status,
        "quality_score": round(score, 1),
        "score_note": "Engineering quality score, not a probability or clinical grade.",
        "channels_assessed": len(channel_results),
        "channels_with_hard_flags": hard_channels,
        "channels_with_review_flags": soft_channels,
        "channels": channel_results,
        "median_channel_band_power_pct": band_medians,
    }


def merge_candidate_epoch_intervals(epochs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent low-artifact candidate epochs into non-overlapping intervals."""
    accepted = sorted(
        (item for item in epochs if item.get("status") == "low_artifact_candidate"),
        key=lambda item: float(item["start_seconds"]),
    )
    merged: list[dict[str, Any]] = []
    for item in accepted:
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
        if merged and start <= float(merged[-1]["end_seconds"]) + 0.001:
            merged[-1]["end_seconds"] = round(max(float(merged[-1]["end_seconds"]), end), 3)
            merged[-1]["epoch_count"] += 1
            merged[-1]["duration_seconds"] = round(merged[-1]["end_seconds"] - merged[-1]["start_seconds"], 3)
        else:
            merged.append({
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "epoch_count": 1,
            })
    return merged


DEFAULT_SPECTRAL_RATIO_SETTINGS = {
    "spectrum_min_hz": 0.5,
    "spectrum_max_hz": 30.0,
    "delta_low_hz": 0.5,
    "delta_high_hz": 4.0,
    "theta_low_hz": 4.0,
    "theta_high_hz": 8.0,
    "alpha_low_hz": 8.0,
    "alpha_high_hz": 13.0,
    "beta_lf_low_hz": 13.0,
    "beta_lf_high_hz": 20.0,
    "beta_hf_low_hz": 20.0,
    "beta_hf_high_hz": 30.0,
    "window_function": "none",
}


def calculate_fft_rhythm_indices(
    data_volts: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate FFT-bin rhythm indices and Neuro-Spectrum-style ratios.

    Indices are each band's summed FFT power divided by total FFT power within
    the configured spectrum range. Ratios are computed from the band indices.
    This is a comparison-oriented calculation, not a clinical EEG classifier.
    """
    config = {**DEFAULT_SPECTRAL_RATIO_SETTINGS, **(settings or {})}
    required = [
        "spectrum_min_hz", "spectrum_max_hz",
        "delta_low_hz", "delta_high_hz",
        "theta_low_hz", "theta_high_hz",
        "alpha_low_hz", "alpha_high_hz",
        "beta_lf_low_hz", "beta_lf_high_hz",
        "beta_hf_low_hz", "beta_hf_high_hz",
    ]
    try:
        bounds = {key: float(config[key]) for key in required}
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Spectral ratio settings must contain valid numeric frequency boundaries.") from exc
    if not all(np.isfinite(value) for value in bounds.values()):
        raise ValueError("Spectral ratio boundaries must be finite numbers.")
    if bounds["spectrum_min_hz"] < 0 or bounds["spectrum_max_hz"] <= bounds["spectrum_min_hz"]:
        raise ValueError("Total spectrum range must have a positive lower-to-upper frequency range.")

    ordered = [
        ("delta", bounds["delta_low_hz"], bounds["delta_high_hz"]),
        ("theta", bounds["theta_low_hz"], bounds["theta_high_hz"]),
        ("alpha", bounds["alpha_low_hz"], bounds["alpha_high_hz"]),
        ("beta_lf", bounds["beta_lf_low_hz"], bounds["beta_lf_high_hz"]),
        ("beta_hf", bounds["beta_hf_low_hz"], bounds["beta_hf_high_hz"]),
    ]
    previous_high = bounds["spectrum_min_hz"]
    for name, low, high in ordered:
        if low < bounds["spectrum_min_hz"] or high > bounds["spectrum_max_hz"] or high <= low:
            raise ValueError(f"{name} band must lie within the total spectrum range and have positive width.")
        if low < previous_high - 1e-9:
            raise ValueError("Rhythm bands must be ordered and must not overlap.")
        previous_high = high
    if previous_high > bounds["spectrum_max_hz"] + 1e-9:
        raise ValueError("Rhythm bands exceed the total spectrum range.")

    window_name = str(config.get("window_function", "none")).lower()
    if window_name not in {"none", "hann"}:
        raise ValueError("Window function must be 'none' or 'hann'.")

    data = np.asarray(data_volts, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("EEG data must be a non-empty channels-by-samples array.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be positive and finite.")
    nyquist = float(sfreq) / 2.0
    effective_max = min(bounds["spectrum_max_hz"], nyquist)
    effective_min = bounds["spectrum_min_hz"]
    if effective_max <= effective_min:
        raise ValueError("The recording Nyquist frequency is below the requested total spectrum minimum.")

    rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}
    band_names = [name for name, _, _ in ordered]
    for index in range(min(data.shape[0], len(channel_names))):
        channel = str(channel_names[index])
        x = data[index]
        x = x[np.isfinite(x)]
        if x.size < max(32, int(round(sfreq * 1.5))):
            rows.append({
                "channel": channel, "status": "not_available",
                "reason": "Too few finite samples for FFT rhythm indices.",
            })
            continue

        centered = x - float(np.mean(x))
        window = np.ones(centered.size, dtype=np.float64) if window_name == "none" else np.hanning(centered.size)
        power = np.square(np.abs(np.fft.rfft(centered * window)))
        frequencies = np.fft.rfftfreq(centered.size, d=1.0 / float(sfreq))

        total_mask = (frequencies >= effective_min) & (frequencies <= effective_max)
        total_power = float(np.sum(power[total_mask]))
        raw_bands: dict[str, float | None] = {}
        indices: dict[str, float | None] = {}
        for name, low, high in ordered:
            actual_low = max(low, effective_min)
            actual_high = min(high, effective_max)
            if actual_high <= actual_low or (name == "beta_hf" and actual_high <= 20.0 and effective_max <= 20.0):
                raw_bands[name] = None
                indices[name] = None
                continue
            mask = (frequencies >= actual_low) & (frequencies < actual_high)
            band_power = float(np.sum(power[mask])) if np.any(mask) else 0.0
            raw_bands[name] = band_power
            indices[name] = round(100.0 * band_power / total_power, 4) if total_power > 0 else None

        beta_total = None
        if raw_bands.get("beta_lf") is not None and raw_bands.get("beta_hf") is not None and total_power > 0:
            beta_total = raw_bands["beta_lf"] + raw_bands["beta_hf"]

        theta_power = raw_bands.get("theta")
        alpha_power = raw_bands.get("alpha")
        beta_lf_power = raw_bands.get("beta_lf")
        ratios = {
            "alpha_theta": round(alpha_power / theta_power, 4) if alpha_power is not None and theta_power is not None and theta_power > 0 else None,
            "theta_beta": round(theta_power / beta_total, 4) if theta_power is not None and beta_total is not None and beta_total > 0 else None,
            "theta_beta_lf": round(theta_power / beta_lf_power, 4) if theta_power is not None and beta_lf_power is not None and beta_lf_power > 0 else None,
        }
        row = {
            "channel": channel,
            "status": "ok" if total_power > 0 else "not_available",
            "band_power_index_pct": indices,
            "ratios": ratios,
            "spectrum_range_hz": [round(effective_min, 3), round(effective_max, 3)],
        }
        rows.append(row)
        aggregate[channel] = raw_bands if total_power > 0 else {}

    return {
        "method": "FFT bin-power sum; each rhythm index is its band power divided by total power in the configured spectrum range",
        "window_function": window_name,
        "settings": {key: config[key] for key in DEFAULT_SPECTRAL_RATIO_SETTINGS},
        "effective_spectrum_range_hz": [round(effective_min, 3), round(effective_max, 3)],
        "nyquist_hz": round(nyquist, 3),
        "channels": rows,
        "formulae": {
            "alpha_theta": "Alpha index / Theta index",
            "theta_beta": "Theta index / (Beta LF index + Beta HF index)",
            "theta_beta_lf": "Theta index / Beta LF index",
        },
        "note": (
            "Comparison-oriented FFT indices. Neuro-Spectrum also allows customised rhythm boundaries and window settings; "
            "set the same boundaries, total frequency range, selected segment and window in both systems before comparing. "
            "These ratios are descriptive qEEG features, not stand-alone diagnoses."
        ),
    }


def aggregate_fft_band_power(
    target: dict[str, dict[str, float]],
    fft_result: dict[str, Any],
) -> None:
    """Add absolute per-channel band power sums from one same-length FFT epoch."""
    for row in fft_result.get("channels", []):
        if row.get("status") != "ok":
            continue
        bands = row.get("_band_power_raw") or row.get("band_power_raw")
        if not bands:
            continue
        current = target.setdefault(row["channel"], {})
        for name, value in bands.items():
            if value is not None:
                current[name] = current.get(name, 0.0) + float(value)


def rhythm_ratio_rows_from_power_sums(
    power_sums: dict[str, dict[str, float]],
    settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create relative rhythm indices and ratios from accumulated FFT band sums."""
    rows = []
    for channel, powers in power_sums.items():
        total = sum(value for value in powers.values() if value is not None)
        delta = powers.get("delta")
        theta = powers.get("theta")
        alpha = powers.get("alpha")
        beta_lf = powers.get("beta_lf")
        beta_hf = powers.get("beta_hf")
        beta_total = (beta_lf + beta_hf) if beta_lf is not None and beta_hf is not None else None
        idx = {
            name: (round(100.0 * value / total, 4) if value is not None and total > 0 else None)
            for name, value in powers.items()
        }
        rows.append({
            "channel": channel,
            "band_power_index_pct": idx,
            "ratios": {
                "alpha_theta": round(alpha / theta, 4) if alpha is not None and theta is not None and theta > 0 else None,
                "theta_beta": round(theta / beta_total, 4) if theta is not None and beta_total is not None and beta_total > 0 else None,
                "theta_beta_lf": round(theta / beta_lf, 4) if theta is not None and beta_lf is not None and beta_lf > 0 else None,
            },
        })
    return rows
