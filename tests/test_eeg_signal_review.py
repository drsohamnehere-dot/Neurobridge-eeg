import numpy as np
import pytest

from server.eeg_signal_review import build_signal_review


def test_signal_review_returns_bounded_channel_features():
    sfreq = 256.0
    t = np.arange(int(8 * sfreq)) / sfreq
    alpha = 35e-6 * np.sin(2 * np.pi * 10 * t)
    theta = 20e-6 * np.sin(2 * np.pi * 6 * t)
    data = np.vstack([alpha, theta, alpha + theta])

    result = build_signal_review(
        data_volts=data,
        sfreq=sfreq,
        channel_names=["F3-C3", "C3-P3", "P3-O1"],
        recording_duration_seconds=60.0,
        window_start_seconds=0.0,
    )

    assert result["status"] == "research_screening_only"
    assert result["channels_processed"] == 3
    assert result["usable_channels"] == 3
    assert result["analysis_window_seconds"] == 8.0
    first = result["channels"][0]
    assert first["rms_uV"] > 0
    assert first["relative_band_power_pct"]["alpha_8_13_hz"] > 90
    assert "transient_candidate_count" in result
    assert "not diagnoses" in result["interpretation"]


def test_signal_review_rejects_invalid_sampling_rate():
    with pytest.raises(ValueError):
        build_signal_review(
            data_volts=np.ones((1, 100)),
            sfreq=0,
            channel_names=["F3"],
            recording_duration_seconds=1.0,
        )


def test_signal_review_rejects_non_2d_data():
    with pytest.raises(ValueError):
        build_signal_review(
            data_volts=np.ones(100),
            sfreq=100,
            channel_names=["F3"],
            recording_duration_seconds=1.0,
        )
