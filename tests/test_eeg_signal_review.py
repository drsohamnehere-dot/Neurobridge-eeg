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


def test_scalp_channel_selection_excludes_auxiliary_channels():
    from server.eeg_signal_review import scalp_eeg_channel_indices

    labels = [
        "PG1", "FP1", "FP2", "PG2", "F7", "F3", "FZ", "F4", "F8", "A1",
        "T3", "C3", "CZ", "C4", "T4", "A2", "T5", "P3", "PZ", "P4",
        "T6", "O1", "OZ", "O2", "X1", "X2", "X3", "X4", "BP1", "BP2",
        "ECG", "BP4", "BP5", "BP6", "BP7", "BP8", "DC1", "DC2", "DC3", "DC4",
    ]
    indices = scalp_eeg_channel_indices(labels)
    assert len(indices) == 20
    assert [labels[i] for i in indices] == [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
        "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
    ]


def test_scalp_channel_selection_supports_reference_and_bipolar_labels():
    from server.eeg_signal_review import scalp_eeg_channel_indices

    labels = ["EEG Fp1-Ref", "F3-C3", "ECG", "DC1", "A1"]
    assert scalp_eeg_channel_indices(labels) == [0, 1]
