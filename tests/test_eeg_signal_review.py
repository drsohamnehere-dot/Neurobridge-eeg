import numpy as np
import pytest

from server.eeg_signal_review import apply_channel_limit, assess_eeg_epoch_artifacts, build_signal_review, merge_candidate_epoch_intervals


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



@pytest.mark.parametrize("capacity", [24, 32, 64])
def test_channel_capacity_limits_select_available_channels_only(capacity):
    available = list(range(80))
    selected = apply_channel_limit(available, capacity)
    assert len(selected) == capacity
    assert selected == available[:capacity]


def test_channel_capacity_does_not_invent_missing_channels():
    available = list(range(20))
    assert apply_channel_limit(available, 24) == available
    assert apply_channel_limit(available, 32) == available
    assert apply_channel_limit(available, 64) == available


def test_channel_capacity_rejects_unsupported_limit():
    with pytest.raises(ValueError, match="Channel limit"):
        apply_channel_limit(list(range(40)), 48)



def _test_sine(amplitude_uv, frequency_hz, sfreq=256.0, seconds=4.0):
    time = np.arange(int(sfreq * seconds)) / sfreq
    return (amplitude_uv * 1e-6 * np.sin(2 * np.pi * frequency_hz * time))[None, :]


def test_low_artifact_epoch_is_only_a_candidate_not_a_diagnosis():
    data = _test_sine(30.0, 10.0)
    result = assess_eeg_epoch_artifacts(data, 256.0, ["C3"])
    assert result["status"] == "low_artifact_candidate"
    assert result["quality_score"] == 100.0
    assert result["channels_assessed"] == 1


def test_flat_epoch_is_flagged_for_review():
    result = assess_eeg_epoch_artifacts(np.zeros((1, 1024)), 256.0, ["C3"])
    assert result["status"] == "review_required"
    assert "possible_flat_or_disconnected_channel" in result["channels"][0]["flags"]


def test_large_amplitude_epoch_is_flagged_for_review():
    data = _test_sine(400.0, 10.0)
    result = assess_eeg_epoch_artifacts(data, 256.0, ["C3"])
    assert result["status"] == "review_required"
    assert "possible_large_amplitude_artifact" in result["channels"][0]["flags"]


def test_50_hz_interference_is_flagged_as_a_candidate():
    time = np.arange(1024) / 256.0
    data = (25e-6 * np.sin(2 * np.pi * 10 * time) + 30e-6 * np.sin(2 * np.pi * 50 * time))[None, :]
    result = assess_eeg_epoch_artifacts(data, 256.0, ["C3"])
    assert "possible_50_hz_line_noise" in result["channels"][0]["flags"]


def test_high_frequency_activity_is_flagged_for_manual_muscle_review():
    time = np.arange(1024) / 256.0
    data = (10e-6 * np.sin(2 * np.pi * 10 * time) + 35e-6 * np.sin(2 * np.pi * 35 * time))[None, :]
    result = assess_eeg_epoch_artifacts(data, 256.0, ["T3"])
    assert "possible_high_frequency_muscle_emg" in result["channels"][0]["flags"]


def test_candidate_intervals_merge_adjacent_epochs_only():
    epochs = [
        {"start_seconds": 0, "end_seconds": 2, "status": "low_artifact_candidate"},
        {"start_seconds": 2, "end_seconds": 4, "status": "low_artifact_candidate"},
        {"start_seconds": 4, "end_seconds": 6, "status": "review_required"},
        {"start_seconds": 8, "end_seconds": 10, "status": "low_artifact_candidate"},
    ]
    merged = merge_candidate_epoch_intervals(epochs)
    assert merged == [
        {"start_seconds": 0.0, "end_seconds": 4.0, "duration_seconds": 4.0, "epoch_count": 2},
        {"start_seconds": 8.0, "end_seconds": 10.0, "duration_seconds": 2.0, "epoch_count": 1},
    ]
