# NeuroBridge AI — Detailed Development Checklist

**Product direction:** one integrated platform with two coordinated tracks:
- **A — Clinical EEG abnormality screening:** candidate detection and clinician-reviewed interpretation.
- **D — Advanced EEG research:** reproducible signal processing, quantitative analysis, connectivity and research ML.

**Important status rule:** a feature is not “complete” merely because a button or report section exists. It is complete only after implementation, technical tests, representative EDF/BDF testing and documented validation. Clinical claims additionally require expert-labelled data and independent clinical validation.

## 0. Research governance, patient safety and reproducibility

- [ ] Define intended use for every module: research, screening support or clinician decision support.
- [ ] Obtain institutional ethics approval and data-governance approval before using patient-identifiable EEG.
- [ ] Use de-identified subject codes in development and testing.
- [ ] Document consent, data retention, access permissions and deletion procedures.
- [ ] Keep the original EDF/BDF immutable; store processing settings and derived outputs separately.
- [ ] Record software version, algorithm version, processing date, file hash and configuration with each analysis.
- [ ] Maintain a clear audit trail for uploads, re-analysis, annotations and report revisions.
- [ ] Establish a backup and recovery process for the SQLite database and uploaded recordings. Render Free storage is not a reliable persistent research archive.

## 1. EDF/BDF channel capacity — first implementation

**Status: implemented in code and merged; production deployment and real-file verification still need confirmation.**

- [x] Add selectable processing capacities: 24, 32 and 64 scalp EEG channels.
- [x] Raise quantitative EDF/BDF processing capacity from 32 to the selected limit.
- [x] Raise waveform preview capacity to 64 channels.
- [x] Add a channel-capacity selector in the EDF Analyzer.
- [x] Pass the selected capacity to both analysis and waveform preview.
- [x] Report requested capacity separately from actual channels found and processed.
- [x] Use only channels actually present and identified in the recording; never fabricate or pad missing channels.
- [x] Extend PDF channel metrics and spectral-power tables to up to 64 channels.
- [x] Add tests for 24/32/64 limits, fewer available channels and unsupported limits.
- [ ] Confirm the merged version is deployed on Render.
- [ ] Test one real 24-channel EDF/BDF.
- [ ] Test one real 32-channel EDF/BDF.
- [ ] Test one real 64-channel EDF/BDF.
- [ ] Test a recording containing fewer than the selected capacity and confirm the UI explains the actual available count.
- [ ] Test files containing auxiliary channels (EOG, ECG, EMG, respiration, DC, event and reference leads) and confirm these are not incorrectly counted as scalp EEG.
- [ ] Confirm memory use and response time with a full 64-channel, long-duration recording.

**Interpretation note:** “24/32/64-channel support” means the software can process up to that many recognised scalp EEG channels. It does not turn a 20-scalp-channel recording into a 24-channel recording. For example, the previously tested file had 40 total signals but 20 recognised scalp EEG channels; a 24-channel capacity setting should therefore process 20 scalp channels.

## 2. Correctness of the core signal engine — next priority

- [ ] Verify EDF/BDF header sampling frequency, samples per record, duration and channel count.
- [ ] Use the file header as the authoritative sampling frequency; flag disagreement with manually entered study metadata.
- [ ] Verify physical dimension and calibration for every channel (V, mV, µV and vendor-specific units).
- [ ] Validate digital-to-physical scaling and detect malformed or missing calibration metadata.
- [ ] Correct RMS reporting: provide AC RMS after baseline removal, and label any DC-inclusive measure separately.
- [ ] Verify peak-to-peak and RMS calculations mathematically; test that peak-to-peak is not smaller than physically possible relative to the reported RMS.
- [ ] Investigate the previous OZ-channel discrepancy (RMS 85.423 µV versus peak-to-peak 0.7 µV) before trusting those metrics.
- [ ] Add unit tests for constant signals, sine waves, mixed-frequency signals, clipped signals, flat channels and NaN/Inf samples.
- [ ] Verify frequency-band integration, PSD normalisation and relative-power denominator.
- [ ] Confirm delta, theta, alpha, beta and gamma bands are reported only where the Nyquist frequency supports them.
- [ ] Record exact filter design, cutoff frequencies, order, phase behaviour and notch settings.
- [ ] Prevent filter edge artefacts from being treated as genuine events.
- [ ] Add explicit warnings for clipping, saturation, flatline, excessive amplitude, line noise and intermittent electrode failure.
- [ ] Use clinically appropriate microvolt units and configurable display gain; do not use one display scale to infer clinical amplitude.

## 3. Channel labels, electrode maps and montages

- [ ] Support common 10–20, 10–10 and extended 10–10 labels.
- [ ] Support referential labels, linked ears/mastoids and bipolar derivation names.
- [ ] Support legacy labels (T3/T4/T5/T6) and modern equivalents (T7/T8/P7/P8).
- [ ] Add an editable channel-map screen for unknown vendor labels.
- [ ] Let a qualified user designate EEG, EOG, ECG, EMG, reference, ground, trigger and other auxiliary channels.
- [ ] Display the original channel name and the interpreted electrode/derivation name together.
- [ ] Add longitudinal bipolar, transverse bipolar and referential montage views.
- [ ] Preserve montage transformations as derived views; never overwrite the source signal.
- [ ] Detect missing or duplicated channels and show an explicit mapping warning.
- [ ] Add a scalp electrode layout showing which electrodes are present, missing or rejected.

## 4. Pre-processing and artefact handling

- [ ] Add configurable high-pass, low-pass and notch filters with safe defaults and visible settings.
- [ ] Add baseline correction and detrending as explicit reproducible options.
- [ ] Add bad-channel detection with human confirmation.
- [ ] Add ICA-based ocular and muscle artefact review with component plots and reject/keep controls.
- [ ] Add EOG/ECG correlation tools where auxiliary channels are available.
- [ ] Add line-noise and movement artefact review.
- [ ] Keep raw, filtered and cleaned signals separately accessible.
- [ ] Record every preprocessing step in the analysis provenance.
- [ ] Warn when filters or reference changes could distort slow waves, spikes or sharp transients.

## 5. Full-recording analysis and event timeline

- [ ] Replace the current first-120-second-only quantitative pass with chunked full-recording processing.
- [ ] Use overlapping windows and define safe window overlap to avoid boundary misses.
- [ ] Preserve absolute recording timestamps when processing chunks.
- [ ] Handle discontinuous recordings, annotations and gaps.
- [ ] Add a timeline of signal quality issues and candidate events.
- [ ] Allow jumping from a report finding to the original waveform time.
- [ ] Support user-defined event markers and event descriptions.
- [ ] Provide summary counts by event type, channel and time range.
- [ ] Make it clear which segments were processed, skipped or failed.

## 6. Advanced quantitative EEG research (Track D)

- [ ] Welch PSD with configurable windows and overlap.
- [ ] Absolute and relative power by channel and region.
- [ ] FFT and time-frequency visualisation.
- [ ] Wavelet analysis and event-related time-frequency measures.
- [ ] Inter-channel coherence and phase synchrony.
- [ ] Connectivity graphs with transparent estimator and frequency-band settings.
- [ ] Asymmetry indices with explicit left/right channel pairing.
- [ ] Regional summaries with montage-aware channel grouping.
- [ ] Optional ERP/event-related analysis when event markers and experimental design exist.
- [ ] Export derived features to CSV/JSON with units, timestamps and processing metadata.
- [ ] Compare sessions longitudinally only when montage, acquisition and preprocessing are sufficiently comparable.

## 7. Clinical abnormality screening (Track A)

Each detector must first produce **candidate events for review**, not a diagnosis.

- [ ] Background rhythm assessment: frequency, amplitude, continuity, organisation and state.
- [ ] Posterior dominant rhythm candidate measurement where the recording and state permit it.
- [ ] Symmetry and synchrony assessment with montage-aware comparisons.
- [ ] Focal slowing candidate detection.
- [ ] Generalised slowing candidate detection.
- [ ] Spike and sharp-wave candidate detection using morphology, duration, field and after-going slow wave evidence.
- [ ] Periodic and rhythmic pattern candidate detection.
- [ ] Electrographic seizure candidate detection with temporal evolution and expert review.
- [ ] Sleep stage and normal sleep-transient support when adequate sleep is recorded.
- [ ] Activation procedure markers (hyperventilation and photic stimulation) and related review.
- [ ] Explicit handling of paediatric age, sleep state, medication and clinical context.
- [ ] Separate artefact candidates from physiological and pathological candidates.
- [ ] Display the exact channel, timestamp, waveform snippet, detector name, version and evidence for every candidate.
- [ ] Provide reviewer accept/reject/uncertain annotation and reason.
- [ ] Do not label an EEG normal or abnormal solely because no candidate event was generated.

## 8. AI model development and validation

- [ ] Define target labels precisely with neurologist/neurophysiologist input.
- [ ] Build an annotation protocol and inter-rater agreement process.
- [ ] Use licensed, consented or properly governed research datasets.
- [ ] Keep patient-level and preferably site-level separation between training, validation and test data.
- [ ] Prevent data leakage from adjacent windows of the same patient across train/test splits.
- [ ] Establish a locked external test set before tuning final model thresholds.
- [ ] Report sensitivity, specificity, PPV, NPV, F1, AUROC/AUPRC and false positives per hour as appropriate.
- [ ] Report confidence intervals and performance by age, state, montage, site and recording quality.
- [ ] Calibrate model confidence and show uncertainty.
- [ ] Compare against expert annotations and appropriate baseline algorithms.
- [ ] Perform external and prospective validation before clinical-use claims.
- [ ] Version datasets, labels, models, thresholds and evaluation code.
- [ ] Monitor model drift and retain a rollback path.

## 9. Explainable clinician-review report

- [ ] Correct demographics, study ID, recording date, montage and acquisition metadata.
- [ ] Recording duration, source sampling frequency and actual scalp/auxiliary channel counts.
- [ ] Recording quality and artefact summary.
- [ ] Full-recording processing coverage and any excluded segments.
- [ ] Channel-wise descriptive metrics with units.
- [ ] Spectral metrics with method and frequency definitions.
- [ ] Candidate findings with timestamp, channel, waveform image and evidence.
- [ ] Separate automated candidates from reviewer-confirmed findings.
- [ ] Structured clinical review sections for background, symmetry, slowing, sleep, epileptiform activity, seizures and activation procedures.
- [ ] A clear section for the clinician's final impression and recommendations.
- [ ] Reviewer name/role, sign-off and report version fields.
- [ ] Explicit uncertainty and limitations; no fabricated clinical findings.
- [ ] Make the PDF tables readable at 24/32/64 channels and ensure long tables split safely over pages.

## 10. User workflow and usability

- [ ] iPad, desktop and laptop upload testing.
- [ ] Clear file status: uploaded, parsing, analysed, failed or missing.
- [ ] Analysis progress and recoverable error messages.
- [ ] Persist selected study and file across tab changes.
- [ ] Display actual channel availability before analysis.
- [ ] Save analysis configuration and allow reproducible re-analysis.
- [ ] Avoid automatic association of unrelated unassigned files without user confirmation.
- [ ] Add a safe download/export path for original files and derived results.
- [ ] Provide a demo mode using synthetic or public de-identified EEG data.

## 11. Hosting, privacy and reliability

- [ ] Verify persistent storage before using the system for research records.
- [ ] Back up database and original recordings independently of the web service.
- [ ] Test restoration from backup.
- [ ] Restrict access and rotate shared credentials.
- [ ] Add per-user accounts and role-based permissions before multi-user clinical research.
- [ ] Encrypt transport and protect stored files.
- [ ] Add upload integrity checks and file-format validation.
- [ ] Add resumable uploads for large recordings.
- [ ] Add audit logging and retention controls.
- [ ] Monitor server memory, CPU, request duration and failures on 64-channel long recordings.

## 12. Separate future research track — thought mapping / BCI

- [ ] Keep thought-mapping experiments separate from routine clinical EEG interpretation.
- [ ] Design cue-based imagined-speech and overt-speech protocols with randomised trials.
- [ ] Capture precise stimulus, trial and response timestamps.
- [ ] Record EOG/EMG and other confound controls where appropriate.
- [ ] Build subject/session-aware datasets with explicit task labels.
- [ ] Evaluate within-subject and cross-subject generalisation separately.
- [ ] Compare performance against chance and non-neural baselines.
- [ ] Do not equate classifier labels with unrestricted access to private thoughts.

## 13. Release gates

- [ ] Engineering tests pass.
- [ ] 24/32/64-channel sample recordings pass ingestion and waveform tests.
- [ ] Channel mapping and auxiliary exclusion are reviewed.
- [ ] Signal metrics pass synthetic-signal and real-recording quality checks.
- [ ] Full-recording coverage and timestamps are verified.
- [ ] Report values reconcile with independently computed reference values.
- [ ] Clinical detectors are evaluated on independent expert-labelled data.
- [ ] Security, privacy, backup and recovery checks pass.
- [ ] Specialist review and institutional approvals are documented before any clinical pilot.
