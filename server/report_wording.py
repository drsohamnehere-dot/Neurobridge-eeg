"""Conservative wording review for text extracted from EEG report PDFs.

This module does not inspect EEG waveforms and must not be used as a diagnostic tool.
"""
import re


CONCEPTS = {
    "epileptiform_discharge": r"epileptiform|spikes?|sharp waves?|polyspikes?",
    "seizure_activity": r"seizures?|ictal|status epilepticus",
    "focal_slowing": r"focal slowing|regional slowing|focal delta|focal theta",
    "generalized_slowing": r"generalized slowing|diffuse slowing|generalised slowing|generalised delta|generalized delta",
    "background_abnormality": r"disorganized background|disorganised background|attenuated background|background asymmetry|asymmetric background|absent posterior dominant rhythm|loss of posterior dominant rhythm",
}
UNCERTAIN = re.compile(
    r"\b(possible|possibly|suspicious for|questionable|equivocal|cannot exclude|cannot rule out|may represent|versus)\b",
    re.I,
)
NEGATED_PREFIX = re.compile(
    r"\b(no|not|without|absence of|negative for|free of|lack of)\s+(?:[a-z]+\s+){0,4}$",
    re.I,
)
NORMAL = re.compile(
    r"\b(normal EEG|EEG within normal limits|within normal limits|unremarkable EEG|normal awake EEG)\b",
    re.I,
)


def review_report_wording(text: str) -> dict:
    pages = re.findall(r"\[PAGE (\d+)\]\s*([\s\S]*?)(?=\[PAGE \d+\]|$)", text or "", re.I)
    if not pages and (text or "").strip():
        pages = [("?", text)]
    results = []

    for concept, pattern in CONCEPTS.items():
        evidence = []
        statuses = []
        for page_number, page_text in pages:
            sentences = re.split(r"(?<=[.!?;])\s+|\n+", page_text)
            for sentence in sentences:
                sentence = sentence.strip()
                match = re.search(pattern, sentence, re.I)
                if not match:
                    continue
                prefix = sentence[max(0, match.start() - 100):match.start()]
                suffix = sentence[match.end():match.end() + 100]
                if UNCERTAIN.search(sentence):
                    status = "context_uncertain"
                elif NEGATED_PREFIX.search(prefix) or re.search(
                    r"\b(not seen|not observed|not identified|not present|were absent|was absent)\b",
                    suffix, re.I
                ):
                    status = "reported_absent"
                else:
                    status = "reported_present"
                statuses.append(status)
                evidence.append({"page": page_number, "wording": sentence[:360]})
        if evidence:
            if "reported_present" in statuses:
                status = "reported_present"
            elif "context_uncertain" in statuses:
                status = "context_uncertain"
            else:
                status = "reported_absent"
            results.append({"finding": concept, "status": status, "evidence": evidence[:5]})

    normal_evidence = []
    for page_number, page_text in pages:
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", page_text):
            if NORMAL.search(sentence) and not UNCERTAIN.search(sentence) and not re.search(
                r"\\b(not|no longer|outside|abnormal|abnormality)\\b.{0,60}\\b(normal EEG|within normal limits|unremarkable EEG)\\b",
                sentence, re.I
            ):
                normal_evidence.append({"page": page_number, "wording": sentence.strip()[:360]})
    if normal_evidence:
        results.append({
            "finding": "normal_report_wording",
            "status": "reported_present",
            "evidence": normal_evidence[:5],
        })

    return {
        "method": "Conservative keyword and sentence-context review of extracted report text",
        "items": results,
        "note": (
            "These labels describe wording in the uploaded report only. They do not confirm "
            "or exclude an EEG abnormality, seizure, or diagnosis. Verify the extracted wording "
            "against the original report and have the complete EEG reviewed by a qualified clinician."
        ),
    }
