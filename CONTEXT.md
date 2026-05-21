# Glossary

Domain terms for the LRS2 fusion evaluation. Terms only; no implementation
details.

## sample

A synthetic multi-speaker test instance built by concatenating K=3 LRS2
validation clips chosen at random with a fixed seed into a single audio+video
stream. The manifest does not carry speaker identity, so speaker
distinctness across the three clips is probabilistic, not guaranteed; LRS2's
≥1000 speakers over ~5000 val clips makes intra-triple collisions rare. The
clustering half of diarization (re-identifying a speaker after a gap) is not
exercised by this corpus and is out of scope for this evaluation.

## USR-per-clip

The oracle baseline. USR is run independently on each of a sample's source
clips against their original ground-truth crops; the predictions are
concatenated in clip order and scored against the concatenated reference
transcript.

## Pipeline-on-concat

The unit under test. The full pipeline (face detection, tracking, mouth crop,
AV-diarization, USR per diarized segment) is run on the stitched sample. The
segment-level predictions are concatenated in time order and scored against
the concatenated reference transcript.

## Fusion success

The combination of two conditions, evaluated over the set of non-failed
samples:

1. `median(Pipeline-on-concat WER) − median(USR-per-clip WER) ≤ 10 pp`.
2. `Tracker-failure rate ≤ 20%`.

Failed samples are reported as the tracker-failure rate and are excluded from
the WER medians, never folded into them.

## Tracker-failure rate

The fraction of samples for which the pipeline drops or merges a source clip's
speaker track. Reported alongside, never folded into, WER.

## AV-diarization

The diarization stage of the pipeline. Consumes the full audio+video stream
and emits per-speaker time segments. Always uses both modalities; the "AV" in
the pipeline's name refers to this stage.

## USR modality

The input modality used by the USR speech-recognition stage when transcribing
a segment. Independent of AV-diarization. Can be V (mouth crops only),
A (audio only), or AV (both). Choosing V-only at USR does not make the
upstream stage less AV.
