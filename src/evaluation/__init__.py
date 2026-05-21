"""Diarization evaluation harness (AVA-AVD).

Submodules:
    clip_avaavd        - download + slice AVA-AVD test clips.
    run_eval_pipeline  - loop the main `Pipeline` over the test split.
    score_diarization  - pyannote.metrics DER/JER scorer + report writer.
"""
