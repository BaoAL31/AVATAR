# AVATAR: Audio-Visual Attribution Transcription And Recognition

## Pipeline Overview

```mermaid
graph TD
    %% ==========================================
    %% DARK MODE STYLES (No class assignment to avoid parser errors)
    %% ==========================================
    classDef input fill:#1e3a5f,stroke:#66ccff,stroke-width:2px,color:#fff;
    classDef process fill:#2c2c2c,stroke:#999999,stroke-width:2px,color:#fff;
    classDef data fill:#3b2e12,stroke:#d4a566,stroke-width:2px,color:#fff;
    classDef output fill:#1a3b25,stroke:#4caf50,stroke-width:2px,color:#fff;

    %% NODES (Quotes + <br/> for guaranteed line breaks)
    RV["Raw Video"]:::input
    RA["Raw Audio Track<br/>16kHz waveform"]:::input
    AU["AU Features NPZ<br/>24-D/frame"]:::data

    subgraph AVD [AV-Diarization]
        FD["Face Detection<br/>& Tracking"]:::process
        VAD["Voice Activity Detection<br/>Silero"]:::process
        FC["Face Clustering<br/>VGGFace"]:::process
        SE["Speaker Embedding<br/>ECAPA-TDNN"]:::process
        DI["Diarization<br/>ASD + Face/Audio Fusion"]:::process
    end

    SS["Speaker Segments<br/>Timestamp segments"]:::data
    FVC["Face Video Crops<br/>Aligned to each track"]:::data
    MCE["Mouth Crop Extraction"]:::process

    USR["USR<br/>Unified Speech Recognition"]:::process

    TR["Transcriptions<br/>Per unique track"]:::data
    FO["Speaker-attributed Transcript<br/>[start -- end] speaker: transcription"]:::output

    %% CONNECTIONS (Using ==> for thicker lines, -.-> for dashed LoRA injection)
    RV ==> FD
    RV ==> VAD
    FD ==> FC
    VAD ==> SE
    FC ==> DI
    SE ==> DI
    
    DI ==> SS
    DI ==> FVC
    FVC ==> MCE
    
    MCE ==> USR
    RA ==> USR
    AU ==> USR
    
    USR ==> TR
    SS ==> FO
    TR ==> FO
```

## USR Block Detail

```mermaid
graph LR
    classDef process fill:#2c2c2c,stroke:#999999,stroke-width:2px,color:#fff;
    classDef input fill:#1e3a5f,stroke:#66ccff,stroke-width:2px,color:#fff;
    classDef data fill:#3b2e12,stroke:#d4a566,stroke-width:2px,color:#fff;
    classDef contrib fill:#3f1f5f,stroke:#d18cff,stroke-width:2px,color:#fff;

    MCE["Mouth Crop Clip<br/>(B,T,H,W)"]:::input
    RA["Raw Audio Track<br/>16kHz waveform"]:::input
    AU["AU Sequence<br/>(B,T,24)"]:::contrib

    VENC["Visual Frontend<br/>Conv3D + ResNet to tokens (B,T,D)"]:::process
    APRE["Audio Feature Map<br/>STFT / log-Mel"]:::process
    APROJ["Audio Frontend<br/>2D Conv stack + projection"]:::process
    AUENC["AU Frontend<br/>Time align + MLP: (B,T,24) to (B,T,D)"]:::contrib
    VSUM["Visual-AU Fusion<br/>Per-step add: z_t = v_t + au_t"]:::contrib
    CAT["Cross-modal Fusion<br/>Concatenate Audio with (Visual+AU)"]:::process
    TENC["Temporal Encoder<br/>Self-attention over time"]:::process
    ENC["AV Encoder Output<br/>Contextual token sequence"]:::process
    DEC["Beam Search Decoder<br/>CTC + Attention + LM"]:::process
    LORA["LoRA Adapters<br/>Low-rank trainable updates in Linear/Conv"]:::contrib
    TR["Transcriptions<br/>Per unique track"]:::data

    MCE ==> VENC
    RA ==> APRE
    APRE ==> APROJ
    AU ==> AUENC
    VENC ==> VSUM
    AUENC ==> VSUM
    APROJ ==> CAT
    VSUM ==> CAT
    CAT ==> TENC
    TENC ==> ENC
    ENC ==> DEC
    LORA -.-> VENC
    LORA -.-> APROJ
    LORA -.-> TENC
    LORA -.-> DEC
    DEC ==> TR
```

Temporal intuition:
- `Conv3D` already mixes nearby frames, then emits a sequence of visual tokens `[v_1..v_T]`.
- AU is also a sequence, aligned to the same timeline (`T`), then projected.
- Fusion is timestep-wise (`z_t = v_t + au_t`), but the model still sees the full sequence `[z_1..z_T]`.
- Transformer layers then model long-range temporal dependencies across these fused tokens.

Notation:
- `D` = feature width (embedding dimension) of each timestep token used by the encoder.
