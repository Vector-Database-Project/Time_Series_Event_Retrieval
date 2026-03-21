# ECG Pipeline Evolution, Single-Event Walkthrough

This figure traces one valid ECG event through the implemented preprocessing flow.  
It shows a larger raw context window, the fixed beat-centered time-domain chunk, and the frequency-domain representation obtained by mean-centering the chunk and applying `rFFT`.

# Selected Event Summary

| Field | Value |
|---|---:|
| Record | 100 |
| Annotation sample | 370 |
| Beat label | N |
| Lead used in figure | MLII (index 0) |
| TD chunk length | 361 samples |
| TD duration | 1.0028 s |
| Sampling rate | 360 Hz |
| Raw context range | [10, 731) samples |
| TD clip range | [190, 551) samples |
| TD mean before FFT | -0.309875 |
| TD mean after centering | -0.000000 |
| FFT bins stored | 181 |
| Dominant non-DC bin | 4.986 Hz |