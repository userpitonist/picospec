date:     2026-11-12T17:43:29Z
os:       Darwin 24.6.0 arm64
cpu:      Apple M1
compiler: Apple clang version 17.0.0 (clang-1700.0.13.5)

Bacteria classifier, one 1000-sample spectrum per inference:

| model | backend | median (us) | min (us) | p90 (us) | spectra/s |
|---|---|---:|---:|---:|---:|
| classifier_f32 | scalar-novec | 3027.17 | 3003.00 | 3135.54 | 330 |
| classifier_f32 | scalar | 1869.27 | 1844.73 | 1937.73 | 535 |
| classifier_f32 | neon | 231.62 | 223.14 | 315.16 | 4317 |
| classifier_int8 | scalar-novec | 1407.75 | 1351.90 | 1720.12 | 710 |
| classifier_int8 | scalar | 184.67 | 183.21 | 207.22 | 5415 |
| classifier_int8 | neon | 109.46 | 108.07 | 114.42 | 9136 |
| classifier_int8 | neon-dotprod | 60.04 | 59.24 | 61.74 | 16656 |

Denoiser, one 256-sample spectrum per inference:

| model | backend | median (us) | min (us) | p90 (us) | spectra/s |
|---|---|---:|---:|---:|---:|
| denoiser_f32 | scalar-novec | 455.27 | 445.34 | 474.49 | 2197 |
| denoiser_f32 | scalar | 258.17 | 255.43 | 267.72 | 3873 |
| denoiser_f32 | neon | 50.66 | 49.73 | 52.44 | 19739 |
| denoiser_int8 | scalar-novec | 249.34 | 244.15 | 255.27 | 4011 |
| denoiser_int8 | scalar | 58.69 | 57.25 | 59.35 | 17039 |
| denoiser_int8 | neon | 29.55 | 28.82 | 30.14 | 33841 |
| denoiser_int8 | neon-dotprod | 22.94 | 22.37 | 23.32 | 43592 |
