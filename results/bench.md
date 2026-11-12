date:     2026-11-10T20:42:16Z
os:       Darwin 24.6.0 arm64
cpu:      Apple M1
compiler: Apple clang version 17.0.0 (clang-1700.0.13.5)

| model | backend | median (us) | min (us) | p90 (us) | spectra/s |
|---|---|---:|---:|---:|---:|
| float32 | scalar-novec | 455.24 | 445.57 | 504.81 | 2197 |
| float32 | scalar | 260.60 | 255.33 | 277.91 | 3837 |
| float32 | neon | 49.47 | 49.05 | 57.08 | 20214 |
| int8 | scalar-novec | 287.33 | 279.65 | 310.27 | 3480 |
| int8 | scalar | 57.98 | 57.58 | 62.12 | 17249 |
| int8 | neon | 29.27 | 29.13 | 30.59 | 34162 |
| int8 | neon-dotprod | 22.85 | 22.67 | 24.95 | 43765 |
