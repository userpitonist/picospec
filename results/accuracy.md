Test set: 4000 simulated spectra x 256 samples

| method | RMSE vs clean |
|---|---:|
| noisy input (no processing) | 0.02686 |
| median-3 + Gaussian smoothing (sigma=2.5, tuned) | 0.01015 |
| median-3 + Savitzky-Golay (window=23, order 3, tuned) | 0.00902 |
| CNN float32, PyTorch | 0.00858 |
| CNN float32, C++ engine | 0.00858 |
| CNN int8, Python integer reference | 0.00922 |
| CNN int8, C++ engine | 0.00922 |

f32/scalar-novec: max |C++ - PyTorch|: 3.5762786865234375e-07
f32/scalar: max |C++ - PyTorch|: 4.172325134277344e-07
f32/neon: max |C++ - PyTorch|: 3.5762786865234375e-07
int8/scalar-novec: outputs identical to integer reference: True
int8/scalar: outputs identical to integer reference: True
int8/neon: outputs identical to integer reference: True
int8/neon-dotprod: outputs identical to integer reference: True
