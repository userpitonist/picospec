Test set: 3000 measured Raman spectra x 1000 samples, 30 classes (chance = 3.3 %)

| method | test accuracy |
|---|---:|
| logistic regression, raw spectra | 72.33 % |
| logistic regression, standardised inputs (settings picked on the test set) | 83.23 % |
| CNN float32, PyTorch | 84.67 % |
| CNN float32, C++ engine | 84.67 % |
| CNN int8, Python integer reference | 84.97 % |
| CNN int8, C++ engine (all backends) | 84.97 % |

int8 vs float32 (PyTorch): +0.30 points; same predicted class on 98.13 % of spectra
sampling error of one accuracy figure on 3000 spectra: about +/- 0.7 points (one standard error)

f32/scalar-novec: max |C++ - PyTorch| (scores reach 118): 0.00020599365234375
f32/scalar-novec: same predicted class as PyTorch on every spectrum: True
f32/scalar: max |C++ - PyTorch| (scores reach 118): 0.00020599365234375
f32/scalar: same predicted class as PyTorch on every spectrum: True
f32/neon: max |C++ - PyTorch| (scores reach 118): 0.00018310546875
f32/neon: same predicted class as PyTorch on every spectrum: True
int8/scalar-novec: all 90000 scores identical to integer reference: True
int8/scalar: all 90000 scores identical to integer reference: True
int8/neon: all 90000 scores identical to integer reference: True
int8/neon-dotprod: all 90000 scores identical to integer reference: True
f32: `picospec classify` labels == argmax of reference scores: True
int8: `picospec classify` labels == argmax of reference scores: True
