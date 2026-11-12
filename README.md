# picospec

A small neural-network inference engine in C++17, written from scratch, with
hand-written Arm NEON kernels and integer-only (int8) inference. Its main
model tells apart 30 isolates of disease-causing bacteria and yeast from
measured Raman spectra. A second,
smaller model denoises spectra from the phone-camera spectrometer I built in
2024.

No dependencies beyond a C++17 compiler and CMake. About 1,100 lines of C++,
plus 500 lines of tests.

## Why I built it

I study bioanalytical science, and spectra are what my instruments produce.
In 2024 I built a spectrometer from a CD (diffraction grating), a slit made of
two sharpener blades and an old iPhone as the detector. The detector was a
phone: a battery-powered Arm device. This project is how I am learning what it
takes to run a neural network on a device like that, quickly and without
floating point.

It is a small runtime with nothing hidden inside a framework: the model file
format, the quantisation maths, the convolution kernels in NEON intrinsics,
and tests that check the fast kernels compute exactly what the slow ones do.

## The task: which bacterium is this?

The data are the public bacteria-ID Raman set from Ho et al., *Rapid
identification of pathogenic bacteria using Raman spectroscopy and deep
learning*, Nature Communications 10, 4927 (2019)
([code](https://github.com/csho33/bacteria-ID); the data are on the Dropbox
link in that repository). Each spectrum is 1,000 samples; the label is one of
30 bacterial and yeast isolates. These are real, noisy measurements, and the
training and test spectra were measured on separate occasions. The dataset is
not stored in this repository.

| set | spectra | used for |
|---|---:|---|
| reference | 60,000 | training (4 epochs) and int8 calibration (1,000 of them) |
| fine-tune | 3,000 | fine-tuning (50 epochs), all 3,000 spectra |
| test | 3,000 | the accuracy figures below, nothing else |

The model is four stride-2 convolutions (1→16, 16→32, 32→64, 64→64 channels;
kernels 9, 7, 5, 5) with ReLU, then one dense layer to 30 scores: 155,614
parameters, 3,659,200 multiply-accumulates per spectrum. No batch-norm and no
pooling, so every layer is one the engine runs directly.

## Results

### Accuracy

Test accuracy on the 3,000 test spectra (30 classes, chance is 3.3 %).
`python/evaluate_classifier.py` reproduces the table; the raw output is in
[results/classifier_accuracy.md](results/classifier_accuracy.md).

| method | test accuracy |
|---|---:|
| logistic regression on the raw spectra | 72.33 % |
| logistic regression, standardised inputs | 83.23 % |
| CNN float32, PyTorch | 84.67 % |
| CNN float32, this engine | 84.67 % |
| CNN int8, Python integer reference | 84.97 % |
| CNN int8, this engine (all four kernels) | **84.97 %** |

Read honestly:

- **Quantising to int8 cost no accuracy.** The int8 model scores 84.97 %
  against 84.67 % for float32. That 0.3-point difference is not an
  improvement, it is noise: one accuracy figure on 3,000 spectra is only good
  to about ±0.7 points. Quantisation changed the predicted class on 1.9 % of
  the spectra, and slightly more of those changes happened to land right
  than wrong.
- **The engine computes what the reference computes.** All 90,000 int8 scores
  (3,000 spectra × 30 classes) are identical, bit for bit, between the Python
  integer reference and each of the four C++ kernels. The float32 engine
  picks the same class as PyTorch on every spectrum; the scores differ by at
  most 0.0002 on values up to 118.
- **The CNN does not clearly beat a well-tuned linear model.** With the same
  training settings the CNN is 12 points ahead of logistic regression. But
  once the inputs are standardised (each sample position scaled by its mean
  and spread over the reference set) and weight decay is raised, logistic
  regression reaches 83.23 %, which is within sampling error of the CNN.
  I chose those two settings by looking at test accuracy, which flatters the
  baseline, but I would rather show the strong baseline than the weak one.
- **One training run.** All 3,000 fine-tune spectra went into training, so
  there is no separate validation set; there is one run with one random
  seed; and the layer sizes and epoch counts were fixed after one earlier
  trial that was scored on this same test set. Treat 85 % as "about 85 %",
  not as a tuned or repeated result. For scale, the dataset's authors report
  above 82 % on this task with a much deeper network (their figure, not
  reproduced here).

### Speed

Measured on an Apple M1 (one performance core, Apple clang 17, `-O3`), one
1,000-sample spectrum per inference. Median of 300 timing samples;
`scripts/bench.sh` reproduces the table ([results/bench.md](results/bench.md)).

| model | kernel | median time | vs. plain float |
|---|---|---:|---:|
| float32 | plain C++, auto-vectoriser off | 3,027 µs | 1.0× |
| float32 | plain C++, `-O3` | 1,869 µs | 1.6× |
| float32 | NEON (FMA, 4 accumulators) | 232 µs | 13.1× |
| int8 | plain C++, auto-vectoriser off | 1,408 µs | 2.2× |
| int8 | plain C++, `-O3` (compiler vectorises) | 185 µs | 16.4× |
| int8 | NEON, Armv8.0 (`SMULL` + `SADALP`) | 109 µs | 27.7× |
| int8 | NEON, Armv8.2 DotProd (`SDOT`) | **60 µs** | **50×** |

- The int8 model with the `SDOT` kernel classifies a spectrum in 60 µs: 50×
  faster than straightforward float code and 3.9× faster than hand-written
  float NEON, at the same accuracy.
- The compiler already gets 7.6× on the int8 loop by itself. The hand-written
  kernels add another 3.1× on top of that, mostly by computing four output
  channels per pass over the input window and by using `SDOT`.
- The int8 model file is 158,060 bytes against 622,588 bytes for float32.

## Second example: denoising my own spectrometer

![Real traces from my spectrometer, before and after the int8 network](results/real_traces.png)

*The two real traces from my instrument (torch light through water and
through diluted Ribena), run through the int8 engine. Qualitative only: there
is no noise-free ground truth for these, so no accuracy is claimed here.*

The first model I built for the engine is a 4-layer, 2,769-parameter CNN
that takes a 256-sample spectrum and returns a cleaner one. It uses the same
kernels with stride 1 and "same" padding. My spectrometer gave me only a
handful of traces, so this model is trained and scored on **simulated**
spectra ([python/spectra.py](python/spectra.py)), which is why it is the
second example and not the first.

RMSE against the noise-free spectrum on 4,000 unseen simulated spectra
(lower is better; `python/evaluate.py`):

| method | RMSE |
|---|---:|
| noisy input | 0.02686 |
| median-3 + Gaussian smoothing, width tuned on validation data | 0.01015 |
| median-3 + Savitzky-Golay, window tuned on validation data | 0.00902 |
| CNN float32 (PyTorch and this engine agree to 4e-7) | **0.00858** |
| CNN int8 (this engine, bit-identical to the Python integer reference) | 0.00922 |

The float network beats the best tuned classical filter by 5 %; the int8
network gives that back and lands 2 % *behind* tuned Savitzky-Golay. On
smooth spectra a 1964 filter is hard to beat, and I would rather show that
than hide it. It runs in 23 µs per spectrum with the `SDOT` kernel, 20×
faster than plain float code.

## How it works

**Layout.** Activations are stored channel-last, so the receptive field of
one output sample is one contiguous run of bytes and a convolution is a dot
product of two such runs. In the classifier's third layer that is 5 taps × 32
channels = 160 bytes, exactly ten 16-byte NEON registers, with no gather and
no tail.

**Stride, padding and the dense layer**
([include/picospec/kernels.hpp](include/picospec/kernels.hpp)). Every layer
has its own stride and padding: output sample *i* reads the *k* input samples
starting at *i* × stride − pad, and a layer over *n* samples produces
⌊(n + 2·pad − k) / stride⌋ + 1 outputs. A stride-2 layer halves the spectrum
(1000 → 500 → 250 → 125 → 63). A dense layer needs no code of its own: it is
a convolution with no padding whose kernel is as long as its input (63
samples × 64 channels = 4,032 values), which gives exactly one output sample
holding the 30 class scores. `argmax` over those scores is the prediction.

**Quantisation** ([python/export.py](python/export.py)). Post-training,
calibrated on 1,000 training spectra. Activations are asymmetric int8 with
one scale and zero-point per tensor; weights are symmetric int8 with one
scale per output channel; biases are int32. The input zero-point is folded
into the bias offline, so the kernel is a pure int8 × int8 dot product.
Padding is done by writing the zero-point (the int8 code for 0.0) into the
buffer margins, so the kernels have no edge cases.

**Requantisation** ([include/picospec/requant.hpp](include/picospec/requant.hpp)).
Each int32 sum is rescaled by a real factor M < 1, stored as a Q0.31
multiplier and a right shift. The scalar code is written as the same two
steps the NEON instructions `SQRDMULH` and `SRSHL` perform, which is why
every kernel produces identical bits, not merely close values.

**Kernels** ([src/](src/)).

- `kernels_scalar.cpp`: the reference. Compiled twice, once with the
  auto-vectoriser disabled, to separate what the compiler does from what the
  intrinsics add.
- `kernels_neon.cpp`: Armv8.0 baseline. `SMULL`/`SMULL2` widen 8×8→16 bit,
  `SADALP` accumulates into 32 bit. The float kernel uses FMA with four
  independent accumulators so consecutive FMAs do not wait on each other.
- `kernels_dotprod.cpp`: `SDOT`, 16 multiplies and 4 sums in one instruction.
  It shares its body with the Armv8.0 kernel through
  `conv_q8_neon_impl.inc`; the two differ in one macro.

**Runtime dispatch** ([src/cpu_features.cpp](src/cpu_features.cpp)). Only
`kernels_dotprod.cpp` is compiled with `-march=armv8.2-a+dotprod`. The CPU is
asked at start-up (`getauxval` on Linux, `sysctl` on macOS) whether it has
`SDOT`, so the same binary still runs on a Cortex-A53 or A72, which do not.

**Design** ([include/picospec/model.hpp](include/picospec/model.hpp)). An
abstract `Model` with two implementations, `FloatModel` and `QuantizedModel`,
behind one `run()` call. The caller owns a `Workspace`, so inference
allocates nothing after the first call and one model can serve several
threads.

**Model files** ([src/model.cpp](src/model.cpp)). A `.psm` file (format
`PSM2`) is a short header and, per layer, its shape (channels, kernel,
stride, padding, activation) and weights. The loader treats the file as
untrusted: every read is bounds-checked, and an int8 layer whose sums could
overflow 32 bits is refused.

## Testing

`ctest` runs 20,000+ checks in under a second:

- fixed-point rounding against hand-worked cases and double arithmetic;
- every optimised int8 kernel against the reference, bit for bit, over 20
  layer shapes × five lengths: "same", stride 2 to 4, no padding, and
  dense-shaped layers;
- stride and padding against small cases worked out by hand, including a
  two-layer classifier whose scores can be checked on paper;
- both int8 models against an independent integer implementation in NumPy:
  identical output on every kernel;
- both float models against PyTorch;
- the loader against truncated files, bad magic, the old format version,
  trailing bytes and impossible shapes: it must throw, never read out of
  bounds.

The golden vectors for the classifier are synthetic inputs from a fixed-seed
generator, not dataset spectra. The suite also passes under AddressSanitizer
and UBSan, and as an x86-64 build (the portable path, no NEON). CI is
configured for Linux x86-64, Linux AArch64 and macOS.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/picospec info models/classifier_int8.psm
./build/picospec bench --model models/classifier_f32.psm --model models/classifier_int8.psm --len 1000
./build/picospec classify --model models/classifier_int8.psm --in spectra.f32
./build/picospec run --model models/denoiser_int8.psm --in spectra.f32 --out denoised.f32
```

Spectra files are raw little-endian float32. `classify` prints one class
index (0 to 29) per spectrum; `run` writes the raw model output.

To retrain, re-quantise and regenerate every number in this README, download
the dataset into `../raman-data` (see
[python/train_classifier.py](python/train_classifier.py)) and run:

```bash
python3 -m pip install -r python/requirements.txt
scripts/reproduce.sh
```

Training the classifier takes about four and a half minutes on an M1.

## Limitations

- One dataset, one training run, no validation set (see the notes under the
  accuracy table). I have not evaluated the dataset's clinical sets.
- The CNN is not clearly better than tuned logistic regression on this test
  set. The project is about running the network, not about a better model.
- The denoiser's accuracy is measured on simulated spectra only.
- Timings are from one machine, a desktop-class core. I have not yet measured
  on a Cortex-A or Cortex-M board, and I have not measured energy.
- One layer type (1-D convolution with any stride and padding, optional
  ReLU; a dense layer is a special case of it). No pooling, batch-norm or
  skip connections. Single-threaded.
- The first layer has one input channel, so its 9-byte window falls back to
  scalar code. It is 2 % of the classifier's arithmetic.

## What I would do next

1. Train properly: hold out part of the fine-tune set for validation, repeat
   over several seeds, and report the spread.
2. Run it on a Raspberry Pi (Cortex-A72: NEON, no `SDOT`) and a Cortex-M
   board, where the Armv8.0 path and the allocation-free design matter.
3. Compare against CMSIS-NN / XNNPACK on the same model to see how far a
   first attempt is from production kernels, and why.
4. Quantisation-aware training for the denoiser, where int8 does lose
   accuracy.

## Repository map

```
include/picospec/   public headers: kernels, requantisation, model
src/                kernels (scalar, NEON, DotProd), loader, CPU detection
tools/              command-line tool: info, run, classify, bench
tests/              unit tests and golden vectors
python/             training, quantisation and evaluation for both models
models/             trained float32 and int8 models (.psm)
data/               the two real traces from my spectrometer
results/            measured accuracy and benchmark output
```

## Data and reproducibility

The instrument, the traces and the question are mine; the Raman dataset is Ho
et al.'s public one. Every number above is produced by the scripts in this
repository on my own machine (Apple M1), except the one figure quoted from the
dataset's paper.

MIT licence.
