# NeuroNet vision

Updated 2026-09-06. Baseline: `master` at `01e0093`.

NeuroNet will be a small, inspectable C++17 machine-learning library for learning,
experimentation, and embedding modest CPU models into applications. A developer
should be able to build it, train a reproducible small model, save it, and obtain
the same predictions from a separate C++ or Python program.

The project's distinctive value is transparent implementations: dense training,
evolutionary optimization, matrix operations, and sequence-model building blocks
that developers can read and test. Correctness, a dependable public API, and
measured performance take priority over adding more model families.

## Where the code is today

The original feature checklist is implemented on `master`: dense networks with
backpropagation, genetic optimization and early stopping, CNN/RNN/LSTM primitives,
transformer encoder and decoder components, Python bindings, MNIST loading,
logging, and package-generation infrastructure. These are different maturity
levels, not a claim that every model is trainable or production ready.

- Dense networks provide the clearest end-to-end training and serialization path.
- CNN, RNN, and LSTM layers are standalone forward primitives; sequence training
  and integration into a general trainable model remain work.
- Transformers provide encoder and encoder-decoder forward computation. Dropout
  is a placeholder, and full training and encoder-decoder persistence are gaps.
- Python bindings cover the dense workflow. Release packaging exists, but clean
  installation and downstream consumption need explicit validation.

Evidence lives in `src/neural_network/`, `src/transformer/`,
`src/python_bindings/pyneuronet.cpp`, `tests/`, and `CMakeLists.txt`.
The older detached local checkout and `development` do not represent this baseline.

## Product commitments

1. **Trust the answer.** Numerical reference tests and gradient checks establish
   behavior; shape-only tests do not establish mathematical correctness.
2. **Repeat an experiment.** Seeded randomness, explicit losses and metrics, and
   saved configuration make training results explainable and reproducible.
3. **Embed with confidence.** Preserve existing public call shapes where possible;
   introduce unambiguous additive APIs and test external consumers.
4. **Measure useful workloads.** Benchmark dense, rectangular, and matrix-vector
   products on the same compiler and thread settings before accepting speedups.
5. **Finish a usable path.** Prioritize a documented C++ and Python train/save/load
   example over adding another isolated layer.

## Scope and success

The next release is a reliability milestone, provisionally 0.2. It is ready only
when clean builds and tests pass, old and new public API calls compile, a fresh
installation supports a consumer, and the dense training example reproduces its
predictions after serialization. Version numbers are planning labels, not dates
or release promises.

GPU/distributed training, large-language-model training, broad framework parity,
and further model families are deferred until those gates hold. Advanced layers
remain explicitly experimental until numerical tests and a usable training path
support stronger claims.

See [the roadmap](ROADMAP.md) for delivery gates and [TODO.MD](TODO.MD) for the
ordered work queue.
