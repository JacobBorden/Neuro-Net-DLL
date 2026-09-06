# NeuroNet work queue

Updated 2026-09-06 against `master` at `01e0093`. Priorities implement
[the vision](VISION.md) and [milestone gates](ROADMAP.md). Check a task only when its
acceptance evidence exists; a proposed PR is not a delivered feature.

## P0 — Unblock integration

- [ ] Repair PR #45 with an additive, unambiguous per-call generation API. Keep the
  existing constructor and patience call semantics; test legacy and new callers,
  repeated calls, zero/invalid counts, and early stopping on current master.
- [x] Consolidate duplicate PRs #214/#215 (closed 2026-09-06). The dense loop
  interchange already exists on master; issue #106 retains the narrow-product finding.
- [ ] Fix #106: preserve an accumulator for single-column multiplication; compare
  reference results and benchmark square, rectangular, and vector workloads.
- [ ] Retire or reconfigure the external stale optimization generator (#197).
  Verify its actual schedule/configuration; repository PR cleanup alone is insufficient.

## P1 — Reliability before feature expansion

- [ ] Enforce C++17 in CMake, modernize the minimum version, discover/link optional
  OpenMP, and run clean Linux/Windows builds with CTest output on failure.
- [ ] Upgrade checkout actions (#88); verify workflow permissions and artifact
  paths without weakening required checks.
- [ ] Register and run Python smoke tests reliably after interpreter discovery;
  verify test enumeration on a clean configure and import the built extension.
- [ ] Add numerical gradient checks for dense losses/activations and reference
  values for CNN/RNN/LSTM and attention masks, including invalid/empty shapes.
- [ ] Make randomized training tests assert deterministic invariants with explicit
  seeds; preserve the same initial population when testing elitism/improvement.
- [ ] Wire benchmark examples into CMake and record compiler, flags, dimensions,
  repetitions, and thread counts. Verify #153's merged API repair before closing it.
- [ ] Resolve the orphan `dependencies/matrix` gitlink metadata without deleting a
  user's dependency checkout; verify a fresh clone and Actions cleanup.
- [ ] Audit JSON ownership/model validation and least-squares conditioning with
  adversarial and numerical tests; use evidence to choose scoped repairs.

## P2 — Complete the dense user workflow

- [ ] Provide seed control and explicit loss/evaluation metrics; document which
  losses and activations the gradient implementation supports.
- [ ] Add mini-batches, validation, and checkpoint/config persistence with a small
  reproducible example and save/load prediction-equivalence tests.
- [ ] Validate the same train/save/load workflow from C++ and Python; document
  unsupported advanced layers in the binding.
- [ ] Assess move-aware matrix overloads (#94) only after profiling and tests
  establish ownership, numerical behavior, and a measured benefit.

## P2 — Distribution and existing experimental layers

- [ ] Add installed CMake config/targets and an external consumer smoke test;
  allow builds without Python or tests and verify staged Python installation.
- [ ] Set a real package contact (#128) and verify DEB/RPM/TGZ metadata and clean
  installation before claiming supported binary distributions.
- [ ] Update module docs for the implemented decoder, CNN/RNN/LSTM primitives,
  dense backpropagation, and current build/dependency requirements.
- [ ] Specify train/eval and dropout behavior, encoder-decoder serialization, and
  one sequence training path; keep forward-only components marked experimental.

## Delivered baseline (implementation present on master)

- [x] Dense forward/backpropagation, activations, and GA with early stopping.
- [x] Standalone CNN, RNN, and LSTM forward primitives.
- [x] Transformer encoder and encoder-decoder forward components.
- [x] Python dense-network bindings and smoke-test source.
- [x] MNIST loading, logging, allocation/OpenMP improvements.
- [x] Install and release-package infrastructure (validation work remains above).

## Deferred

GPU/distributed execution, additional model families, and broad framework parity.
Revisit after M1/M2 acceptance evidence, not simply after more code is added.
