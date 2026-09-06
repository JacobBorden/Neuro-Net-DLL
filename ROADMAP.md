# NeuroNet roadmap

Updated 2026-09-06; grounded in `master` at `01e0093`.
See [VISION.md](VISION.md) for scope and [TODO.MD](TODO.MD) for actionable work.
Milestones are ordered by dependency and exit criteria, not calendar promises.

| Milestone | Outcome | Exit criteria |
| --- | --- | --- |
| M0 — Integration health | One current baseline and actionable PR queue | PR #45 has compatible per-call generation control and passing checks; duplicate matrix PRs #214/#215 are reconciled against master; real matrix-vector regression #106 has a tested replacement; stale-generator issue #197 is addressed or has an explicit external owner/blocker. |
| M1 — Reliable core (0.2 candidate) | Clean builds and mathematically credible primitives | Explicit C++17 and optional OpenMP configuration; Linux and Windows build/test coverage; Python smoke test appears in CTest; gradient/reference tests for dense training, masks, and recurrent/convolution layers; deterministic GA invariant tests; matrix benchmarks include square and narrow products. |
| M2 — Reproducible dense workflow | Train, evaluate, save, and reuse a small model | Seed control; explicit loss/metric contract; mini-batches and validation; checkpoint/config persistence; C++ and Python examples with loss improvement and save/load prediction equivalence. Add optimizer choices only after gradient checks pass. |
| M3 — Supported experimental sequences | Existing sequence components have honest, testable contracts | Document state/reset and tensor shapes; numerical causal/cross-attention tests; real train/eval/dropout semantics; encoder-decoder persistence; select and demonstrate one sequence training path before expanding APIs. |
| M4 — Consumable release | A fresh machine can use the library | Optional Python/test dependencies; installed CMake package and external consumer smoke test; Python import/training test from staged install; validated package metadata and artifacts; versioned API/model-format compatibility notes. |

M4 build/install foundations can proceed alongside M1; publishing is gated on M1
and M2. Sequence training in M3 must not delay a reliable dense-only release.

## Already delivered, with limits

Dense backpropagation and GA optimization, activations, standalone CNN/RNN/LSTM
forward layers, encoder-decoder forward components, Python dense bindings, MNIST
loaders, logging, and CPack/release workflows exist. Their presence closes the old
feature checklist; it does not close the reliability gates above. Transformer
dropout and sequence training are not complete.

## Integration policy

`master` is the canonical integration branch (also the GitHub default as of this
review). `development` is historical and lacks substantial implemented features.
New PRs should target `master`; do not merge or retarget stale branches wholesale.
Inspect their intended change against current master and preserve only useful work.
There is no automatic promotion lane between branches.

A PR is ready after its current head passes relevant tests and visible CI, review
findings are addressed, and its diff adds something not already present. Performance
changes require correctness checks plus comparable measurements across affected
shapes. Do not substitute repeated reruns for a deterministic test repair.
Do not merge a PR merely because its checks are green.

## Maintenance cadence

- Existing PR steward: weekdays at 14:30 America/Halifax. Inspect changes since the
  last run, repair scoped blockers, and preserve merge gates and user work.
- Weekly health and roadmap review: Monday at 09:00 America/Halifax. Verify changed
  code, build/test registration, dependencies, packaging, and milestone evidence;
  maintain one focused follow-up at a time, avoiding duplicate PRs/issues.

Maintenance must remember head SHAs and previous outcomes, notify only on meaningful
changes or actionable blockers, and avoid repeatedly reporting unchanged issues.
External bot configuration such as the generator in #197 must be verified separately;
changing this repository does not disable that service.
