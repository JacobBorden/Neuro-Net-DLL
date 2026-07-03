## 2024-07-03 - Genetic Algorithm Flaky Test
**Learning:** Fixing flaky stochastic tests (like in Genetic Algorithms) by hardcoding fixed random seeds (e.g., `random_engine_.seed(42)`) directly into the production source code breaks production randomness.
**Action:** Always handle seed configuration via dependency injection or test-specific configuration methods, rather than modifying production source code, to fix test flakiness.
