## 2026-11-03 - Optimize A* Pathfinder
**Learning:** In `NeuralPathfinder::FindOptimalPathAStar` (`src/optimization/neural_pathfinder.cpp`), checking if a popped node's `f_score` is strictly worse than its best recorded score (`g_score.at(current_node) + CalculateHeuristic(...)`) prunes stale duplicate entries from `open_set`, avoiding redundant neighbor expansions in A* search.
**Action:** Always check and prune stale entries when using `std::priority_queue` in A* since C++ priority queues don't support updating priorities in-place.
