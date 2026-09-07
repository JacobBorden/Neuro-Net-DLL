## 2026-09-07 - [Optimize JSON Parsing in Matrix Deserialization]
**Learning:** Avoid redundant lookups in std::unordered_map by storing the iterator from find() and using it instead of calling find() and at() multiple times for the same key.
**Action:** Replaced double map lookups with iterator assignment and it->second member access.
