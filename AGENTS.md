# Phasis Development Guide

## Core Rules
- No nested functions.
- No imports inside functions, except existing Phasis-package imports required for spawn safety, circular imports, or bootstrap flow.
- Keep the code working after every small patch.
- Prefer one small, testable functional change per commit.
- Preserve behavior unless a change is explicitly requested.

## Parallelism And Platform Safety
- Preserve macOS spawn safety and Linux fork/forkserver compatibility.
- Keep worker inputs lightweight and pass explicit paths or simple tuples to parallel workers.
- Prefer stage-owned implementations plus thin wrappers over new shared monoliths.

## Cache And Pipeline Patterns
- Use the centralized cache contract by default:
  - `MemCache.load(memFile)`
  - `stage_signature(...)`
  - `cache.hit(...)`
  - `cache.record(...)`
- Preserve compatibility with legacy memfile sections when downstream code still depends on them.
- Avoid broad refactors unless the task explicitly calls for them.

## Editing Style
- Default to conservative patches.
- Keep code ASCII unless the file already requires otherwise.
- Add brief comments only where the logic is non-obvious.
