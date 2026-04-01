# Phasis Handoff

## Current Development Style
- Work incrementally and keep each patch testable.
- Preserve biological/scoring behavior unless the task explicitly requests a behavior change.
- Keep multiprocessing changes compatible with macOS spawn and Linux fork/forkserver execution.

## Current Cache Direction
- Phase I and Phase II stages use the centralized cache pattern in `phasis.cache`.
- Preserve legacy memfile compatibility sections where downstream stages still read them.

## Current Performance Focus
- The main recent bottlenecks have been Phase I cluster planning and cluster scoring on scaffold-heavy datasets.
- Prefer removing redundant file I/O and improving stage-level progress visibility before considering broader algorithmic changes.

## Typical Verification
- `python -m py_compile` on touched modules.
- Cold and warm reruns on a small dataset when practical.
- HPC log review for quiet parent-side bottlenecks and adaptive parallel recovery behavior.
