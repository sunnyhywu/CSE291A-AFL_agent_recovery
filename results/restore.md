# World-state restoration costs (from existing traces)

- Environment reset: median 1282 ms over 54 clean episodes; one environment step: median 62.0 ms (p95 88.2 ms) over 611 steps.
- 546 of 1162 fired faults (47%) changed executed actions and need world-state restoration; the rest need only agent-state rollback.
- Replay prefix (clean actions before the first divergent one): median 4, max 31 actions, i.e. about 1.53 s of reset-and-replay (max 3.20 s).
- Compensation (undo faulty actions newest first): median 6, max 47 inverse actions.

| Faulty suffix after divergence | Accidental | Adversarial | All | Share |
|---|---:|---:|---:|---:|
| observation only (nothing to undo) | 1 | 0 | 1 | 0% |
| compensable by inverse actions | 172 | 161 | 333 | 61% |
| irreversible action the clean run also takes | 81 | 60 | 141 | 26% |
| irreversible action the clean run never takes | 39 | 32 | 71 | 13% |
