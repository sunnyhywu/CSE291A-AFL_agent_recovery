# Failure diversity

A behavior signature is the propagation path (roles in the order the fault first changed them, `environment` = world trajectory diverged), outcome class, spread (changed calls: none, 1-3, 4-15, 16+), delay to the first changed action (none, now, short 1-4 steps, latent 5+), and unauthorized/irreversible flags.

## Systematic grid (seed 42, early/mid/late)

- 1162 fired faults, **92 distinct signatures**, 27 seen once; 25 distinct propagation paths.
- The 10 most common signatures cover 56% of faults.
- With the seed-43/44 repeats: 114 distinct signatures.

| Stage | Fired | Distinct signatures | Distinct paths |
|---|---:|---:|---:|
| environment | 302 | 27 | 8 |
| observation | 353 | 31 | 9 |
| memory | 354 | 25 | 4 |
| planner | 153 | 9 | 4 |

| Most common signatures | Faults |
|---|---:|
| `memory|reasoning_only|spread:1-3|delay:none` | 106 |
| `observation>memory|reasoning_only|spread:1-3|delay:none` | 92 |
| `planner>verifier|reasoning_only|spread:1-3|delay:none` | 86 |
| `environment>observation>memory|reasoning_only|spread:1-3|delay:none` | 68 |
| `environment>observation>memory|reasoning_only|spread:4-15|delay:none` | 67 |
| `observation>memory|reasoning_only|spread:4-15|delay:none` | 66 |
| `memory|reasoning_only|spread:4-15|delay:none` | 51 |
| `environment>observation>memory>planner>constraints>verifier|actions_changed|spread:16+|delay:now|unauthorized` | 40 |

Random subsamples of the grid (expected distinct signatures, 95% band):

| Faults | 12 | 24 | 48 | 96 | 200 | 400 | 800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Distinct | 9.8 [7, 12] | 16.5 [13, 20] | 25.3 [21, 30] | 36.3 [30, 43] | 50.3 [44, 57] | 65.2 [59, 71] | 82.2 [77, 87] |

## Fuzzer-versus-random pilot

Same fault space and budget, the same 12 initial random faults, 18 games.

| Arm | Episodes | Fired | Distinct signatures | Not in grid | Severe episodes | Distinct severe signatures | Outcome changed | Post-fault tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fuzzer (signature-guided) | 96 | 96 | 37 | 3 | 26 | 11 | 3 | 2,698,885 |
| Random injection | 96 | 96 | 36 | 1 | 16 | 8 | 3 | 2,003,253 |

| Fuzzer mutation | Proposed | New signature |
|---|---:|---:|
| step | 11 | 1 |
| operator | 19 | 12 |
| role | 19 | 6 |
| game | 20 | 7 |
| seed | 15 | 1 |
