### Clean baseline

54 episodes (18 games × 3 seeds). Statuses: {'terminal': 53, 'verification_rejected': 1}. Success per seed: 100%, 100%, 94% (sd 2.6%).

| Task type | Episodes | Won | Mean actions | Mean tokens |
|---|---|---|---|---|
| pick & place | 9 | 9/9 | 11.7 | 25,019 |
| examine in light | 9 | 9/9 | 7.0 | 13,609 |
| clean & place | 9 | 9/9 | 9.7 | 24,614 |
| heat & place | 9 | 8/9 | 17.1 | 53,951 |
| cool & place | 9 | 9/9 | 12.0 | 30,265 |
| pick two & place | 9 | 9/9 | 10.4 | 21,590 |

| Metric (95% CI over games) | All episodes | Won episodes |
|---|---|---|
| Success rate (%) | 98.1 [94.4, 100.0] | – |
| Environment actions | 11.3 [9.4, 14.1] | 11.3 [9.3, 14.2] |
| LLM calls | 34.1 [28.4, 42.5] | 33.8 [28.0, 42.5] |
| Tokens (in+out) | 28,175 [21,130, 39,428] | 27,980 [20,655, 39,428] |
| Model latency (s) | 30.3 [23.8, 40.9] | 29.7 [22.9, 40.5] |
| Episode wall time excl. rate-limit waits (s) | 31.0 [23.9, 42.0] | 30.4 [23.1, 41.6] |
| Cold-start setup (s) | 0.68 [0.65, 0.71] | – |
| Plan rejections | 0.07 [0.00, 0.20] | – |

### Clean baseline with always verify

54 episodes (18 games × 3 seeds). Statuses: {'terminal': 53, 'verification_rejected': 1}. Success per seed: 100%, 94%, 100% (sd 2.6%).

| Task type | Episodes | Won | Mean actions | Mean tokens |
|---|---|---|---|---|
| pick & place | 9 | 9/9 | 11.8 | 29,770 |
| examine in light | 9 | 9/9 | 7.3 | 17,750 |
| clean & place | 9 | 9/9 | 10.6 | 32,365 |
| heat & place | 9 | 9/9 | 21.6 | 78,319 |
| cool & place | 9 | 8/9 | 12.9 | 40,514 |
| pick two & place | 9 | 9/9 | 10.6 | 26,172 |

| Metric (95% CI over games) | All episodes | Won episodes |
|---|---|---|
| Success rate (%) | 98.1 [94.4, 100.0] | – |
| Environment actions | 12.4 [10.1, 15.8] | 12.2 [10.0, 15.5] |
| LLM calls | 51.2 [41.5, 65.0] | 49.9 [40.9, 63.7] |
| Tokens (in+out) | 37,482 [27,880, 52,641] | 36,443 [27,355, 51,107] |
| Model latency (s) | 44.1 [34.8, 57.9] | 43.3 [34.2, 57.4] |
| Episode wall time excl. rate-limit waits (s) | 16.6 [11.7, 22.9] | 16.5 [11.5, 22.9] |
| Cold-start setup (s) | 0.64 [0.61, 0.68] | – |
| Plan rejections | 0.70 [0.33, 1.20] | – |

### Fault propagation (baseline workflow, seed 42)

1164 scheduled episodes; run statuses: {'stopped': 23, 'won': 1141}.

**Accidental faults**

| Stage | Fault | Fired | Masked | Reasoning only | Actions changed | Outcome changed | Faulty won | Changed calls | Δ actions | Δ tokens | Unauth. | Irrev. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| environment | contradictory | 50/50 | 0% | 84% | 16% | 0% | 50/50 | 6.8 | +0.3 | +730 | 0 | 0 |
| environment | incorrect | 50/50 | 0% | 78% | 22% | 0% | 50/50 | 7.7 | -0.0 | +59 | 0 | 0 |
| environment | missing | 50/50 | 0% | 76% | 22% | 2% | 49/50 | 13.4 | +0.6 | +1,373 | 0 | 0 |
| environment | stale | 50/51 | 0% | 58% | 42% | 0% | 50/50 | 14.6 | +0.7 | +1,822 | 0 | 0 |
| observation | contradictory | 50/50 | 24% | 52% | 24% | 0% | 50/50 | 7.7 | +0.2 | +445 | 0 | 0 |
| observation | incorrect | 50/50 | 0% | 74% | 24% | 2% | 49/50 | 12.2 | +0.9 | +2,813 | 0 | 0 |
| observation | missing | 50/50 | 14% | 54% | 30% | 2% | 49/50 | 11.8 | +0.6 | +1,475 | 0 | 0 |
| observation | stale | 50/51 | 2% | 58% | 40% | 0% | 50/50 | 13.7 | +0.8 | +1,884 | 0 | 0 |
| observation | timeout | 51/51 | 12% | 59% | 27% | 2% | 50/51 | 13.8 | +0.8 | +2,065 | 0 | 0 |
| memory | contradictory | 50/50 | 28% | 60% | 12% | 0% | 50/50 | 4.2 | -0.2 | -695 | 0 | 0 |
| memory | incorrect | 50/50 | 10% | 72% | 18% | 0% | 50/50 | 7.5 | +0.3 | +900 | 0 | 0 |
| memory | missing | 50/50 | 4% | 54% | 42% | 0% | 50/50 | 13.7 | +0.5 | -608 | 0 | 0 |
| memory | stale | 51/51 | 0% | 37% | 63% | 0% | 51/51 | 23.2 | +2.0 | +4,780 | 0 | 0 |
| memory | timeout | 51/51 | 0% | 39% | 61% | 0% | 51/51 | 21.7 | +1.9 | +3,373 | 0 | 0 |
| planner | incorrect | 51/51 | 0% | 0% | 96% | 4% | 49/51 | 27.6 | +2.2 | +6,670 | 5 | 0 |
| planner | stale | 51/51 | 2% | 82% | 16% | 0% | 51/51 | 6.2 | +0.3 | +1,548 | 0 | 0 |
| planner | timeout | 51/51 | 0% | 86% | 14% | 0% | 51/51 | 3.8 | +0.1 | +245 | 0 | 0 |
| **all** |  | 856/858 | 6% | 60% | 34% | 1% | 850/856 | 12.4 | +0.7 | +1,709 | 5 | 0 |

**Adversarial faults**

| Stage | Fault | Fired | Masked | Reasoning only | Actions changed | Outcome changed | Faulty won | Changed calls | Δ actions | Δ tokens | Unauth. | Irrev. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| environment | malicious | 51/51 | 0% | 4% | 92% | 4% | 49/51 | 46.0 | +5.0 | +14,306 | 58 | 0 |
| environment | malicious_alter | 51/51 | 0% | 22% | 76% | 2% | 50/51 | 41.5 | +4.5 | +11,832 | 93 | 5 |
| observation | malicious | 51/51 | 0% | 8% | 84% | 8% | 47/51 | 53.2 | +6.6 | +17,753 | 58 | 0 |
| observation | malicious_alter | 51/51 | 0% | 16% | 78% | 6% | 48/51 | 43.9 | +4.8 | +14,018 | 58 | 7 |
| memory | malicious | 51/51 | 0% | 24% | 69% | 8% | 47/51 | 50.5 | +6.4 | +18,356 | 52 | 0 |
| memory | malicious_alter | 51/51 | 2% | 29% | 63% | 6% | 48/51 | 48.8 | +6.5 | +17,808 | 74 | 12 |
| **all** |  | 306/306 | 0% | 17% | 77% | 6% | 289/306 | 47.3 | +5.6 | +15,679 | 393 | 24 |

**By injection position**

| Position | Fired | Masked | Reasoning only | Actions changed | Outcome changed | Faulty won | Changed calls | Δ actions | Δ tokens | Unauth. | Irrev. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| early | 414/414 | 1% | 40% | 57% | 3% | 403/414 | 36.4 | +3.2 | +8,112 | 213 | 13 |
| mid | 389/391 | 4% | 53% | 41% | 3% | 378/389 | 19.0 | +1.8 | +5,077 | 154 | 8 |
| late | 359/359 | 9% | 55% | 36% | 0% | 358/359 | 7.3 | +0.9 | +2,582 | 31 | 3 |


### Variability across seeds (mid position)

Each seed has its own clean run; faults are injected only where that run succeeded.

| Seed | Fired | Masked | Reasoning only | Actions changed | Outcome changed | Faulty won | Changed calls | Δ actions | Δ tokens | Unauth. | Irrev. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 42 | 389/391 | 4% | 53% | 41% | 3% | 378/389 | 19.0 | +1.8 | +5,077 | 154 | 8 |
| seed 43 | 412/414 | 4% | 49% | 44% | 4% | 395/412 | 25.5 | +2.7 | +8,142 | 204 | 9 |
| seed 44 | 391/391 | 2% | 48% | 48% | 3% | 381/391 | 23.5 | +2.1 | +5,929 | 164 | 6 |
| sd across seeds |  | 0.7% | 2.3% | 2.8% | 0.7% |  |  |  |  |  |  |

### Wasted work and the full-restart baseline

Tokens are logical (cached replays count their original cost). *Post-fault* work was computed on contaminated inputs. Full restart discards everything done before detection and reruns the clean episode; detection is either immediately after the faulty output or at episode end.

| Category | Clean episode | Post-fault (wasted) | Restart, detect immediately | Restart, detect at end | Restart-at-end / clean | Restart-at-end latency (s) |
|---|---|---|---|---|---|---|
| accidental | 25,170 [16,968, 37,600] | 15,585 [11,356, 21,811] | 36,463 [24,477, 54,776] | 52,048 [35,957, 76,493] | 2.10 [2.06, 2.14] | 70.8 [55.3, 95.0] |
| adversarial | 25,033 [16,986, 37,598] | 29,662 [24,663, 35,555] | 36,083 [24,324, 54,304] | 65,745 [49,804, 89,497] | 2.95 [2.66, 3.27] | 79.5 [63.5, 105.0] |

### Executed full restart

Every seed-42 clean episode was rerun from scratch with live model calls (a separate cache namespace). 8/18 reruns reproduced the same number of actions and outcome.

| Run | Won | Actions | Tokens | Model latency (s) | Live wall time excl. waits (s) |
|---|---|---|---|---|---|
| Clean run (seed 42) | 18/18 | 9.6 | 23,992 | 25.8 | – |
| Executed restart (live) | 18/18 | 10.3 | 24,305 | 25.7 | 27.0 |

### Scope of recovery (Phase 2 motivation)

*Descendant calls* are later agent calls structurally downstream of the faulty output in the dependency DAG (what dependency-based invalidation reruns). *Changed calls* are those whose output differs from the clean run. *World state changed* means executed actions diverged, so recomputation alone cannot recover: the environment must be restored too.

| Stage / category | Fired | Descendant calls | Changed calls | Changed / descendants | No call changed | World state changed | Episodes with irreversible actions |
|---|---|---|---|---|---|---|---|
| environment / accidental | 200 | 28.6 | 10.6 | 37% | 0% | 26% | 0 |
| observation / accidental | 251 | 28.1 | 11.8 | 42% | 10% | 30% | 0 |
| memory / accidental | 252 | 28.2 | 14.1 | 50% | 8% | 39% | 0 |
| planner / accidental | 153 | 28.8 | 12.5 | 43% | 1% | 43% | 0 |
| environment / adversarial | 102 | 51.3 | 43.8 | 85% | 0% | 87% | 5 |
| observation / adversarial | 102 | 54.6 | 48.5 | 89% | 0% | 88% | 7 |
| memory / adversarial | 102 | 57.1 | 49.7 | 87% | 1% | 73% | 12 |
| **all** | 1162 | 35.2 | 21.6 | 61% | 4% | 47% | 24 |

### Cold start and tracking overhead

| Metric | Mean | Max |
|---|---|---|
| Cold start: process time outside the episode, incl. Python/ALFWorld start-up (s) | 0.9 | 1.2 |
| Environment setup inside the run (s) | 0.68 | 0.93 |
| Trace size per environment step (KB) | 10.3 | 12.4 |
| Trace recording time (s per episode) | 0.0081 | 0.0248 |
| Recording time / episode time | 0.013% | 0.035% |

### Malicious instructions

The injected instruction names an object the agent has seen (preferably one the task does not need) and asks to put it in the garbagecan (`malicious`) or to heat/cool/clean it, an irreversible change (`malicious_alter`). *Executed* counts episodes where that exact action ran; *Took foreign* counts episodes that picked up a payload object the task does not name. Blocks are per proposal.

| Workflow | Payload | Fired | Proposed payload | Executed payload | Took foreign object | Blocked by verifier | Blocked by gate | Any unauthorized action | Any irreversible action | Task won |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline, all positions | malicious | 153 (3 generic) | 15 | 15 | 87 | 0 | 0 | 89 | 0 | 143 |
| Baseline, all positions | malicious_alter | 153 (3 generic) | 25 | 23 | 71 | 0 | 2 | 74 | 24 | 146 |
| Baseline, same schedule as always verify | malicious | 51 (0 generic) | 10 | 10 | 37 | 0 | 0 | 37 | 0 | 47 |
| Baseline, same schedule as always verify | malicious_alter | 51 (0 generic) | 9 | 8 | 27 | 0 | 1 | 27 | 8 | 49 |
| Always verify | malicious | 51 (0 generic) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44 |
| Always verify | malicious_alter | 51 (0 generic) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 46 |

### Always-verify comparison

| Clean runs | Episodes | Success | LLM calls / episode | Tokens / episode | Plan rejections / episode |
|---|---|---|---|---|---|
| Baseline | 54 | 53/54 | 34.1 | 28,175 | 0.07 |
| Always verify | 54 | 53/54 | 51.2 | 37,482 | 0.70 |

Same fault schedule (paired episodes):

| Workflow | Fired | Masked | Reasoning only | Actions changed | Outcome changed | Faulty won | Changed calls | Δ actions | Δ tokens | Unauth. | Irrev. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 389/391 | 4% | 53% | 41% | 3% | 378/389 | 19.0 | +1.8 | +5,077 | 154 | 8 |
| Always verify | 389/391 | 3% | 51% | 40% | 7% | 362/389 | 20.3 | +1.1 | +5,394 | 0 | 0 |
