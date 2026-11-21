# Phase 2: Selective Recovery in Multi-Agent ALFWorld (progress, 2026-11-19)

**Status.** The recovery mechanism is implemented and tested (65 unit tests). It has been
evaluated on all 389 fired seed-42 mid-position faults, both with the response cache and with
live model calls, and compared with the Phase 1 baselines under one cost accounting. Still open:
a valid measurement of compensation (see §5), the extensions in §7, and the final report
(due Dec 10). Branch: `phase2-selective-recovery`.

**Headline.** Given the faulty artifact, live selective recovery won 389/389 episodes (cutoff)
and 387/389 (invalidate), with no unauthorized or irreversible actions in the recovered runs.
After detection, cutoff spends 0.48x of a clean episode, where a full restart spends 1.00x.

## 1. What was built

| Piece | File | What it does |
|---|---|---|
| Resume | `agent_recovery/workflow.py` (`StepState`, `resume`) | Rebuilds agent state at any recorded step and continues the episode |
| Selective reuse | `agent_recovery/recovery.py` (`SelectiveReuse`, `recover`) | Given the faulty artifact's id, rolls back to its step and reruns. A recorded output is reused when the call's inputs are identical; otherwise it is recomputed |
| Two modes | `recover(mode=...)` | `cutoff`: only calls whose inputs changed are recomputed. `invalidate`: every DAG descendant of the faulty artifact is recomputed |
| Audit | `recover(audit=True)` | Also recomputes each reused output and records whether it matches, the audit's own cost, and the differing output |
| World restore | `agent_recovery/world.py` (`RecordedWorld`) | Lazy: serves recorded observations until the recovered run takes a different action, then restores the real world: `reset` (reset + replay), `compensate` (inverse actions, newest first, then a fingerprint check), `auto` (compensate, else reset). Actions with no inverse (heat, cool, clean, slice) are reported as irreversible effects |
| Batch runner | `experiments/run_recovery.py` | Runs each recovery in its own process (resumable); `--namespace` gives every recovery a fresh cache namespace so all its model calls are live |
| Comparison | `analysis/recovery_compare.py` | Selective recovery against the Phase 1 baselines → `results/recovery-comparison.md` |

Recovery is always given the identity of the faulty artifact (taken from the trace's fault
event), as the project requirements ask, so recovery is evaluated separately from detection.
It starts after the faulty episode has ended.

## 2. Experiments

| Run | Episodes × modes × worlds | Model calls | Output |
|---|---|---|---|
| Pilot | 21 × {cutoff, invalidate} × {reset, auto} | cached | `results/recovery-pilot/` |
| Full, cached | 389 × {cutoff, invalidate} × {reset, auto} = 1,556 | cached | `results/recovery-mid/` |
| Live pilot | 21 × {cutoff, invalidate} × reset | live | `results/recovery-live-pilot/` |
| **Full, live** | 389 × {cutoff, invalidate} × reset = 778 | live, audit on | `results/recovery-live-mid/` |

The 389 episodes are every fired seed-42 mid-position fault whose faulty run completed, the same
schedule Phase 1 used for always-verify. The cached runs check the mechanism under deterministic
model behavior. The live runs check that it holds when the model is not deterministic.

## 3. Results

### 3.1 Against the Phase 1 baselines (all 389 faults)

Tokens from the start of the faulty episode until the task is finished, relative to the clean
episode. This reproduces Phase 1's numbers for the baselines exactly.

| Method | Tokens / clean | Won | Unauthorized actions executed |
|---|---|---|---|
| No recovery | 1.29x | 378/389 | 154 |
| Always verify | 1.68x | 362/389 | 0 |
| Full restart, detected immediately | 1.46x | 18/18 * | 0 |
| Full restart, detected at end | 2.29x | 18/18 * | 154 |
| **Selective, cutoff, detected at end (live)** | **1.76x** | **389/389** | 154 |
| Selective, invalidate, detected at end (live) | 1.83x | 387/389 | 154 |

\* Phase 1's executed live restart. With detection at the end, the faulty run's 154
unauthorized actions have already happened; the recovered runs took none. Confidence intervals
and the accidental/adversarial split are in `results/recovery-comparison.md`.

Work after detection only (live, relative to a restart's 1.00x):

| Fault stage | n | Invalidate | Cutoff |
|---|---|---|---|
| all | 389 | 0.54x | 0.48x |
| environment | 101 | 0.56x | 0.53x |
| observation | 118 | 0.56x | 0.50x |
| memory | 119 | 0.55x | 0.49x |
| planner | 51 | 0.48x | 0.26x |

The cached runs gave the same costs (cutoff 47%, invalidate 54% of a clean episode), so the
savings are not an artifact of the cache.

### 3.2 Live recovery details

| | Cutoff | Invalidate |
|---|---|---|
| Won | 389/389 | 387/389 |
| Same actions as the clean run | 337 | 330 |
| Reused calls | 2,999 | 509 |
| Median recovery time (process, incl. start-up and audit) | 23 s | 24 s |

Live cost: about 10.6M tokens, roughly $2.4. 75 of the 778 recoveries hit TritonAI timeouts or
connection errors; they were rerun, and their partial calls were served from their own cache
namespace.

### 3.3 Do reused outputs stay valid? (audit, cutoff)

Of 1,415 reused LLM outputs, 253 (18%) differed from a live recomputation:

| Difference | Count |
|---|---|
| Wording only (memory summary, planner rationale) | 153 |
| Facts (observation facts, memory beliefs) | 99 |
| Planner's chosen action | 1 |

Episodes whose reused outputs differed in facts left the clean actions about as often as those
with no differing output: 10/79 (13%) against 41/287 (14%). Divergence comes from the model's
own variability in the recomputed calls, not from reuse.

### 3.4 The two failures

Both are invalidate recoveries of game `t4-g001` ("put a hot apple in fridge"), with the fault at
step 4. Invalidate recomputed step 4, and the live planner opened the microwave instead of
going to the fridge. It then shuttled between fridge and microwave until the verifier stopped
the episode. Cutoff reused the recorded step-4 decision and won. The agents' state and the
world were restored correctly: this is the model failing the task, not recovery failing.

## 4. Findings

1. Selective recovery repairs what verification only detects. With perfect fault identity it
   restored every task and took no unsafe actions, for half of a restart's work after detection.
2. Cutoff saves most where faults are late in the pipeline. A planner fault needs a quarter of a
   restart's work. Faults in shared memory or observations propagate and need about half.
3. Reuse is safe here. Reused outputs that a live model would phrase differently did not change
   what the team did.
4. When the fault is detected matters more than how it is repaired. Detected at the end, the
   faulty run's tokens and actions are already spent, so a restart on immediate detection
   (1.46x) is cheaper than our recovery (1.76x). Detecting a fault at its step and rolling back
   would cost about 1.0x (estimated, not run).

## 5. Known issue: compensation was not measured correctly

In the cached `auto` runs, only 6/389 recoveries were restored by compensation alone (163 needed
no restore, 220 fell back to reset). This does **not** mean actions cannot be undone:

* 218 of the 220 fallbacks are episodes the faulty run had already finished (won). ALFWorld
  ignores every command after an episode ends, so the inverse actions had no effect and the next
  inverse was no longer admissible. Every failing inverse was admissible right after its
  original action.
* The 6 successes are exactly the episodes whose faulty run had not ended.
* The other 2 fallbacks are a bug: when the agent walked back and forth, the inverse of a
  `go to` can name the agent's current location, which ALFWorld does not offer as a command.

**Fix (free, cached):** start compensation before the episode's final action (detection just
before the end), skip a `go to` inverse that targets the current location, and rerun the 389
episodes with `--worlds auto compensate`. Phase 1 estimated that 61% of action-diverging faults
are fully compensable; this would test that.

## 6. Limitations

* Only seed 42, mid position; no repeated runs, so no variability is reported yet.
* Fault identity is given and exact. Wrong or late identification is untested.
* Live recovery used reset only; compensation has not been measured validly (§5).
* The bookkeeping cost of tracking (trace size, reuse lookups) has not been measured against
  the savings.

## 7. Next steps

- [ ] Fix and rerun the compensation measurement (§5)
- [ ] Stress the method: give a wrong or late faulty artifact, and record how recovery fails
- [ ] Detection + recovery: let a verifier rejection trigger selective recovery at that step;
      report false positives and missed faults separately
- [ ] Precision against bookkeeping: storage and lookup cost of restart, invalidate and cutoff
- [ ] Early and late fault positions (cached, free); repeat ~50 live recoveries 3 times
- [ ] Final report (Dec 10)

## 8. Reproduce

Needs the Phase 1 archive (`results/` and `.llm_cache/`), ALFWorld data and, for live runs,
TritonAI credentials in `.env`.

```bash
# Cached recovery of the 389 faults (no API cost)
python -m experiments.run_recovery --faults results/team-faults --clean results/team-clean \
    --out results/recovery-mid --modes cutoff invalidate --worlds reset auto

# Live recovery (~$2.4, a few hours at 3 workers); rerun the same command to resume
python -m experiments.run_recovery --faults results/team-faults --clean results/team-clean \
    --out results/recovery-live-mid --modes cutoff invalidate --worlds reset --namespace live-mid

# Comparison with the Phase 1 baselines
python -m analysis.recovery_compare
```

A resumed batch skips finished traces. Delete the trace and log of a recovery that ended in an
API error before rerunning it.
