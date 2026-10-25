# Fuzzing-Guided Failure Discovery and Selective Recovery in Multi-Agent ALFWorld

This project studies how faulty intermediate artifacts propagate into downstream
reasoning, actions, and environment state, and how recovery can preserve valid work.
Structural dependency identifies potentially affected work; it does not establish
causal influence. Reasoning repair and world-state repair are separate problems.

## Milestone 1: traceable baseline

The core is plain Python, CPU-only, one text-environment mission, no API calls.
The implementation deliberately uses ALFWorld's handcoded expert as an **oracle
control**. State and memory agents are deterministic transformations, and the
planner selects the expert's first command. This establishes execution and tracing;
it does not yet establish an independent reasoning agent or meaningful propagation
experiments. In particular, the planner declares state and memory inputs but its
choice is driven by the oracle. Replacing this policy is required before using
fault injection to evaluate the research hypotheses.

```text
ALFWorld observation -> StateAgent -> MemoryAgent
                           |              |
                           +--> OraclePlanner <-- explicit expert-hint artifact
                                      |
                              deterministic executor
                                      |
                                  ALFWorld
```

Each next observation depends on both the executed action and the previous
observation, retaining the world-history path. These are conservative provenance
edges, not claims of causal influence or complete world-state snapshots.

## Files

```text
baseline.py              artifact/event models, DAG, agent interfaces, adapter and loop
verify_trace.py          standalone successful-trace validator
configs/alfworld.yaml    upstream base configuration; runner overrides text/CPU settings
requirements.txt         direct dependencies
requirements.lock.txt    exact installed environment for the verified run
tests/test_trace.py     branching DAG and serialization checks
final-001.jsonl          locally generated successful trace (ignored by version control)
```

`Artifact` has ID, kind, producer, input IDs, environment step, and JSON content.
`ExecutionEvent` has sequence, kind, step, artifact references, elapsed time, and
details. Artifact IDs increase within one trace; the trace file identifies the run.
The DAG uses an artifact dictionary and parent-to-child adjacency sets. Parents
must exist before children; `descendants(id)` returns structural descendants.
The `Agent` protocol maps input artifacts to an output artifact through the trace.
No graph database or orchestration framework is needed.

The loop resets one fixed game, records observation and explicit oracle input,
runs state/memory/planner, records the chosen action, executes it, and records the
result. It stops on environment termination, success, or the step limit. A timeout
or unsuccessful terminal state returns exit code 1; an exception records `run_error`
once tracing has started. Setup errors before trace creation surface normally.

## Run and verify

Tested with Python 3.10 on Linux. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
alfworld-download
export ALFWORLD_DATA="$HOME/.cache/alfworld"
python -m unittest discover -s tests -v
PYTHONHASHSEED=0 python baseline.py --seed 42 --trace run-001.jsonl
python verify_trace.py run-001.jsonl
```

The download command uses the upstream default dataset download, which also
includes assets beyond those used by this text-only runner. No visual runtime is
installed. For a custom download directory, pass `alfworld-download --data-dir PATH`
and set `ALFWORLD_DATA` to that same absolute directory.

Trace filenames must be new: existing results are never overwritten. The runner
sorts eligible training pick-and-place game paths and selects the first game.
Use `--gamefile /absolute/path/to/game.tw-pddl` to select an eligible game explicitly,
and `--max-steps 150` to set the action budget. Set `PYTHONHASHSEED=0` before launching Python: the handcoded expert can otherwise
break object-selection ties differently across processes despite a fixed environment seed.
The adapter also seeds Python and NumPy and uses a synchronous TextWorld batch
so the expert shares the seeded process.
The resolved game path, effective
configuration, seed, policy label, and ALFWorld version are recorded at run start.
This is a training-split infrastructure check, not held-out evaluation.

## Logging and completion criteria

JSONL records contain every artifact's content and parents, complete action and
observation history, admissible commands, expert hints, transition score/done/won,
configuration and seed, timing, and run termination or runtime error. Each record
is flushed immediately. World-state read/write sets and token accounting are not
available in this baseline. Elapsed event time is cumulative since trace creation,
not a per-agent latency measurement. Memory stores observation history verbatim;
this is intentionally small-scale and can grow quadratically across a long trace.

The milestone's oracle-control baseline is complete when:

1. A real ALFWorld mission returns environment `won=True` within the step budget.
2. Every action traces through a plan to explicit inputs; all parents are present.
3. Each action has a logged transition and resulting observation.
4. The independent validator and provenance test pass.
5. Repeating the same game/config/seed yields the same actions and observations
   (timing fields are excluded).

The first local run completed in **5 actions**, with environment `won=True`.
See `final-001.jsonl` and `final-002.jsonl`; both pass validation and have identical
artifacts, including actions and observations. Rerun the validator to check the evidence.

## Next milestones and experimental cautions

Core future work: replace the oracle planner with a stable observation/memory-based
policy, then controlled mutation, paired corrected-input runs, recovery baselines,
and reset-and-replay. Coverage-guided mutation and more complex branching are later
extensions. None is implemented here.

Before experiments, define output equivalence separately for facts, plans, and
actions, freeze model settings and task seeds, and compare paired runs. Replaying
actions alone does not prove that a world state was restored: verify observations
and terminal outcomes. Do not count expert-driven success as LLM-agent performance,
or descendants as empirically influenced outputs.

Configuration and adapter API follow the official project:
https://github.com/alfworld/alfworld
The vendored configuration originates from `configs/base_config.yaml` in that
repository; the installed package and lock file pin the executed API version.
