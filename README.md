# Failure Propagation and Recovery in Multi-Agent ALFWorld

CSE 291A, Fall 2026, Option 5. Team: Ho-Yen Wu, Jui-Yen Wang, Meng-Chi Tsai.

**Question:** can tracking dependencies between agent results support efficient recovery while
preserving correct and safe task execution?

**Status:** Phase 1 (fault injection and propagation measurement) is complete. Phase 2 (selective
recovery) is designed but not implemented.

**Start here:** the [Phase 1 report](docs/report/phase1.pdf) (6 pages). A longer writeup is in
[docs/PHASE1.md](docs/PHASE1.md).

## Results

Result tables: [phase1-tables.md](results/phase1-tables.md), [diversity.md](results/diversity.md),
[restore.md](results/restore.md). Each experiment folder under [results/](results/) has its per-episode
CSV and fault schedule, and [results/examples/](results/examples/) has one clean and one faulty trace.

| Experiment | Folder | Episodes |
| --- | --- | --- |
| Clean baseline (18 unseen games x 3 seeds) | `team-clean` | 54 |
| Clean with always-verify | `team-verify-clean` | 54 |
| Executed full restart (live calls) | `team-restart` | 18 |
| Faults, seed 42, early/mid/late | `team-faults` | 1,164 |
| Faults, seeds 43 and 44, mid | `team-faults-s43`, `team-faults-s44` | 805 |
| Always-verify, same seed-42 mid faults | `team-verify-faults` | 391 |
| Fuzzer versus random pilot | `fuzz-pilot` | 192 |

`eval-*` holds the earlier single-LLM comparison, and `pilot-*` holds development runs.

Main findings (seed 42, 1,162 fired faults):

| | Accidental (856) | Adversarial (306) |
| --- | --- | --- |
| Masked | 5.6% | 0.3% |
| Reasoning only | 60.2% | 17.0% |
| Actions changed | 33.5% | 77.1% |
| Task outcome changed | 0.7% | 5.6% |
| Episodes with unauthorized actions | 3 | 163 |
| Episodes with irreversible actions | 0 | 24 |

- Clean task success is 98.1% (53/54).
- Full restart costs 1.4-2.9x a clean episode.
- Always-verify blocked all unauthorized actions but lost 27 of 389 tasks, against 11 without it.
- Only 61% of structurally dependent calls change after a fault, and 47% of faults change the world
  state.
- In a 96-episode pilot, the fuzzer tied random injection on overall diversity (37 vs. 36 signatures) but
  found more severe behaviors (11 vs. 8 severe signatures). This is one campaign per arm, so it is
  suggestive only.

Section 5 of the report covers the state restore design, and Section 7 the limitations.

## Setup

Tested on macOS (Apple Silicon, CPU only) with Python 3.10.22.

```bash
git clone https://github.com/sunnyhywu/CSE291A-AFL_agent_recovery.git
cd CSE291A-AFL_agent_recovery
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt -r requirements-report.txt
alfworld-download
export ALFWORLD_DATA="$HOME/.cache/alfworld"
```

To replay episodes or inspect all traces, download the
[results archive](https://drive.google.com/file/d/1veQ9ZARgQkYjpyBHWZhACtSCiFUa6nEH/view?usp=sharing)
(77 MB) and unpack it at the repository root with `tar -xzf CSE291A-phase1-results.tar.gz`. It restores
`results/` and `.llm_cache/`. It leaves out 8 rate-limit failure files from an early development pilot,
because their error messages include a hash of our API key.

## Reproducing the results

**Without the archive or an API key:**

```bash
python -m unittest discover -s tests
PYTHONHASHSEED=0 python -m agent_recovery --trace demo-oracle.jsonl   # expert planner, no LLM
python verify_trace.py demo-oracle.jsonl
```

**With the archive, no API key needed.** Model responses come from the stored cache:

```bash
# Replay a clean run and the same game with a prompt injection into shared memory.
PYTHONHASHSEED=0 python -m agent_recovery --agents configs/llm_team.yaml --split valid_unseen \
  --task-types 1 --game-index 0 --seed 42 --max-steps 50 --trace demo-clean.jsonl
PYTHONHASHSEED=0 python -m agent_recovery --agents configs/llm_team.yaml --split valid_unseen \
  --task-types 1 --game-index 0 --seed 42 --max-steps 50 --trace demo-fault.jsonl \
  --fault '{"step": 1, "role": "memory", "operator": "malicious", "seed": 2105765495}'

# Regenerate all tables, then the figures and PDF (the PDF needs pdflatex).
experiments/phase1.sh
docs/report/build.sh
```

`phase1.sh` skips episodes whose traces exist, so with the archive it only recomputes the analysis.

**From scratch.** Put a TritonAI key in `.env` (`cp .env.example .env`) and run `experiments/phase1.sh`
without the archive. This took us about 11 hours and $11, and it is resumable. Live model calls are not
deterministic even at temperature 0, so expect similar overall results rather than identical traces.

## Code

| Path | Contents |
| --- | --- |
| [agent_recovery/](agent_recovery/) | Workflow, LLM agents, fault injection, dependency trace, action policy |
| [configs/](configs/) | `llm_team.yaml` is the evaluated workflow; `llm_team_verify.yaml` is always-verify |
| [experiments/](experiments/) | `phase1.sh` runs everything; `run_clean.py`, `run_faults.py` and `fuzz.py` are the runners |
| [analysis/](analysis/) | Clean-versus-faulty comparison, tables, figures, diversity and restore costs |
| [docs/report/](docs/report/) | Report source, figures and PDF |
| [tests/](tests/), [verify_trace.py](verify_trace.py) | Unit tests and the trace validator |

The model is `api-glm-5.3-flash` on UCSD TritonAI at temperature 0. Pinned packages are in
[requirements.lock.txt](requirements.lock.txt).
