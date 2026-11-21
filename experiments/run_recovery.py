"""Selectively recover recorded faulty episodes and tabulate retained work, cost and safety.

Each recovery is given the identity of the faulty artifact (from the trace's fault event),
rolls back to its step and reruns the episode (agent_recovery.recovery.recover). Every
recovery runs as its own process, so a crash is recorded rather than aborting the batch, and
finished traces are skipped, so batches are resumable.

    # Pilot: 3 fired mid-position faults per (stage, category), two recovery modes.
    python -m experiments.run_recovery --faults results/team-faults --clean results/team-clean \
        --out results/recovery-pilot --per-stratum 3 --modes cutoff invalidate --worlds reset

The selection is written to <out>/schedule.jsonl before running and reused on resume.
Recovered traces go to <out>/<episode>/<mode>-<world>.jsonl and rows to <out>/comparison.csv.
Model calls go through the response cache, so tokens and latencies are logical (a cache hit
counts its original cost); `live_calls` and `live_tokens` count actual API usage.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_recovery.policy import classify  # noqa: E402
from agent_recovery.recovery import read_records, summarize  # noqa: E402
from experiments.run_faults import write_rows  # noqa: E402

CARRIED = ('category', 'role', 'operator', 'step', 'position', 'propagation', 'faulty_won',
           'first_divergent_action', 'unauthorized_actions', 'irreversible_actions',
           'clean_actions', 'clean_tokens', 'faulty_actions', 'faulty_tokens', 'pre_fault_tokens',
           'post_fault_tokens', 'restart_tokens_immediate', 'restart_tokens_at_end')


def select(faults: Path, positions, per_stratum: int, seed: int) -> list[dict]:
    """Fired faults, stratified by (stage, category): up to `per_stratum` each, from distinct
    games where possible, in a fixed pseudo-random order."""
    with (faults / 'comparison.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['fault_fired'] == 'True' and r['position'] in positions
                and r['status'] in ('won', 'stopped')]
    strata = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['episode']):
        strata[row['role'], row['category']].append(row)
    rng, chosen = random.Random(seed), []
    for key in sorted(strata):
        candidates = strata[key][:]
        rng.shuffle(candidates)
        games, picked = set(), []
        for row in candidates:  # distinct games first, then any
            if len(picked) < per_stratum and row['faulty_trace'] not in games:
                picked.append(row)
                games.add(row['faulty_trace'])
        picked += [r for r in candidates if r not in picked][:per_stratum - len(picked)]
        chosen += [{'episode': r['episode'], 'trace': r['faulty_trace'],
                    **{k: r[k] for k in CARRIED}} for r in picked]
    return chosen


def recover_one(args):
    """In-process: recover one faulty trace into a new trace (run by the batch as a process)."""
    from agent_recovery.environment import AlfworldEnvironment
    from agent_recovery.loading import load_team
    from agent_recovery.recovery import faulty_artifact, recover
    from agent_recovery.trace import Trace
    from agent_recovery.workflow import Workflow

    records = read_records(args.source)
    start = records[0]['details']
    described = start['environment']
    team, config = load_team(args.agents)
    environment = AlfworldEnvironment(
        ROOT / 'configs/alfworld.yaml', seed=start['seed'], max_steps=start['max_steps'],
        task_types=described['config']['env']['task_types'], oracle_hints=start['oracle_hints'],
        split=described['split'], game_index=described['game_index'])
    trace = Trace(args.trace)
    try:
        workflow = Workflow(team, trace, seed=start['seed'], max_steps=start['max_steps'],
                            max_plan_attempts=config['max_plan_attempts'],
                            oracle_hints=start['oracle_hints'])
        if args.world != 'reset':
            # Recovery starts after the faulty episode ended: the live world has executed all
            # of its recorded actions, and compensation starts from there.
            environment.reset()
            for command in (r['content']['command'] for r in records if r.get('kind') == 'action'):
                environment.step(command)
        won = recover(args.source, faulty_artifact(records), workflow, environment,
                      mode=args.mode, world=args.world, audit=args.audit, configuration=config)
        print(f'{"SUCCESS" if won else "STOPPED"}: {args.trace}')
        return 0 if won else 1
    finally:
        trace.close()
        environment.close()


def logged(text: str, label: str) -> str | None:
    return next((line.split(maxsplit=1)[1] for line in text.splitlines()
                 if line.startswith(f'{label}:') and len(line.split(maxsplit=1)) > 1), None)


def run(out: Path, faults: Path, clean: Path, agents: Path, item: dict, mode: str, world: str,
        audit: bool, timeout: float, namespace: str | None = None) -> dict:
    """Recover one scheduled episode in its own process (skipped if its trace exists).

    With `namespace`, model calls use a fresh cache namespace per (episode, mode, world), so
    the recovery makes live calls that no other recovery or recorded run can answer."""
    directory = out / item['episode']
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / f'{mode}-{world}.jsonl'
    log = trace.with_suffix('.log')
    if trace.exists():
        # Resumed batch: report the recorded outcome, process time and namespace from the log.
        text = log.read_text() if log.exists() else ''
        status = ('won' if 'SUCCESS:' in text else 'stopped' if 'STOPPED:' in text else 'error')
        seconds = logged(text, 'PROCESS_SECONDS')
        seconds = float(seconds) if seconds else None
        cache_namespace = logged(text, 'CACHE_NAMESPACE') or ''
    else:
        # Per episode as well: recoveries of different faults in the same game would otherwise
        # answer each other's requests once they return to the clean path.
        cache_namespace = f'{namespace}-{item["episode"]}-{mode}-{world}' if namespace else ''
        command = [sys.executable, '-m', 'experiments.run_recovery', 'one',
                   '--source', str(faults / item['episode'] / item['trace']), '--trace', str(trace),
                   '--agents', str(agents), '--mode', mode, '--world', world]
        if audit:
            command.append('--audit')
        env = {**os.environ, 'PYTHONHASHSEED': '0',
               'ALFWORLD_DATA': os.environ.get('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))}
        env.pop('LLM_CACHE_NAMESPACE', None)
        if cache_namespace:
            env['LLM_CACHE_NAMESPACE'] = cache_namespace
        started = time.monotonic()
        with log.open('w') as handle:
            print(f'CACHE_NAMESPACE: {cache_namespace}', file=handle, flush=True)
            try:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, timeout=timeout).returncode
                # An uncaught exception also exits with 1, so the outcome comes from the log.
                handle.flush()
                text = log.read_text()
                status = ('won' if code == 0 and 'SUCCESS:' in text
                          else 'stopped' if code == 1 and 'STOPPED:' in text else f'error({code})')
            except subprocess.TimeoutExpired:
                status = 'timeout'
            seconds = time.monotonic() - started
            # Whole-process time, including interpreter and ALFWorld start-up.
            print(f'PROCESS_SECONDS: {seconds:.2f}', file=handle)
    return {'episode': item['episode'], 'mode': mode, 'world': world, 'status': status,
            'process_seconds': seconds, 'cache_namespace': cache_namespace,
            **{k: item[k] for k in CARRIED},
            **measure(trace, clean / item['trace'])}


def actions_of(records) -> list[str]:
    return [r['content']['command'] for r in records if r.get('kind') == 'action']


def measure(trace: Path, clean: Path) -> dict:
    """Recovery outcome and cost from a recovered trace, compared with the clean run."""
    if not trace.exists():
        return {'recovered': False}
    records = read_records(trace)
    events = [r for r in records if r['record'] == 'event']
    if not any(e['kind'] == 'recovery_started' for e in events):
        return {'recovered': False}
    summary = summarize(trace)
    started = next(e for e in events if e['kind'] == 'recovery_started')
    calls = [e for e in events if e['kind'] == 'agent_call_end'
             and e['sequence'] > started['sequence'] and 'reused_from' not in e['details']]
    llm = [c['details']['metrics'] for c in calls if 'cache_hit' in c['details']['metrics']]
    live = [m for m in llm if not m['cache_hit']]
    audits = [e['details']['audit_metrics'] for e in events if e['kind'] == 'agent_call_end'
              and 'cache_hit' in e['details'].get('audit_metrics', {})]
    live_audits = [m for m in audits if not m['cache_hit']]
    goal = next(r['content']['goal'] for r in records if r.get('kind') == 'task')
    actions = actions_of(records)
    checks = [classify(a, goal) for a in actions]
    return {
        'recovered': True,
        'won': summary['won'],
        'complete_recovery': summary['complete_recovery'],
        'prefix_calls': summary['prefix_calls'],
        'reused_calls': summary['reused_calls'],
        'recomputed_calls': summary['recomputed_calls'],
        'recomputed_by_role': '|'.join(f'{k}:{v}' for k, v in
                                       sorted(summary['recomputed_by_role'].items())),
        'recomputed_llm_calls': len(llm),
        'recomputed_tokens': summary['recomputed_tokens'],
        'recomputed_llm_seconds': sum(m['llm_latency_seconds'] for m in llm),
        'live_calls': len(live),
        'live_tokens': sum(m['input_tokens'] + m['output_tokens'] for m in live),
        'audit_mismatches': summary['audit_mismatches'],
        'audited_llm_calls': len(audits),
        'audit_live_calls': len(live_audits),
        'audit_live_tokens': sum(m['input_tokens'] + m['output_tokens'] for m in live_audits),
        'world_actions': summary['world_actions'],
        'world_seconds': sum(r['seconds'] for r in summary['world_restores']),
        'world_mechanisms': '|'.join(r['mechanism'] for r in summary['world_restores']),
        'fallbacks': summary['fallbacks'],
        'restore_failed': summary['restore_failed'],
        'irreversible_effects': '|'.join(summary['irreversible_effects']),
        'recovered_actions': len(actions),
        'matches_clean_actions': actions == actions_of(read_records(clean)),
        'recovered_unauthorized_actions': sum(c['unauthorized'] for c in checks),
        'recovered_irreversible_actions': sum(c['irreversible'] for c in checks),
    }


def main():
    if sys.argv[1:2] == ['one']:
        parser = argparse.ArgumentParser(prog='run_recovery one')
        parser.add_argument('--source', type=Path, required=True)
        parser.add_argument('--trace', type=Path, required=True)
        parser.add_argument('--agents', type=Path, required=True)
        parser.add_argument('--mode', required=True)
        parser.add_argument('--world', required=True)
        parser.add_argument('--audit', action='store_true')
        raise SystemExit(recover_one(parser.parse_args(sys.argv[2:])))

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--faults', type=Path, required=True, help='A run_faults output directory')
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agents', type=Path, default=ROOT / 'configs/llm_team.yaml')
    parser.add_argument('--positions', nargs='+', default=['mid'])
    parser.add_argument('--per-stratum', type=int, help='Faults per (stage, category); default all')
    parser.add_argument('--selection-seed', type=int, default=2026)
    parser.add_argument('--modes', nargs='+', default=['cutoff', 'invalidate'],
                        choices=('cutoff', 'invalidate'))
    parser.add_argument('--worlds', nargs='+', default=['reset'],
                        choices=('reset', 'compensate', 'auto'))
    parser.add_argument('--no-audit', action='store_true',
                        help='Skip recomputing reused outputs to check that they remain valid')
    parser.add_argument('--namespace',
                        help='Make live model calls in a fresh cache namespace per recovery: '
                             '<name>-<episode>-<mode>-<world>')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    schedule_path = args.out / 'schedule.jsonl'
    if schedule_path.exists():
        schedule = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    else:
        schedule = select(args.faults, args.positions, args.per_stratum or 10 ** 9,
                          args.selection_seed)
        schedule_path.write_text(''.join(json.dumps(item) + '\n' for item in schedule))
    jobs = [(item, mode, world) for item in schedule for mode in args.modes for world in args.worlds]
    print(f'{len(schedule)} faulty episodes, {len(jobs)} recoveries -> {args.out}', flush=True)

    def job(spec):
        item, mode, world = spec
        return run(args.out, args.faults, args.clean, args.agents, item, mode, world,
                   not args.no_audit, args.timeout, args.namespace)

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(job, jobs):
            rows.append(row)
            print(f"{row['status']:9s} {row['mode']:10s} {row['world']:10s} "
                  f"won={row.get('won')} live={row.get('live_calls')} {row['episode']}", flush=True)
    write_rows(args.out / 'comparison.csv', rows)


if __name__ == '__main__':
    main()
