"""Generate a fixed fault schedule from clean traces, run faulty episodes, and compare.

Only games whose clean run succeeded are injected, so failures are attributable to
the fault rather than to the model. The schedule is written before running, and every
method/rerun consumes the same file.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random
import re

from agent_recovery.injection import TARGETS, can_fire
from analysis.compare import compare, load
from experiments.run_clean import ROOT, episode

POSITIONS = {'early': lambda n: 1, 'mid': lambda n: max(1, n // 2), 'late': lambda n: max(1, n - 2)}


def clean_games(clean_dir: Path, clean_seed: int) -> dict:
    """Successful clean runs with this workflow seed, keyed by trace name: the injection targets."""
    games = {}
    for trace in sorted(clean_dir.glob('*.jsonl')):
        match = re.match(r'(\w+)-t(\d)-g(\d+)-s(\d+)\.jsonl', trace.name)
        if not match or int(match.group(4)) != clean_seed:
            continue
        clean = load(trace)
        if not clean['won']:
            continue
        split, task_type, index, run_seed = match.groups()
        games[trace.name] = {
            'split': split, 'task_type': int(task_type), 'game_index': int(index),
            'seed': int(run_seed), 'steps': len(clean['actions']),
            'texts': {a['step']: a['content']['text'] for a in clean['artifacts'].values()
                      if a['kind'] == 'observation'}}
    return games


def fire_step(game, role, operator, target):
    """First step at or after target where the fault changes its target; faults that cannot
    fire anywhere keep target and are reported unfired."""
    return next((s for s in range(target, game['steps']) if can_fire(role, operator, game['texts'][s])),
                target)


def schedule_item(name, game, position, role, operator, step, seed):
    return {'clean_trace': name, 'split': game['split'], 'task_type': game['task_type'],
            'game_index': game['game_index'], 'seed': game['seed'], 'position': position,
            'fault': {'step': step, 'role': role, 'operator': operator, 'seed': seed}}


def make_schedule(clean_dir: Path, positions, roles, seed: int, clean_seed: int):
    rng = random.Random(seed)
    schedule, seen = [], set()
    for name, game in clean_games(clean_dir, clean_seed).items():
        for position in positions:
            target = POSITIONS[position](game['steps'])
            for role in roles:
                for operator in sorted(TARGETS[role]):
                    step = fire_step(game, role, operator, target)
                    if (name, step, role, operator) in seen:
                        continue  # Short episodes map several positions to one step.
                    seen.add((name, step, role, operator))
                    schedule.append(schedule_item(name, game, position, role, operator, step,
                                                  rng.randrange(2 ** 31)))
    return schedule


def run_item(out: Path, clean_dir: Path, agents: Path, item, max_steps: int, name=None) -> dict:
    """Run one scheduled faulty episode (skipped if its trace exists) and compare it."""
    directory = out / (name or fault_name(item))
    directory.mkdir(exist_ok=True)
    trace, status = episode(directory, agents, item['split'], item['task_type'],
                            item['game_index'], item['seed'], max_steps, 1800,
                            ('--fault', json.dumps(item['fault'])))
    return {'episode': trace.parent.name, 'position': item['position'], 'status': status,
            **compare(clean_dir / item['clean_trace'], trace)}


def write_rows(path: Path, rows):
    fields = sorted({key for row in rows for key in row}, key=lambda k: (k != 'episode', k))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fault_name(item):
    f = item['fault']
    return (f"{Path(item['clean_trace']).stem}-{f['role']}-{f['operator']}-st{f['step']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agents', type=Path, default=ROOT / 'configs/llm_planner.yaml')
    parser.add_argument('--positions', nargs='+', default=['mid'], choices=list(POSITIONS))
    parser.add_argument('--roles', nargs='+', default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument('--schedule-seed', type=int, default=2026)
    parser.add_argument('--clean-seed', type=int, default=42,
                        help='Inject only into clean runs with this workflow seed')
    parser.add_argument('--limit-games', type=int)
    parser.add_argument('--schedule', type=Path,
                        help='Reuse another experiment\'s schedule (filtered by --positions), so '
                             'methods are compared under identical faults')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    schedule_path = args.out / 'schedule.jsonl'
    if schedule_path.exists():
        schedule = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    elif args.schedule:
        schedule = [item for item in map(json.loads, args.schedule.read_text().splitlines())
                    if item['position'] in args.positions]
        schedule_path.write_text(''.join(json.dumps(item) + '\n' for item in schedule))
    else:
        schedule = make_schedule(args.clean, args.positions, args.roles, args.schedule_seed,
                                 args.clean_seed)
        if args.limit_games:
            keep = sorted({item['clean_trace'] for item in schedule})[:args.limit_games]
            schedule = [item for item in schedule if item['clean_trace'] in keep]
        schedule_path.write_text(''.join(json.dumps(item) + '\n' for item in schedule))
    print(f'{len(schedule)} faulty episodes in {schedule_path}', flush=True)

    def run(item):
        return run_item(args.out, args.clean, args.agents, item, args.max_steps)

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(run, schedule):
            rows.append(row)
            print(f"{row['status']:9s} {row.get('propagation', 'not_fired'):22s} {row['episode']}",
                  flush=True)
    write_rows(args.out / 'comparison.csv', rows)


if __name__ == '__main__':
    main()
