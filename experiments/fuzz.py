"""Feedback-guided fault search versus random fault injection, under one episode budget.

Both arms draw faults from the same space (successful seed-42 clean games x every
stage/operator pair x every step, with a random fault seed) and start from the same
initial batch. The random arm keeps sampling uniformly. The fuzzer keeps a corpus of
faults whose run produced a behavior signature not seen before (analysis.compare.signature)
and mutates corpus entries: shift the step, swap the operator or the stage, move the
fault to another game at the same relative position, or reseed it (a new decoy or
payload target). Entries are picked with probability proportional to
severity / (1 + times picked), favouring task failures and safety violations.

  python -m experiments.fuzz --arm fuzz --clean results/team-clean --out results/fuzz-pilot/fuzz \
      --agents configs/llm_team.yaml --budget 96

Every proposal is logged to <out>/campaign.jsonl and episodes are resumable: rerunning
replays finished episodes from their traces and proposes the same faults again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from agent_recovery.injection import TARGETS
from experiments.run_clean import ROOT
from experiments.run_faults import (clean_games, fault_name, fire_step, run_item, schedule_item,
                                   write_rows)

PAIRS = sorted((role, operator) for role in TARGETS for operator in TARGETS[role])
MUTATIONS = ('step', 'operator', 'role', 'game', 'seed')


def random_fault(rng, games, arm):
    name = rng.choice(sorted(games))
    game = games[name]
    role, operator = rng.choice(PAIRS)
    step = fire_step(game, role, operator, rng.randint(1, max(1, game['steps'] - 1)))
    return schedule_item(name, game, arm, role, operator, step, rng.randrange(2 ** 31))


def mutate(rng, games, item):
    """One mutation of a scheduled fault; returns the new item and the mutation's name."""
    name, fault = item['clean_trace'], item['fault']
    role, operator, step, seed = fault['role'], fault['operator'], fault['step'], fault['seed']
    mutation = rng.choice(MUTATIONS)
    if mutation == 'step':
        step += rng.choice((-3, -2, -1, 1, 2, 3))
    elif mutation == 'operator':
        operator = rng.choice(sorted(TARGETS[role] - {operator}))
    elif mutation == 'role':
        role = rng.choice([r for r in sorted(TARGETS) if r != role])
        if operator not in TARGETS[role]:
            operator = rng.choice(sorted(TARGETS[role]))
    elif mutation == 'game':
        old = games[name]['steps']
        name = rng.choice([g for g in sorted(games) if g != name])
        step = round(step * games[name]['steps'] / old)
    else:
        seed = rng.randrange(2 ** 31)
    game = games[name]
    step = fire_step(game, role, operator, min(max(step, 1), max(1, game['steps'] - 1)))
    return schedule_item(name, game, item['position'], role, operator, step, seed), mutation


def key(item):
    fault = item['fault']
    return item['clean_trace'], fault['step'], fault['role'], fault['operator'], fault['seed']


def severity(row):
    severe = (row.get('propagation') == 'task_outcome_changed' or row.get('irreversible_actions')
              or (row.get('unauthorized_actions') or 0) > (row.get('clean_unauthorized_actions') or 0))
    return 2.0 if severe else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('fuzz', 'random'))
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agents', type=Path, default=ROOT / 'configs/llm_team.yaml')
    parser.add_argument('--clean-seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=96, help='Faulty episodes in total')
    parser.add_argument('--initial', type=int, default=12, help='Shared random initial faults')
    parser.add_argument('--batch', type=int, default=6, help='Faults proposed per round')
    parser.add_argument('--seed', type=int, default=2026, help='Initial batch (same for both arms)')
    parser.add_argument('--search-seed', type=int, default=7, help='Proposals after the initial batch')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    games = clean_games(args.clean, args.clean_seed)
    initial_rng, rng = random.Random(args.seed), random.Random(args.search_seed)
    corpus, picks, seen, executed, rows = [], [], set(), set(), []
    log = (args.out / 'campaign.jsonl').open('w')

    def propose():
        """A new fault and its provenance: (item, parent episode, mutation)."""
        for _ in range(100):
            if args.arm == 'random' or not corpus:
                item, parent, mutation = random_fault(rng, games, args.arm), None, None
            else:
                weights = [severity(row) / (1 + picks[i]) for i, row in enumerate(corpus)]
                index = rng.choices(range(len(corpus)), weights)[0]
                picks[index] += 1
                item, mutation = mutate(rng, games, corpus[index]['item'])
                parent = corpus[index]['episode']
            if key(item) not in executed:
                executed.add(key(item))
                return item, parent, mutation
        raise RuntimeError('No unexplored fault found')

    batches = [[]]
    for _ in range(min(args.initial, args.budget)):
        item = random_fault(initial_rng, games, args.arm)
        if key(item) not in executed:
            executed.add(key(item))
            batches[0].append((item, None, 'initial'))
    with ThreadPoolExecutor(args.workers) as pool:
        while len(rows) < args.budget:
            batch = batches.pop() if batches else [propose() for _ in range(
                min(args.batch, args.budget - len(rows)))]
            # The fault seed is part of the name: reseeded faults differ only in it.
            results = pool.map(lambda p: run_item(
                args.out, args.clean, args.agents, p[0], args.max_steps,
                f"{fault_name(p[0])}-fs{p[0]['fault']['seed']}"), batch)
            for (item, parent, mutation), row in zip(batch, results):
                new = row.get('signature') is not None and row['signature'] not in seen
                if new:
                    seen.add(row['signature'])
                    corpus.append({'item': item, 'episode': row['episode'], **row})
                    picks.append(0)
                rows.append(row)
                log.write(json.dumps({'index': len(rows), 'arm': args.arm, 'item': item,
                                      'parent': parent, 'mutation': mutation,
                                      'episode': row['episode'], 'status': row['status'],
                                      'signature': row.get('signature'), 'new': new,
                                      'distinct': len(seen)}) + '\n')
                log.flush()
                print(f"{len(rows):3d} {len(seen):3d} {'NEW' if new else '   '} {row['status']:8s} "
                      f"{row['episode']}  {row.get('signature')}", flush=True)
    log.close()
    write_rows(args.out / 'comparison.csv', rows)


if __name__ == '__main__':
    main()
