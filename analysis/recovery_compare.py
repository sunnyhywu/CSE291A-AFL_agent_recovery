"""Selective recovery against the Phase 1 baselines, under one cost accounting.

    python -m analysis.recovery_compare --faults results/team-faults/comparison.csv \
        --verify results/team-verify-faults/comparison.csv \
        --cached results/recovery-mid/comparison.csv \
        --live results/recovery-live-mid/comparison.csv --out results/recovery-comparison.md

Accounting (Phase 1, analysis/compare.py): the tokens spent from the start of the faulty
episode until the task is finished, relative to the clean episode. A recovery that starts when
the faulty episode ends pays for the whole faulty episode first:

* No recovery: the faulty episode.
* Always verify: the faulty episode of the verifying workflow (detection built in, no repair).
* Full restart, detected at end: the faulty episode, then a clean episode.
* Full restart, detected immediately: the work before the fault, then a clean episode.
* Selective recovery (detected at end, as run): the faulty episode, then the recomputed calls.
  Reused calls cost nothing; audit recomputation is an evaluation check and is not counted.

Every method is evaluated on the identical seed-42 mid-position fault schedule. Restart success
is the executed live restart rate from Phase 1 (18/18). 95% CIs resample games.
"""
import argparse
from pathlib import Path

from analysis.figures import bootstrap
from analysis.report import read

CATEGORIES = (('all', 'All faults'), ('accidental', 'Accidental'), ('adversarial', 'Adversarial'))


def methods(cached: dict, live: dict):
    """(name, tokens(b, v, e), won(b, v, e) or None, unauthorized executed(b, v, e))."""
    def recovery(rows, mode):
        return lambda b, v, e: b['faulty_tokens'] + rows[e, mode]['recomputed_tokens']

    def won(rows, mode):
        return lambda b, v, e: rows[e, mode]['won']

    def unauthorized(rows, mode):
        # Detected at the end: the faulty run's actions were already executed in the world.
        return lambda b, v, e: b['unauthorized_actions'] + rows[e, mode]['recovered_unauthorized_actions']

    return (
        ('No recovery', lambda b, v, e: b['faulty_tokens'], lambda b, v, e: b['faulty_won'],
         lambda b, v, e: b['unauthorized_actions']),
        ('Always verify', lambda b, v, e: v['faulty_tokens'], lambda b, v, e: v['faulty_won'],
         lambda b, v, e: v['unauthorized_actions']),
        ('Full restart, detected immediately', lambda b, v, e: b['restart_tokens_immediate'], None,
         lambda b, v, e: 0),
        ('Full restart, detected at end', lambda b, v, e: b['restart_tokens_at_end'], None,
         lambda b, v, e: b['unauthorized_actions']),
        ('Selective, invalidate (live)', recovery(live, 'invalidate'), won(live, 'invalidate'),
         unauthorized(live, 'invalidate')),
        ('Selective, cutoff (live)', recovery(live, 'cutoff'), won(live, 'cutoff'),
         unauthorized(live, 'cutoff')),
        ('Selective, invalidate (cached)', recovery(cached, 'invalidate'), won(cached, 'invalidate'),
         unauthorized(cached, 'invalidate')),
        ('Selective, cutoff (cached)', recovery(cached, 'cutoff'), won(cached, 'cutoff'),
         unauthorized(cached, 'cutoff')),
    )


def recovery_only(rows: dict, mode: str):
    """Work after detection only: recomputed tokens against a restart's clean episode."""
    return lambda r: rows[r['episode'], mode]['recomputed_tokens'] / r['b']['clean_tokens']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--faults', type=Path, default=Path('results/team-faults/comparison.csv'))
    parser.add_argument('--verify', type=Path, default=Path('results/team-verify-faults/comparison.csv'))
    parser.add_argument('--cached', type=Path, default=Path('results/recovery-mid/comparison.csv'))
    parser.add_argument('--live', type=Path, default=Path('results/recovery-live-mid/comparison.csv'))
    parser.add_argument('--out', type=Path, default=Path('results/recovery-comparison.md'))
    args = parser.parse_args()

    baseline = {r['episode']: r for r in read(args.faults)}
    verify = {r['episode']: r for r in read(args.verify)}
    cached = {(r['episode'], r['mode']): r for r in read(args.cached) if r['world'] == 'reset'}
    live = {(r['episode'], r['mode']): r for r in read(args.live) if r['world'] == 'reset'}
    episodes = sorted({e for e, _ in live})
    missing = [e for e in episodes if e not in verify or not verify[e]['faulty_complete']
               or any((e, m) not in cached for m in ('cutoff', 'invalidate'))]
    if missing:
        raise SystemExit(f'{len(missing)} scheduled episodes lack a paired baseline: {missing[:3]}')
    rows = [{'episode': e, 'b': baseline[e], 'v': verify[e]} for e in episodes]

    lines = [
        '# Selective recovery against the Phase 1 baselines',
        '',
        f'Identical schedule: {len(rows)} fired seed-42 mid-position faults '
        f'({len({r["episode"].split("-s42")[0] for r in rows})} games). '
        'Tokens from the start of the faulty episode until the task is finished, relative to the '
        'clean episode (mean [95% CI over games]). Generated by `analysis/recovery_compare.py`.',
        '',
    ]
    for category, title in CATEGORIES:
        subset = [r for r in rows if category == 'all' or r['b']['category'] == category]
        lines += [f'## {title} (n={len(subset)})', '',
                  '| Method | Tokens / clean | Won | Unauthorized actions executed |',
                  '|---|---|---|---|']
        for name, tokens, won, unauthorized in methods(cached, live):
            mean, low, high = bootstrap(
                subset, lambda r: tokens(r['b'], r['v'], r['episode']) / r['b']['clean_tokens'])
            share = (f'{sum(bool(won(r["b"], r["v"], r["episode"])) for r in subset)}/{len(subset)}'
                     if won else '18/18 †')
            count = sum(unauthorized(r['b'], r['v'], r['episode']) for r in subset)
            lines.append(f'| {name} | {mean:.2f}× [{low:.2f}, {high:.2f}] | {share} | {count:g} |')
        lines.append('')

    lines += ['## Work after detection only (detected at end)', '',
              'Tokens spent once the fault is known, relative to the clean episode a full restart '
              'reruns (1.00×).', '',
              '| Fault stage | n | Restart | Invalidate (live) | Cutoff (live) |', '|---|---|---|---|---|']
    for stage in ('all', 'environment', 'observation', 'memory', 'planner'):
        subset = [r for r in rows if stage == 'all' or r['b']['role'] == stage]
        cells = []
        for mode in ('invalidate', 'cutoff'):
            mean, low, high = bootstrap(subset, recovery_only(live, mode))
            cells.append(f'{mean:.2f}× [{low:.2f}, {high:.2f}]')
        lines.append(f'| {stage} | {len(subset)} | 1.00× | {cells[0]} | {cells[1]} |')
    lines += [
        '',
        '## Notes',
        '',
        '* † Restart success is Phase 1\'s executed live restart (18/18 seed-42 games won).',
        '* "Unauthorized actions executed" counts actions taken in the world over the whole '
        'process. A recovery that starts at episode end cannot prevent the faulty run\'s actions; '
        'it only restores the simulated world afterwards (reset). Recovered runs themselves took '
        'none.',
        '* Selective recovery was run only with detection at episode end. Detecting the fault at '
        'its step would leave nothing to reuse: rolling back would cost about one clean episode, '
        'against a restart\'s pre-fault work plus one clean episode.',
        '* Live recovery made every recomputed call through the API in a fresh cache namespace per '
        'recovery; 75 of 778 recoveries hit API timeouts and were rerun. Always verify and the '
        'fault runs are Phase 1\'s cached runs.',
        '* Audit recomputation (checking reused outputs) is not counted: it is an evaluation '
        'check, not part of recovery.',
    ]
    text = '\n'.join(lines) + '\n'
    args.out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
