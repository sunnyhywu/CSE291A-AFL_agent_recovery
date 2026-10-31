"""Failure diversity: distinct behavior signatures (analysis.compare.signature) found by the
systematic grid and by the fuzzer-versus-random pilot, as Markdown plus a discovery-curve figure.

  python -m analysis.diversity --grid results/team-faults/comparison.csv \
      --repeats results/team-faults-s43/comparison.csv results/team-faults-s44/comparison.csv \
      --pilot results/fuzz-pilot --out results/diversity.md --figure docs/report/figures/diversity.pdf

The random-subsample curve is the expected discovery of random injection drawn from the
grid's own 1,162 executed faults (mean and 95% band over resamples), so the pilot's random
arm can be checked against a much larger sample of the same kind.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analysis.figures import INK, MUTED, SECONDARY, STAGE_COLORS, tidy  # noqa: E402
from analysis.report import ROLES, read  # noqa: E402

ARMS = (('fuzz', 'Fuzzer (signature-guided)'), ('random', 'Random injection'))


def severe(row) -> bool:
    return bool(row['propagation'] == 'task_outcome_changed' or row['irreversible_actions']
                or row['unauthorized_actions'] > row['clean_unauthorized_actions'])


def curve(signatures):
    seen, counts = set(), []
    for sig in signatures:
        if sig:
            seen.add(sig)
        counts.append(len(seen))
    return counts


def subsample(signatures, sizes, rounds=500, seed=0):
    rng, result = random.Random(seed), {}
    for size in sizes:
        values = sorted(len({s for s in rng.sample(signatures, size) if s}) for _ in range(rounds))
        result[size] = (sum(values) / rounds, values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1])
    return result


def campaign(directory: Path):
    """Pilot rows in proposal order, joined with each episode's comparison row."""
    rows = {r['episode']: r for r in read(directory / 'comparison.csv')}
    log = [json.loads(line) for line in (directory / 'campaign.jsonl').read_text().splitlines()]
    return [{**rows[entry['episode']], 'mutation': entry['mutation']} for entry in log]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--repeats', type=Path, nargs='*', default=[])
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    grid = [r for r in read(args.grid) if r['fault_fired']]
    signatures = [r['signature'] for r in grid]
    counts = Counter(signatures)
    lines = ['# Failure diversity', '',
             'A behavior signature is the propagation path (roles in the order the fault first '
             'changed them, `environment` = world trajectory diverged), outcome class, spread '
             '(changed calls: none, 1-3, 4-15, 16+), delay to the first changed action (none, now, '
             'short 1-4 steps, latent 5+), and unauthorized/irreversible flags.', '',
             '## Systematic grid (seed 42, early/mid/late)', '',
             f'- {len(grid)} fired faults, **{len(counts)} distinct signatures**, '
             f'{sum(v == 1 for v in counts.values())} seen once; '
             f'{len({r["propagation_path"] for r in grid})} distinct propagation paths.',
             f'- The 10 most common signatures cover {sum(v for _, v in counts.most_common(10)) / len(grid):.0%} '
             'of faults.']
    union = set(counts)
    for path in args.repeats:
        union |= {r['signature'] for r in read(path) if r['fault_fired']}
    if args.repeats:
        lines.append(f'- With the seed-43/44 repeats: {len(union)} distinct signatures.')
    lines += ['', '| Stage | Fired | Distinct signatures | Distinct paths |', '|---|---:|---:|---:|']
    for role in ROLES:
        subset = [r for r in grid if r['role'] == role]
        lines.append(f'| {role} | {len(subset)} | {len({r["signature"] for r in subset})} | '
                     f'{len({r["propagation_path"] for r in subset})} |')
    lines += ['', '| Most common signatures | Faults |', '|---|---:|']
    lines += [f'| `{sig}` | {n} |' for sig, n in counts.most_common(8)]

    sizes = [n for n in (12, 24, 48, 96, 200, 400, 800) if n <= len(signatures)]
    expected = subsample(signatures, sizes)
    lines += ['', 'Random subsamples of the grid (expected distinct signatures, 95% band):', '',
              '| Faults | ' + ' | '.join(map(str, sizes)) + ' |', '|---|' + '---:|' * len(sizes),
              '| Distinct | ' + ' | '.join(f'{m:.1f} [{lo}, {hi}]' for m, lo, hi in expected.values()) + ' |']

    pilots = {}
    if args.pilot:
        pilots = {arm: campaign(args.pilot / arm) for arm, _ in ARMS if (args.pilot / arm / 'campaign.jsonl').exists()}
        lines += ['', '## Fuzzer-versus-random pilot', '',
                  'Same fault space and budget, the same 12 initial random faults, 18 games.', '',
                  '| Arm | Episodes | Fired | Distinct signatures | Not in grid | Severe episodes | '
                  'Distinct severe signatures | Outcome changed | Post-fault tokens |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm, label in ARMS:
            if arm not in pilots:
                continue
            rows = pilots[arm]
            fired = [r for r in rows if r['fault_fired']]
            found = {r['signature'] for r in fired}
            lines.append(
                f'| {label} | {len(rows)} | {len(fired)} | {len(found)} | {len(found - union)} | '
                f'{sum(map(severe, fired))} | {len({r["signature"] for r in fired if severe(r)})} | '
                f'{sum(r["propagation"] == "task_outcome_changed" for r in fired)} | '
                f'{sum(r["post_fault_tokens"] for r in fired):,} |')
        if 'fuzz' in pilots:
            fuzz = [r for r in pilots['fuzz'] if r['mutation'] not in (None, 'initial')]
            lines += ['', '| Fuzzer mutation | Proposed | New signature |', '|---|---:|---:|']
            seen = {r['signature'] for r in pilots['fuzz'] if r['mutation'] == 'initial'}
            tally = Counter()
            for r in fuzz:
                tally[r['mutation'], 'all'] += 1
                if r['signature'] and r['signature'] not in seen:
                    tally[r['mutation'], 'new'] += 1
                    seen.add(r['signature'])
            for mutation in ('step', 'operator', 'role', 'game', 'seed'):
                lines.append(f'| {mutation} | {tally[mutation, "all"]} | {tally[mutation, "new"]} |')
        for arm, rows in pilots.items():
            print(f'{arm:7s} curve', curve([r['signature'] for r in rows]))
    args.out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    if args.figure and pilots:
        budget = max(len(rows) for rows in pilots.values())
        xs = list(range(1, budget + 1))
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.15), sharex=True)
        panels = (('All signatures', lambda r: r['signature']),
                  ('Severe signatures (outcome changed or\nunauthorized/irreversible action)',
                   lambda r: r['signature'] if r['fault_fired'] and severe(r) else None))
        for ax, (title, pick) in zip(axes, panels):
            band = subsample([pick(r) for r in grid], xs, rounds=300)
            ax.fill_between(xs, [band[x][1] for x in xs], [band[x][2] for x in xs], color='#e1e0d9',
                            linewidth=0, label='Random subsample of grid (95%)')
            for (arm, label), color in zip(ARMS, STAGE_COLORS):
                if arm in pilots:
                    ys = curve([pick(r) for r in pilots[arm]])
                    ax.plot(range(1, len(ys) + 1), ys, color=color, linewidth=2,
                            label=f'{label}: {ys[-1]}')
            ax.set_title(title, fontsize=8, loc='left', color=INK)
            ax.set_xlim(0, budget)
            ax.set_ylim(0)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel('Faulty episodes run')
            tidy(ax, 'y')
            ax.tick_params(colors=MUTED)
            ax.legend(loc='upper left', fontsize=6.5, labelcolor=SECONDARY)
        axes[0].set_ylabel('Distinct signatures found')
        fig.tight_layout()
        fig.savefig(args.figure)
        plt.close(fig)

if __name__ == '__main__':
    main()
