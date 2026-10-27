"""Phase 1 report figures (vector PDF) from the comparison CSVs.

  python -m analysis.figures --faults results/team-faults/comparison.csv \
      --verify-faults results/team-verify-faults/comparison.csv --out docs/report/figures

Prints every plotted number so the report text can quote them exactly.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analysis.report import game_of, read  # noqa: E402

# Reference palette (dataviz skill), light mode; validated: ordinal ramp and 3-slot categorical.
INK, SECONDARY, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
OUTCOME_COLORS = ('#86b6ef', '#3987e5', '#1c5cab', '#0d366b')  # Ordinal: least to most severe.
STAGE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
BAR = '#2a78d6'
OUTCOMES = (('masked', 'Masked'), ('reasoning_only', 'Reasoning only'),
            ('actions_changed', 'Actions changed'), ('task_outcome_changed', 'Task outcome changed'))
STAGES = (('environment', 'Raw observation'), ('observation', 'Observation agent'),
          ('memory', 'Memory agent'), ('planner', 'Planner'))

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': SECONDARY, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': SECONDARY,
    'ytick.labelcolor': INK, 'text.color': INK, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'legend.frameon': False, 'pdf.fonttype': 42,
})


def bootstrap(rows, value, rounds=2000, seed=0):
    """Mean and 95% CI of value(row), resampling games (episodes of one game correlate)."""
    groups = defaultdict(list)
    for row in rows:
        groups[game_of(row['episode'])].append(value(row))
    games = list(groups.values())
    rng, means = random.Random(seed), []
    for _ in range(rounds):
        sample = [v for g in rng.choices(games, k=len(games)) for v in g]
        means.append(sum(sample) / len(sample))
    means.sort()
    flat = [v for g in games for v in g]
    return sum(flat) / len(flat), means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def tidy(ax, grid_axis):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def propagation(faults, out: Path):
    """Outcome class shares per stage, accidental and adversarial faults (100% stacked)."""
    fired = [r for r in faults if r['fault_fired']]
    rows = []
    for category, title in (('accidental', 'Accidental'), ('adversarial', 'Adversarial')):
        for stage, label in STAGES:
            subset = [r for r in fired if r['category'] == category and r['role'] == stage]
            if subset:
                shares = [sum(r['propagation'] == key for r in subset) / len(subset) for key, _ in OUTCOMES]
                rows.append((f'{title} · {label}', len(subset), shares))
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    for y, (label, n, shares) in enumerate(reversed(rows)):
        left = 0.0
        for share, color, (_, name) in zip(shares, OUTCOME_COLORS, OUTCOMES):
            ax.barh(y, share, left=left, height=0.62, color=color, edgecolor='white', linewidth=1.2)
            if share >= 0.08:
                ax.text(left + share / 2, y, f'{share:.0%}', ha='center', va='center', fontsize=7,
                        color=INK if color == OUTCOME_COLORS[0] else 'white')
            left += share
        print(f'propagation  {label:40s} n={n:4d} ' + ' '.join(f'{s:.3f}' for s in shares))
    labels = [f'{label}  (n={n})' for label, n, _ in reversed(rows)]
    ax.set_yticks(range(len(rows)), labels)
    split = sum(1 for label, _, _ in rows if label.startswith('Adversarial')) - 0.5
    ax.axhline(split, color=AXIS, linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel('Share of fired faults (seed 42, all positions)')
    tidy(ax, 'x')
    handles = [matplotlib.patches.Patch(color=c, label=name) for c, (_, name) in zip(OUTCOME_COLORS, OUTCOMES)]
    ax.legend(handles=handles, ncols=4, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=7.5,
              handlelength=1.2, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(out / 'propagation.pdf')
    plt.close(fig)


def cost(faults, verify, out: Path):
    """Tokens to finish an episode under each response to a fault, relative to the clean run,
    on the identical seed-42 mid-position schedule. Annotated with the share of tasks won."""
    baseline = {r['episode']: r for r in faults if r['fault_fired'] and r['position'] == 'mid'}
    paired = [(baseline[r['episode']], r) for r in verify
              if r['fault_fired'] and r['episode'] in baseline and r['faulty_complete']]
    methods = (
        ('No recovery\n(continue on faulty state)', lambda b, v: b['faulty_tokens'] / b['clean_tokens'],
         lambda b, v: b['faulty_won']),
        ('Always verify', lambda b, v: v['faulty_tokens'] / b['clean_tokens'], lambda b, v: v['faulty_won']),
        ('Full restart,\nfault detected immediately', lambda b, v: b['restart_tokens_immediate'] / b['clean_tokens'],
         None),
        ('Full restart,\nfault detected at end', lambda b, v: b['restart_tokens_at_end'] / b['clean_tokens'],
         None),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True, sharex=True)
    top = 0
    for ax, category in zip(axes, ('accidental', 'adversarial')):
        subset = [(b, v) for b, v in paired if b['category'] == category]
        rows = [{'episode': b['episode'], 'pair': (b, v)} for b, v in subset]
        for y, (name, ratio, won) in enumerate(reversed(methods)):
            mean, low, high = bootstrap(rows, lambda r: ratio(*r['pair']))
            ax.barh(y, mean, height=0.58, color=BAR)
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], color=INK, linewidth=0.8,
                        capsize=2.5)
            # Restart success is the executed live restarts' rate (18/18), marked with a dagger.
            share = (sum(won(b, v) for b, v in subset) / len(subset)) if won else 1.0
            ax.text(high + 0.08, y, f'{mean:.2f}×  ·  {share:.0%} won{"" if won else "†"}',
                    va='center', fontsize=7, color=SECONDARY)
            top = max(top, high)
            print(f'cost  {category:11s} {name.replace(chr(10), " "):48s} {mean:.3f} '
                  f'[{low:.3f}, {high:.3f}] won={share:.3f} n={len(subset)}')
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_title(f'{category.capitalize()} faults (n={len(subset)})', fontsize=8, loc='left', color=INK)
        ax.set_xlabel('Tokens relative to the clean episode (×)')
        tidy(ax, 'x')
    axes[0].set_xlim(0, top + 1.3)  # One scale for both panels.
    axes[0].set_yticks(range(len(methods)), [name for name, _, _ in reversed(methods)])
    fig.tight_layout()
    fig.savefig(out / 'cost.pdf')
    plt.close(fig)


def delay(faults, out: Path):
    """Steps between injection and the first changed action, for faults that changed actions."""
    buckets = (('0', 0, 0), ('1', 1, 1), ('2–4', 2, 4), ('5–9', 5, 9), ('≥10', 10, 10 ** 9))
    stages = STAGES[:3]  # Planner faults always change the action at once.
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    width = 0.26
    for i, ((stage, label), color) in enumerate(zip(stages, STAGE_COLORS)):
        delays = [r['steps_to_divergence'] for r in faults
                  if r['fault_fired'] and r['role'] == stage and r['steps_to_divergence'] is not None]
        shares = [sum(lo <= d <= hi for d in delays) / len(delays) for _, lo, hi in buckets]
        xs = [b + (i - 1) * width for b in range(len(buckets))]
        ax.bar(xs, shares, width=width - 0.03, color=color, label=f'{label} (n={len(delays)})')
        for x, share in zip(xs, shares):
            ax.text(x, share + 0.012, f'{share:.0%}', ha='center', va='bottom', fontsize=6.5, color=SECONDARY)
        print(f'delay  {label:18s} n={len(delays):4d} ' + ' '.join(f'{s:.3f}' for s in shares)
              + f'  max={max(delays)}')
    ax.set_xticks(range(len(buckets)), [name for name, _, _ in buckets])
    ax.set_xlabel('Environment steps from injection to the first changed action')
    ax.set_ylabel('Share of faults that changed actions')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.85)
    tidy(ax, 'y')
    ax.legend(loc='upper right', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / 'delay.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faults', type=Path, required=True)
    parser.add_argument('--verify-faults', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    faults, verify = read(args.faults), read(args.verify_faults)
    propagation(faults, args.out)
    cost(faults, verify, args.out)
    delay(faults, args.out)


if __name__ == '__main__':
    main()
