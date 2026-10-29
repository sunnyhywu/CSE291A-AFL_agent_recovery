"""World-state restoration costs measured from existing traces (no new runs).

For every fired fault whose executed actions diverged, the world must be brought back to
its state before the first divergent action d. Two mechanisms apply in ALFWorld:
reset-and-replay (reset, then re-execute the recorded clean actions 0..d-1) and
compensation (undo the faulty actions from d on with inverse actions, newest first). The
action-effect registry (agent_recovery.policy.effect) decides whether compensation is sound.

  python -m analysis.restore --faults results/team-faults --clean results/team-clean \
      --out results/restore.md
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from agent_recovery.policy import effect
from analysis.report import read


def timings(trace: Path):
    """(reset seconds, [environment step seconds]) from one trace's events."""
    events = [r for r in map(json.loads, trace.read_text().splitlines()) if r['record'] == 'event']
    start = next(e for e in events if e['kind'] == 'run_start')
    first_call = next(e for e in events if e['kind'] == 'agent_call_start')
    steps, requested = [], None
    for event in events:
        if event['kind'] == 'action_requested':
            requested = event['elapsed_seconds']
        elif event['kind'] == 'transition' and requested is not None:
            steps.append(event['elapsed_seconds'] - requested)
            requested = None
    return first_call['elapsed_seconds'] - start['elapsed_seconds'], steps


def actions(trace: Path):
    return [r['content']['command'] for r in map(json.loads, trace.read_text().splitlines())
            if r['record'] == 'artifact' and r['kind'] == 'action']


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faults', type=Path, required=True)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    resets, steps = [], []
    for trace in sorted(args.clean.glob('*.jsonl')):
        reset, step = timings(trace)
        resets.append(reset)
        steps += step
    step_ms = statistics.median(steps) * 1000
    reset_ms = statistics.median(resets) * 1000

    rows = [r for r in read(args.faults / 'comparison.csv') if r['fault_fired']]
    diverged = [r for r in rows if r['first_divergent_action'] is not None]
    suffix_classes, prefix_lengths, undo_lengths, by_category = Counter(), [], [], Counter()
    for row in diverged:
        trace = next((args.faults / row['episode']).glob('*.jsonl'))
        faulty, clean = actions(trace), actions(args.clean / trace.name)
        d = row['first_divergent_action']
        suffix = [effect(a) for a in faulty[d:]]
        # An irreversible action the clean run also takes later leaves the same final state.
        foreign = [a for a in faulty[d:] if effect(a) == 'irreversible' and a not in clean[d:]]
        kind = ('unrecoverable' if foreign else 'irreversible' if 'irreversible' in suffix else
                'compensable' if 'compensable' in suffix else 'none')
        suffix_classes[kind] += 1
        by_category[row['category'], kind] += 1
        prefix_lengths.append(row['first_divergent_action'])
        undo_lengths.append(sum(e == 'compensable' for e in suffix))
    n = len(diverged)
    lines = [
        '# World-state restoration costs (from existing traces)', '',
        f'- Environment reset: median {reset_ms:.0f} ms over {len(resets)} clean episodes; '
        f'one environment step: median {step_ms:.1f} ms (p95 {quantile(steps, 0.95) * 1000:.1f} ms) '
        f'over {len(steps)} steps.',
        f'- {n} of {len(rows)} fired faults ({n / len(rows):.0%}) changed executed actions and need '
        'world-state restoration; the rest need only agent-state rollback.',
        f'- Replay prefix (clean actions before the first divergent one): median '
        f'{statistics.median(prefix_lengths):.0f}, max {max(prefix_lengths)} actions, i.e. about '
        f'{(reset_ms + statistics.median(prefix_lengths) * step_ms) / 1000:.2f} s of reset-and-replay '
        f'(max {(reset_ms + max(prefix_lengths) * step_ms) / 1000:.2f} s).',
        f'- Compensation (undo faulty actions newest first): median '
        f'{statistics.median(undo_lengths):.0f}, max {max(undo_lengths)} inverse actions.', '',
        '| Faulty suffix after divergence | Accidental | Adversarial | All | Share |',
        '|---|---:|---:|---:|---:|']
    for kind, label in (('none', 'observation only (nothing to undo)'),
                        ('compensable', 'compensable by inverse actions'),
                        ('irreversible', 'irreversible action the clean run also takes'),
                        ('unrecoverable', 'irreversible action the clean run never takes')):
        lines.append(f'| {label} | {by_category["accidental", kind]} | '
                     f'{by_category["adversarial", kind]} | {suffix_classes[kind]} | '
                     f'{suffix_classes[kind] / n:.0%} |')
    args.out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
