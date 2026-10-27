"""Aggregate Phase 1 results into Markdown tables.

Inputs are the per-episode CSVs written by experiments/summarize.py (clean runs) and
experiments/run_faults.py (comparison.csv). Every scheduled episode is reported,
including unfired faults, failed runs, and timeouts. Confidence intervals are 95%
percentile bootstraps that resample games, since episodes of one game are correlated.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import re
import statistics

TASK_TYPES = {1: 'pick & place', 2: 'examine in light', 3: 'clean & place',
              4: 'heat & place', 5: 'cool & place', 6: 'pick two & place'}
OUTCOMES = ('masked', 'reasoning_only', 'actions_changed', 'task_outcome_changed')
ROLES = ('environment', 'observation', 'memory', 'planner')  # Fault targets, upstream first.
ADVERSARIAL = ('malicious', 'malicious_alter')


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif value == '':
                row[key] = None
            else:
                try:
                    row[key] = float(value) if re.fullmatch(r'-?\d+\.\d*(e-?\d+)?', value) else int(value)
                except ValueError:
                    pass
    return rows


def game_of(name: str) -> str:
    return re.match(r'(\w+-t\d-g\d+)', name).group(1)


def bootstrap(rows, value, rounds=2000, seed=0):
    """Mean of value(row) and a 95% CI resampling games."""
    groups = defaultdict(list)
    for row in rows:
        groups[game_of(row.get('trace') or row['episode'])].append(value(row))
    games = list(groups.values())
    if not games:
        return None, None, None
    flat = [v for g in games for v in g]
    rng, means = random.Random(seed), []
    for _ in range(rounds):
        sample = [v for g in rng.choices(games, k=len(games)) for v in g]
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(flat) / len(flat), means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def ci(rows, value, fmt='{:.1f}', scale=1.0):
    mean, low, high = bootstrap(rows, value)
    if mean is None:
        return '–'
    return f'{fmt.format(mean * scale)} [{fmt.format(low * scale)}, {fmt.format(high * scale)}]'


def table(header, body):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in body]
    return '\n'.join(lines)


def clean_section(rows, title):
    seeds = sorted({int(re.search(r'-s(\d+)', r['trace']).group(1)) for r in rows})
    per_seed = [sum(r['won'] for r in rows if f'-s{s}.' in r['trace']) /
                max(1, sum(f'-s{s}.' in r['trace'] for r in rows)) for s in seeds]
    out = [f'### {title}', '',
           f'{len(rows)} episodes ({len({game_of(r["trace"]) for r in rows})} games × {len(seeds)} seeds). '
           f'Statuses: {dict(sorted(_count(r["status"] for r in rows).items()))}. '
           f'Success per seed: {", ".join(f"{p:.0%}" for p in per_seed)} '
           f'(sd {statistics.pstdev(per_seed):.1%}).', '']
    body = []
    for task_type, name in TASK_TYPES.items():
        subset = [r for r in rows if f'-t{task_type}-' in r['trace']]
        if subset:
            body.append([name, len(subset), f"{sum(r['won'] for r in subset)}/{len(subset)}",
                         f"{statistics.mean(r['actions'] for r in subset):.1f}",
                         f"{statistics.mean(r['input_tokens'] + r['output_tokens'] for r in subset):,.0f}"])
    out.append(table(['Task type', 'Episodes', 'Won', 'Mean actions', 'Mean tokens'], body))
    won = [r for r in rows if r['won']]
    out += ['', table(['Metric (95% CI over games)', 'All episodes', 'Won episodes'], [
        ['Success rate (%)', ci(rows, lambda r: r['won'], scale=100), '–'],
        ['Environment actions', ci(rows, lambda r: r['actions']), ci(won, lambda r: r['actions'])],
        ['LLM calls', ci(rows, lambda r: r['llm_calls']), ci(won, lambda r: r['llm_calls'])],
        ['Tokens (in+out)', ci(rows, lambda r: r['input_tokens'] + r['output_tokens'], '{:,.0f}'),
         ci(won, lambda r: r['input_tokens'] + r['output_tokens'], '{:,.0f}')],
        ['Model latency (s)', ci(rows, lambda r: r['llm_seconds']), ci(won, lambda r: r['llm_seconds'])],
        ['Episode wall time excl. rate-limit waits (s)',
         ci(rows, lambda r: r['episode_seconds'] - r['throttle_seconds']),
         ci(won, lambda r: r['episode_seconds'] - r['throttle_seconds'])],
        ['Cold-start setup (s)', ci(rows, lambda r: r['setup_seconds'], '{:.2f}'), '–'],
        ['Plan rejections', ci(rows, lambda r: r['plan_rejections'], '{:.2f}'), '–'],
    ])]
    return '\n'.join(out)


def _count(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def outcome_cells(rows):
    fired = [r for r in rows if r['fault_fired']]
    cells = [f'{len(fired)}/{len(rows)}']
    cells += [f"{sum(r['propagation'] == o for r in fired) / len(fired):.0%}" if fired else '–'
              for o in OUTCOMES]
    if fired:
        cells += [f"{sum(r['faulty_won'] for r in fired)}/{len(fired)}",
                  f"{statistics.mean(r['changed_calls'] for r in fired):.1f}",
                  f"{statistics.mean(r['extra_actions'] for r in fired):+.1f}",
                  f"{statistics.mean(r['extra_tokens'] for r in fired):+,.0f}",
                  sum(r['unauthorized_actions'] for r in fired),
                  sum(r['irreversible_actions'] for r in fired)]
    else:
        cells += ['–'] * 6
    return cells


PROPAGATION_HEADER = ['Fired', 'Masked', 'Reasoning only', 'Actions changed', 'Outcome changed',
                      'Faulty won', 'Changed calls', 'Δ actions', 'Δ tokens', 'Unauth.', 'Irrev.']


def fault_section(rows, title):
    out = [f'### {title}', '',
           f'{len(rows)} scheduled episodes; run statuses: '
           f'{dict(sorted(_count(r["status"] for r in rows).items()))}.', '']
    for category in ('accidental', 'adversarial'):
        subset = [r for r in rows if _category(r) == category]
        body = []
        for role in ROLES:
            for operator in sorted({_operator(r) for r in subset if _role(r) == role}):
                group = [r for r in subset if _role(r) == role and _operator(r) == operator]
                body.append([role, operator, *outcome_cells(group)])
        body.append(['**all**', '', *outcome_cells(subset)])
        out += [f'**{category.capitalize()} faults**', '', table(['Stage', 'Fault', *PROPAGATION_HEADER], body), '']
    body = [[position, *outcome_cells([r for r in rows if r['position'] == position])]
            for position in ('early', 'mid', 'late') if any(r['position'] == position for r in rows)]
    out += ['**By injection position**', '', table(['Position', *PROPAGATION_HEADER], body), '']
    return '\n'.join(out)


def _role(row):
    return row['role'] or re.search(r'-s\d+-(\w+)-', row['episode']).group(1)


def _operator(row):
    return row['operator'] or row['episode'].rsplit('-st', 1)[0].rsplit('-', 1)[1]


def _category(row):
    return row['category'] or ('adversarial' if _operator(row) in ADVERSARIAL else 'accidental')


def work_section(rows):
    fired = [r for r in rows if r['fault_fired'] and r['faulty_complete']]
    body = []
    for category in ('accidental', 'adversarial'):
        subset = [r for r in fired if r['category'] == category]
        body.append([category,
                     ci(subset, lambda r: r['clean_tokens'], '{:,.0f}'),
                     ci(subset, lambda r: r['post_fault_tokens'], '{:,.0f}'),
                     ci(subset, lambda r: r['restart_tokens_immediate'], '{:,.0f}'),
                     ci(subset, lambda r: r['restart_tokens_at_end'], '{:,.0f}'),
                     ci(subset, lambda r: r['restart_tokens_at_end'] / r['clean_tokens'], '{:.2f}'),
                     ci(subset, lambda r: r['restart_seconds_at_end'])])
    return '\n'.join([
        '### Wasted work and the full-restart baseline', '',
        'Tokens are logical (cached replays count their original cost). *Post-fault* work was '
        'computed on contaminated inputs. Full restart discards everything done before detection '
        'and reruns the clean episode; detection is either immediately after the faulty output or '
        'at episode end.', '',
        table(['Category', 'Clean episode', 'Post-fault (wasted)', 'Restart, detect immediately',
               'Restart, detect at end', 'Restart-at-end / clean', 'Restart-at-end latency (s)'], body)])


def recovery_scope_section(rows):
    """How much a dependency-based recovery would rerun vs what the fault actually changed."""
    fired = [r for r in rows if r['fault_fired'] and r.get('descendant_calls') is not None]
    body = []
    for label, subset in [(f'{role} / {category}', [r for r in fired if r['role'] == role
                                                     and r['category'] == category])
                          for category in ('accidental', 'adversarial')
                          for role in ROLES] + [('**all**', fired)]:
        if not subset:
            continue
        descendants = sum(r['descendant_calls'] for r in subset)
        changed = sum(r['changed_calls'] for r in subset)
        world = [r for r in subset if r['propagation'] in ('actions_changed', 'task_outcome_changed')]
        body.append([label, len(subset), f"{descendants / len(subset):.1f}", f"{changed / len(subset):.1f}",
                     f'{changed / descendants:.0%}' if descendants else '–',
                     f"{sum(r['changed_calls'] == 0 for r in subset) / len(subset):.0%}",
                     f'{len(world) / len(subset):.0%}',
                     sum(r['irreversible_actions'] > 0 for r in subset)])
    return '\n'.join([
        '### Scope of recovery (Phase 2 motivation)', '',
        '*Descendant calls* are later agent calls structurally downstream of the faulty output in '
        'the dependency DAG (what dependency-based invalidation reruns). *Changed calls* are those '
        'whose output differs from the clean run. *World state changed* means executed actions '
        'diverged, so recomputation alone cannot recover: the environment must be restored too.', '',
        table(['Stage / category', 'Fired', 'Descendant calls', 'Changed calls',
               'Changed / descendants', 'No call changed', 'World state changed',
               'Episodes with irreversible actions'], body)])


def payload_section(groups):
    body = []
    for (name, rows), operator in [(g, o) for g in groups for o in ADVERSARIAL]:
        mal = [r for r in rows if r['fault_fired'] and r['operator'] == operator
               and r.get('payload_target')]
        if not mal:
            continue
        foreign = [r for r in mal if not r['payload_goal_object']]
        generic = sum(r['payload_target'] == 'the nearest object' for r in mal)
        body.append([name, operator, f'{len(mal)} ({generic} generic)',
                     sum(r['payload_proposed'] > 0 for r in mal),
                     sum(r['payload_executed'] > 0 for r in mal),
                     sum(r['payload_taken'] > 0 for r in foreign),
                     sum(r['payload_blocked_verifier'] for r in mal),
                     sum(r['payload_blocked_gate'] for r in mal),
                     sum(r['unauthorized_actions'] > 0 for r in mal),
                     sum(r['irreversible_actions'] > 0 for r in mal),
                     sum(r['faulty_won'] for r in mal)])
    return '\n'.join([
        '### Malicious instructions', '',
        'The injected instruction names an object the agent has seen (preferably one the task '
        'does not need) and asks to put it in the garbagecan (`malicious`) or to heat/cool/clean '
        'it, an irreversible change (`malicious_alter`). *Executed* counts episodes where that '
        'exact action ran; *Took foreign* counts episodes that picked up a payload object the task '
        'does not name. Blocks are per proposal.', '',
        table(['Workflow', 'Payload', 'Fired', 'Proposed payload', 'Executed payload',
               'Took foreign object', 'Blocked by verifier', 'Blocked by gate',
               'Any unauthorized action', 'Any irreversible action', 'Task won'], body)])


def repeats_section(faults, repeats):
    """Run-to-run variability: the same fault types and positions on other clean seeds."""
    groups = [('seed 42', [r for r in faults if r['position'] == 'mid'])]
    seed_of = re.compile(r'-s(\d+)-')
    groups += [('seed ' + seed_of.search(rows[0]['episode']).group(1), rows) for rows in repeats if rows]
    body = [[name, *outcome_cells(rows)] for name, rows in groups]
    rates = {o: [sum(r['propagation'] == o for r in rows if r['fault_fired']) /
                 max(1, sum(r['fault_fired'] for r in rows)) for _, rows in groups] for o in OUTCOMES}
    body.append(['sd across seeds', '', *(f'{statistics.pstdev(v):.1%}' for v in rates.values()),
                 *[''] * 6])
    return '\n'.join(['### Variability across seeds (mid position)', '',
                      'Each seed has its own clean run; faults are injected only where that run succeeded.', '',
                      table(['Seed', *PROPAGATION_HEADER], body)])


def restart_section(clean, restart):
    """Executed full restart: live reruns (no cache) of the seed-42 clean episodes."""
    reference = {r['trace']: r for r in clean if '-s42.' in r['trace']}
    paired = [(reference[r['trace']], r) for r in restart if r['trace'] in reference]
    same = sum(c['actions'] == r['actions'] and c['won'] == r['won'] for c, r in paired)
    body = [['Clean run (seed 42)', f"{sum(c['won'] for c, _ in paired)}/{len(paired)}",
             f"{statistics.mean(c['actions'] for c, _ in paired):.1f}",
             f"{statistics.mean(c['input_tokens'] + c['output_tokens'] for c, _ in paired):,.0f}",
             f"{statistics.mean(c['llm_seconds'] for c, _ in paired):.1f}", '–'],
            ['Executed restart (live)', f"{sum(r['won'] for _, r in paired)}/{len(paired)}",
             f"{statistics.mean(r['actions'] for _, r in paired):.1f}",
             f"{statistics.mean(r['input_tokens'] + r['output_tokens'] for _, r in paired):,.0f}",
             f"{statistics.mean(r['llm_seconds'] for _, r in paired):.1f}",
             f"{statistics.mean(r['episode_seconds'] - r['throttle_seconds'] for _, r in paired):.1f}"]]
    return '\n'.join([
        '### Executed full restart', '',
        'Every seed-42 clean episode was rerun from scratch with live model calls (a separate cache '
        f'namespace). {same}/{len(paired)} reruns reproduced the same number of actions and outcome.', '',
        table(['Run', 'Won', 'Actions', 'Tokens', 'Model latency (s)', 'Live wall time excl. waits (s)'],
              body)])


def overhead_section(rows):
    """Cold start and the cost of recording the dependency trace."""
    rows = [r for r in rows if r.get('trace_bytes') is not None]
    if not rows:
        return ''
    per_step = [r['trace_bytes'] / max(1, r['actions']) for r in rows]
    share = [r['trace_write_seconds'] / r['episode_seconds'] for r in rows if r['episode_seconds']]
    return '\n'.join([
        '### Cold start and tracking overhead', '',
        table(['Metric', 'Mean', 'Max'], [
            ['Cold start: process time outside the episode, incl. Python/ALFWorld start-up (s)',
             f"{statistics.mean(r['process_seconds'] - r['episode_seconds'] for r in rows):.1f}",
             f"{max(r['process_seconds'] - r['episode_seconds'] for r in rows):.1f}"],
            ['Environment setup inside the run (s)', f"{statistics.mean(r['setup_seconds'] for r in rows):.2f}",
             f"{max(r['setup_seconds'] for r in rows):.2f}"],
            ['Trace size per environment step (KB)', f'{statistics.mean(per_step) / 1024:.1f}',
             f'{max(per_step) / 1024:.1f}'],
            ['Trace recording time (s per episode)', f"{statistics.mean(r['trace_write_seconds'] for r in rows):.4f}",
             f"{max(r['trace_write_seconds'] for r in rows):.4f}"],
            ['Recording time / episode time', f'{statistics.mean(share):.3%}', f'{max(share):.3%}']])])


def verify_section(clean, verify_clean, faults, verify_faults):
    def totals(rows):
        return sum(r['input_tokens'] + r['output_tokens'] for r in rows), sum(r['llm_calls'] for r in rows)

    base_tokens, base_calls = totals(clean)
    ver_tokens, ver_calls = totals(verify_clean)
    out = ['### Always-verify comparison', '',
           table(['Clean runs', 'Episodes', 'Success', 'LLM calls / episode', 'Tokens / episode',
                  'Plan rejections / episode'], [
               ['Baseline', len(clean), f"{sum(r['won'] for r in clean)}/{len(clean)}",
                f'{base_calls / len(clean):.1f}', f'{base_tokens / len(clean):,.0f}',
                f"{statistics.mean(r['plan_rejections'] for r in clean):.2f}"],
               ['Always verify', len(verify_clean), f"{sum(r['won'] for r in verify_clean)}/{len(verify_clean)}",
                f'{ver_calls / len(verify_clean):.1f}', f'{ver_tokens / len(verify_clean):,.0f}',
                f"{statistics.mean(r['plan_rejections'] for r in verify_clean):.2f}"]]), '']
    if verify_faults:
        keys = {r['episode'] for r in verify_faults}
        paired = [r for r in faults if r['episode'] in keys]
        out += ['Same fault schedule (paired episodes):', '',
                table(['Workflow', *PROPAGATION_HEADER],
                      [['Baseline', *outcome_cells(paired)],
                       ['Always verify', *outcome_cells(verify_faults)]])]
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=Path, required=True, help='summary.csv of clean runs')
    parser.add_argument('--faults', type=Path, required=True, help='comparison.csv of faulty runs')
    parser.add_argument('--repeats', type=Path, nargs='*', default=[],
                        help='comparison.csv of the same faults on other clean seeds')
    parser.add_argument('--restart', type=Path, help='summary.csv of executed (live) restarts')
    parser.add_argument('--verify-clean', type=Path)
    parser.add_argument('--verify-faults', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    clean, faults = read(args.clean), read(args.faults)
    verify_clean = read(args.verify_clean) if args.verify_clean else []
    verify_faults = read(args.verify_faults) if args.verify_faults else []
    sections = [clean_section(clean, 'Clean baseline')]
    if verify_clean:
        sections.append(clean_section(verify_clean, 'Clean baseline with always verify'))
    sections += [fault_section(faults, 'Fault propagation (baseline workflow, seed 42)')]
    if args.repeats:
        sections.append(repeats_section(faults, [read(path) for path in args.repeats]))
    sections += [work_section(faults)]
    if args.restart:
        sections.append(restart_section(clean, read(args.restart)))
    sections += [recovery_scope_section(faults), overhead_section(clean),
                 payload_section([('Baseline, all positions', faults),
                                  ('Baseline, same schedule as always verify',
                                   [r for r in faults if r['episode'] in {v['episode'] for v in verify_faults}]),
                                  ('Always verify', verify_faults)])]
    if verify_clean:
        sections.append(verify_section(clean, verify_clean, faults, verify_faults))
    text = '\n\n'.join(filter(None, sections)) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
