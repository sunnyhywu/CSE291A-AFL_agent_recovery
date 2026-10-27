"""Per-episode metrics from traces, including incomplete and failed runs."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys


def summarize(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    events = [r for r in records if r['record'] == 'event']
    artifacts = [r for r in records if r['record'] == 'artifact']
    start = next((e for e in events if e['kind'] == 'run_start'), None)
    end = next((e for e in events if e['kind'] in ('run_end', 'run_error')), None)
    calls = [e for e in events if e['kind'] == 'agent_call_end']
    llm = [e['details']['metrics'] for e in calls if 'input_tokens' in e['details']['metrics']]
    plans = [a for a in artifacts if a['kind'] == 'plan']
    log = path.with_suffix('.log')
    process = re.search(r'PROCESS_SECONDS: ([\d.]+)', log.read_text()) if log.exists() else None
    tracking = end['details'].get('tracking', {}) if end and end['kind'] == 'run_end' else {}
    environment = start['details']['environment'] if start else {}
    won = bool(end and end['kind'] == 'run_end' and end['details']['won'])
    actions = sum(a['kind'] == 'action' for a in artifacts)
    status = (end['details'].get('reason', 'error') if end and end['kind'] == 'run_end'
              else 'error' if end else 'incomplete')
    if status == 'terminal' and not won and start and actions >= start['details']['max_steps']:
        status = 'step_limit'  # ALFWorld itself ends the game at the step limit.
    return {
        'trace': path.name,
        'gamefile': environment.get('gamefile', ''),
        'status': status,
        'won': won,
        'actions': actions,
        'agent_calls': len(calls),
        'llm_calls': len(llm),
        'plan_rejections': sum(e['kind'] == 'plan_rejected' for e in events),
        'parse_failures': sum(p['content'].get('parse') == 'failed' for p in plans),
        'input_tokens': sum(m['input_tokens'] for m in llm),
        'output_tokens': sum(m['output_tokens'] for m in llm),
        'llm_seconds': round(sum(m['llm_latency_seconds'] for m in llm), 2),
        'cache_hits': sum(m['cache_hit'] for m in llm),
        'throttle_seconds': round(sum(m.get('throttle_seconds', 0.0) for m in llm), 2),
        # Cold start (environment construction) is separate from the episode itself.
        'setup_seconds': round(start['elapsed_seconds'], 2) if start else None,
        'episode_seconds': round(end['elapsed_seconds'] - start['elapsed_seconds'], 2)
                           if start and end else None,
        'process_seconds': float(process.group(1)) if process else None,
        'trace_bytes': tracking.get('bytes'),
        'trace_write_seconds': round(tracking['write_seconds'], 4) if tracking else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    rows = [summarize(p) for p in sorted(args.directory.glob('*.jsonl'))]
    if not rows:
        sys.exit('No traces found')
    writer = csv.DictWriter(args.csv.open('w') if args.csv else sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    won = [r for r in rows if r['won']]
    print(f'\nepisodes={len(rows)} won={len(won)} ({len(won) / len(rows):.0%}) '
          f'statuses={sorted({r["status"] for r in rows})}', file=sys.stderr)
    for key in ('actions', 'llm_calls', 'input_tokens', 'output_tokens', 'llm_seconds',
                'setup_seconds', 'episode_seconds'):
        values = [r[key] for r in rows]
        print(f'{key:14s} mean={statistics.mean(values):9.1f} max={max(values)} '
              f'total={sum(values)}', file=sys.stderr)


if __name__ == '__main__':
    main()
