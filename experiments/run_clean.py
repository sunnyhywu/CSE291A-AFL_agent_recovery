"""Run clean (fault-free) episodes as independent processes and summarize their traces.

Each episode is one CLI invocation, so a crash or timeout in one game is recorded
rather than aborting the batch. Existing traces are skipped, making batches resumable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def episode(out: Path, agents: Path, split: str, task_type: int, index: int, seed: int,
            max_steps: int, timeout: float, extra=(), cache_namespace=''):
    trace = out / f'{split}-t{task_type}-g{index:03d}-s{seed}.jsonl'
    if trace.exists():
        # Resumed batch: report the finished episode's recorded outcome.
        log = trace.with_suffix('.log')
        text = log.read_text() if log.exists() else ''
        return trace, 'won' if 'SUCCESS:' in text else 'stopped' if 'STOPPED:' in text else 'error'
    command = [sys.executable, '-m', 'agent_recovery', '--agents', str(agents), '--split', split,
               '--task-types', str(task_type), '--game-index', str(index), '--seed', str(seed),
               '--max-steps', str(max_steps), '--trace', str(trace), *extra]
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'LLM_CACHE_NAMESPACE': cache_namespace,
           'ALFWORLD_DATA': os.environ.get('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))}
    log = trace.with_suffix('.log')
    try:
        started = time.monotonic()
        with log.open('w') as handle:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
            # Whole-process time, including interpreter and ALFWorld start-up (cold start).
            print(f'PROCESS_SECONDS: {time.monotonic() - started:.2f}', file=handle)
        return trace, {0: 'won', 1: 'stopped'}.get(code, f'error({code})')
    except subprocess.TimeoutExpired:
        return trace, 'timeout'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agents', type=Path, default=ROOT / 'configs/llm_planner.yaml')
    parser.add_argument('--split', default='train', choices=('train', 'valid_seen', 'valid_unseen'))
    parser.add_argument('--games', nargs='+', required=True, metavar='TASK_TYPE:INDEX',
                        help='e.g. 1:0 2:0 3:5')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--cache-namespace', default='',
                        help='Isolate this batch from cached responses (e.g. executed restarts)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(int(t), int(i), seed) for spec in args.games for t, i in [spec.split(':')]
            for seed in args.seeds]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(episode, args.out, args.agents, args.split, t, i, seed,
                               args.max_steps, args.timeout, (), args.cache_namespace)
                   for t, i, seed in jobs]
        for future in futures:
            trace, status = future.result()
            print(f'{status:10s} {trace.name}', flush=True)


if __name__ == '__main__':
    main()
