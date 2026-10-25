import argparse
import os
from pathlib import Path

from .environment import AlfworldEnvironment
from .loading import load_team
from .trace import Trace
from .workflow import Workflow


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Five-role ALFWorld workflow with replaceable agents')
    parser.add_argument('--config', type=Path, default=root / 'configs/alfworld.yaml')
    parser.add_argument('--agents', type=Path, default=root / 'configs/agents.yaml')
    parser.add_argument('--agent', action='append', default=[], metavar='ROLE=MODULE:FACTORY')
    parser.add_argument('--no-oracle', action='store_true', help='Disable expert wrapper and oracle artifacts')
    parser.add_argument('--trace', type=Path, default=Path('multi-agent.jsonl'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--max-plan-attempts', type=int, help='Total planner/verifier attempts per step')
    parser.add_argument('--gamefile', type=Path)
    parser.add_argument('--task-types', nargs='+', type=int, choices=range(1, 7), default=[1])
    args = parser.parse_args()
    if args.max_steps < 1 or (args.max_plan_attempts is not None and args.max_plan_attempts < 1):
        parser.error('Limits must be positive')
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))
    team, config = load_team(args.agents, args.agent, False if args.no_oracle else None)
    if args.max_plan_attempts is not None:
        config['max_plan_attempts'] = args.max_plan_attempts
    trace = Trace(args.trace)
    env = None
    try:
        env = AlfworldEnvironment(args.config, seed=args.seed, max_steps=args.max_steps,
                                 gamefile=args.gamefile, task_types=args.task_types,
                                 oracle_hints=config['oracle_hints'])
        workflow = Workflow(team, trace, seed=args.seed, max_steps=args.max_steps,
                            max_plan_attempts=config['max_plan_attempts'],
                            oracle_hints=config['oracle_hints'])
        won = workflow.run(env, configuration=config)
        print(f'{"SUCCESS" if won else "STOPPED"}: {args.trace} '
              f'({workflow.agent_calls} agent calls)')
        return 0 if won else 1
    except Exception as exc:
        # Workflow errors are already logged; record adapter startup failures here.
        if trace.sequence == 0:
            trace.event('run_error', 0, error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            trace.close()


if __name__ == '__main__':
    raise SystemExit(main())
