"""Single-episode, oracle-control baseline. No fault injection or recovery."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import os
import random
from pathlib import Path
from typing import Protocol


from agent_recovery.trace import Artifact, ExecutionEvent, Trace


class Agent(Protocol):
    def run(self, trace: Trace, step: int, *inputs: Artifact) -> Artifact: ...


class StateAgent:
    def run(self, trace, step, observation):
        return trace.add('state', 'state_agent', (observation.id,), step,
                         {'text': observation.content['text'],
                          'admissible_commands': observation.content['admissible_commands']})


class MemoryAgent:
    def run(self, trace, step, state, previous=None):
        parents = (state.id,) + ((previous.id,) if previous else ())
        history = (previous.content['observations'] if previous else []) + [state.content['text']]
        return trace.add('memory', 'memory_agent', parents, step, {'observations': history})


class OraclePlanner:
    def run(self, trace, step, state, memory, oracle):
        plan = oracle.content['commands']
        if not plan:
            raise RuntimeError('Expert supplied no action')
        action = plan[0]
        # ALFWorld uses look to initialize its handcoded expert.
        if action not in state.content['admissible_commands'] and action != 'look':
            raise RuntimeError(f'Expert action is inadmissible: {action}')
        return trace.add('plan', 'oracle_planner', (state.id, memory.id, oracle.id), step,
                         {'action': action, 'policy': 'alfworld_handcoded_expert'})


def snapshot(trace, step, obs, info, parents):
    observation = trace.add('observation', 'environment', parents, step,
                            {'text': str(obs[0]),
                             'admissible_commands': list(info['admissible_commands'][0]),
                             'gamefile': str(info['extra.gamefile'][0])})
    oracle = trace.add('oracle_hint', 'environment_expert', (observation.id,), step,
                       {'commands': list(info['extra.expert_plan'][0])})
    return observation, oracle


def run(args):
    import yaml
    from alfworld.agents.environment import get_environment
    config = yaml.safe_load(args.config.read_text())
    config['env'].update(type='AlfredTWEnv', domain_randomization=False, task_types=[1],
                         expert_type='handcoded')
    config['general'].update(use_cuda=False, random_seed=args.seed, training_method='dagger')
    config['dagger']['training']['max_nb_steps_per_episode'] = args.max_steps
    owner = get_environment('AlfredTWEnv')(config, train_eval='train')
    owner.game_files = sorted(owner.game_files)
    if args.gamefile:
        selected = str(args.gamefile.resolve())
        if selected not in owner.game_files:
            raise ValueError('Gamefile must belong to configured training pick-and-place dataset')
        owner.game_files = [selected]
    else:
        owner.game_files = owner.game_files[:1]
    if not owner.game_files:
        raise RuntimeError('No games found; run alfworld-download and check ALFWORLD_DATA')
    owner.num_games = len(owner.game_files)
    trace = Trace(args.trace)
    env = None
    try:
        trace.event('run_start', 0, seed=args.seed, config=config, gamefiles=owner.game_files,
                    alfworld_version=version('alfworld'), policy='oracle_control',
                    python_hash_seed=os.environ.get('PYTHONHASHSEED', 'unset'))
        # Synchronous batch keeps expert RNG in this seeded process.
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler, AlfredInfos, AlfredExpert)
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        env_id = textworld.gym.register_games(
            owner.game_files,
            textworld.EnvInfos(won=True, admissible_commands=True,
                               extras=['gamefile', 'expert_plan']),
            batch_size=1, asynchronous=False, max_episode_steps=args.max_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert])
        env = textworld.gym.make(env_id)
        env.seed(args.seed)
        obs, info = env.reset()
        observation, oracle = snapshot(trace, 0, obs, info, ())
        memory = None
        state_agent, memory_agent, planner = StateAgent(), MemoryAgent(), OraclePlanner()
        for step in range(args.max_steps):
            state = state_agent.run(trace, step, observation)
            memory = memory_agent.run(trace, step, state, memory)
            plan = planner.run(trace, step, state, memory, oracle)
            action = trace.add('action', 'deterministic_executor', (plan.id,), step,
                               {'command': plan.content['action']})
            trace.event('action_requested', step, (action.id,), command=action.content['command'])
            obs, scores, dones, info = env.step([action.content['command']])
            observation, oracle = snapshot(trace, step + 1, obs, info, (action.id, observation.id))
            won = bool(info['won'][0])
            trace.event('transition', step + 1, (action.id, observation.id),
                        score=float(scores[0]), done=bool(dones[0]), won=won)
            if dones[0] or won:
                trace.event('run_end', step + 1, won=won, reason='terminal')
                return won
        trace.event('run_end', args.max_steps, won=False, reason='step_limit')
        return False
    except Exception as exc:
        trace.event('run_error', -1, error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        trace.file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/alfworld.yaml')
    parser.add_argument('--trace', type=Path, default=Path('trace.jsonl'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--gamefile', type=Path)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    raise SystemExit(0 if run(args) else 1)
