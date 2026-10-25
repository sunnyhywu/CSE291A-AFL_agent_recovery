"""Batch-one ALFWorld adapter; the workflow also accepts fake adapters in tests."""
from dataclasses import dataclass
from importlib.metadata import version
import os
from pathlib import Path
import random
from typing import Protocol


@dataclass(frozen=True)
class Frame:
    text: str
    commands: tuple[str, ...]
    gamefile: str
    score: float = 0.0
    done: bool = False
    won: bool = False
    oracle_commands: tuple[str, ...] | None = None


class Environment(Protocol):
    def describe(self) -> dict: ...
    def reset(self) -> Frame: ...
    def step(self, command: str) -> Frame: ...
    def close(self) -> None: ...


class AlfworldEnvironment:
    def __init__(self, config_path: Path, seed=42, max_steps=150, gamefile=None,
                 task_types=(1,), oracle_hints=True):
        import numpy as np
        import textworld
        import textworld.gym
        import yaml
        from alfworld.agents.environment import get_environment
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler, AlfredInfos, AlfredExpert)

        self.seed, self.oracle_hints = seed, oracle_hints
        self.config = yaml.safe_load(config_path.read_text())
        self.config['env'].update(type='AlfredTWEnv', domain_randomization=False,
                                  task_types=list(task_types), expert_type='handcoded')
        self.config['general'].update(use_cuda=False, random_seed=seed, training_method='dagger')
        self.config['dagger']['training']['max_nb_steps_per_episode'] = max_steps
        # Apply the game selection after sorting, independently of filesystem traversal order.
        self.config['dataset']['num_train_games'] = -1
        owner = get_environment('AlfredTWEnv')(self.config, train_eval='train')
        games = sorted(str(Path(p).resolve()) for p in owner.game_files)
        if gamefile is not None:
            selected = str(gamefile.resolve())
            if selected not in games:
                raise ValueError('Gamefile must be an eligible training game for --task-types')
        elif games:
            selected = games[0]
        else:
            raise RuntimeError('No games found; check ALFWORLD_DATA and run alfworld-download')
        self.gamefile = selected
        random.seed(seed)
        np.random.seed(seed)
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        extras = ['gamefile']
        if oracle_hints:
            wrappers.append(AlfredExpert)
            extras.append('expert_plan')
        env_id = textworld.gym.register_games(
            [selected], textworld.EnvInfos(won=True, admissible_commands=True, extras=extras),
            batch_size=1, asynchronous=False, max_episode_steps=max_steps, wrappers=wrappers)
        self.env = textworld.gym.make(env_id)
        self.env.seed(seed)

    def describe(self):
        return {'backend': 'alfworld', 'alfworld_version': version('alfworld'),
                'config': self.config, 'gamefile': self.gamefile,
                'split': 'train', 'oracle_hints': self.oracle_hints,
                'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'unset')}

    def _frame(self, obs, info, score=0.0, done=False):
        return Frame(text=str(obs[0]), commands=tuple(info['admissible_commands'][0]),
                     gamefile=str(info['extra.gamefile'][0]), score=float(score),
                     done=bool(done), won=bool(info.get('won', [False])[0]),
                     oracle_commands=tuple(info['extra.expert_plan'][0]) if self.oracle_hints else None)

    def reset(self):
        obs, info = self.env.reset()
        return self._frame(obs, info)

    def step(self, command):
        obs, scores, dones, info = self.env.step([command])
        return self._frame(obs, info, scores[0], dones[0])

    def close(self):
        self.env.close()
