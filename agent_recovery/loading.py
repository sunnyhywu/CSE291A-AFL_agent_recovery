"""Load trusted local Python factories; this is a code extension point, not a sandbox."""
from copy import deepcopy
import importlib
from pathlib import Path

from .contracts import ROLES
from .workflow import AgentTeam


def load_team(path: Path, overrides=(), oracle_override=None):
    import yaml
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('Agent config must be a mapping with schema_version: 1')
    config = deepcopy(config)
    if set(config) - {'schema_version', 'oracle_hints', 'max_plan_attempts', 'roles'}:
        raise ValueError('Unknown agent configuration key')
    if type(config.get('oracle_hints')) is not bool:
        raise ValueError('oracle_hints must be true or false')
    if type(config.get('max_plan_attempts')) is not int or config['max_plan_attempts'] < 1:
        raise ValueError('max_plan_attempts must be a positive integer')
    if not isinstance(config.get('roles'), dict) or set(config['roles']) != set(ROLES):
        raise ValueError(f'Configure exactly these roles: {", ".join(ROLES)}')
    for override in overrides:
        role, separator, factory = override.partition('=')
        if not separator or role not in ROLES or not factory:
            raise ValueError('--agent must be ROLE=module:factory')
        config['roles'][role] = {'factory': factory, 'kwargs': {}}
    if oracle_override is not None:
        config['oracle_hints'] = oracle_override
    agents = {}
    for role in ROLES:
        spec = config['roles'][role]
        if not isinstance(spec, dict) or set(spec) - {'factory', 'kwargs'}:
            raise ValueError(f'Invalid factory specification for {role}')
        target = spec.get('factory', '')
        if not isinstance(target, str) or target.count(':') != 1:
            raise ValueError(f'{role}: factory must be module:attribute')
        module_name, attribute = target.split(':')
        factory = getattr(importlib.import_module(module_name), attribute)
        kwargs = spec.get('kwargs', {})
        if not isinstance(kwargs, dict):
            raise ValueError(f'{role}: kwargs must be a mapping')
        agent = factory(**kwargs)
        if not callable(getattr(agent, 'run', None)):
            raise TypeError(f'{target} must create an object with run(request)')
        agents[role] = agent
    team = AgentTeam(**agents)
    if not config['oracle_hints'] and getattr(team.planner, 'requires_oracle', False):
        raise ValueError('Replace OraclePlanner with --agent planner=module:factory before --no-oracle')
    return team, config
