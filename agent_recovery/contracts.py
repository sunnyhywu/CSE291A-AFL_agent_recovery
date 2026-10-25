"""Public hook contract. Hooks never receive the environment or trace registry."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .trace import Artifact

ROLES = ('observation', 'memory', 'constraints', 'planner', 'verifier')
OUTPUT_KINDS = dict(zip(ROLES, ('state', 'memory', 'constraints', 'plan', 'verification')))


@dataclass(frozen=True)
class AgentRequest:
    role: str
    step: int
    attempt: int
    seed: int
    inputs: Mapping[str, Artifact]


@dataclass(frozen=True)
class AgentResult:
    content: dict[str, Any]
    # Optional numeric usage/cost metrics, supplied by your hook.
    metrics: dict[str, float] = field(default_factory=dict)


class Agent(Protocol):
    def run(self, request: AgentRequest) -> AgentResult: ...


def validate_result(role: str, content: dict):
    if not isinstance(content, dict):
        raise TypeError(f'{role}: result.content must be a dictionary')

    def field(name, expected):
        value = content.get(name)
        if type(value) is not expected:
            raise ValueError(f'{role}: {name} must be {expected.__name__}')
        return value

    def strings(name):
        if not all(isinstance(item, str) for item in field(name, list)):
            raise ValueError(f'{role}: {name} must contain strings')

    if role == 'observation':
        field('summary', str)
        strings('facts')
    elif role == 'memory':
        field('summary', str)
        strings('observations')
        strings('actions')
    elif role == 'constraints':
        field('goal', str)
        strings('allowed_actions')
    elif role == 'planner':
        if not field('action', str).strip():
            raise ValueError('planner: action cannot be blank')
    elif role == 'verifier':
        field('approved', bool)
        field('reason', str)
    else:
        raise ValueError(f'Unknown role: {role}')
