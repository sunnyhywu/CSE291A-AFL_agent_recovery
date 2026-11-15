"""Resume a recorded episode from the start of step k (state restore layers 1 and 2).

Layer 1, agent results and shared memory: artifacts are immutable, so the agent state before
step k is the trace prefix written before step k's first agent call. Rolling back means
pointing the workflow at those artifacts (`rebuild_state`).

Layer 2, a resettable deterministic environment: rolling the agents back does not roll the
world back. The environment is reset and the recorded actions 0..k-1 are re-executed without
querying any agent, and every replayed observation and command list is checked against the
recording (`replay`). A mismatch raises `ReplayMismatch` instead of resuming on a wrong world.

    trace = Trace(new_path)
    state = restore(source_path, step, trace, environment)
    Workflow(team, trace, ...).resume(environment, state)
"""
import json
from pathlib import Path
import time

from .contracts import ROLES
from .environment import Environment, Frame
from .trace import Artifact, Trace
from .workflow import StepState


class ReplayMismatch(RuntimeError):
    """The environment did not reproduce the recorded world state."""


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def prefix(records: list[dict], step: int) -> list[dict]:
    """The records written before the first agent call of `step`."""
    if not records or records[0].get('kind') != 'run_start':
        raise ValueError('Trace does not start with run_start')
    for index, record in enumerate(records):
        if record['record'] == 'event' and record['kind'] in ('run_resumed', 'state_restored'):
            raise ValueError('Resuming from an already resumed trace is not supported')
        if (record['record'] == 'event' and record['kind'] == 'agent_call_start'
                and record['step'] == step):
            return records[:index]
    raise ValueError(f'Trace has no agent call at step {step}')


def _artifacts(records):
    return [Artifact(r['id'], r['kind'], r['producer'], tuple(r['inputs']), r['step'], r['content'])
            for r in records if r['record'] == 'artifact']


def rebuild_state(records: list[dict], step: int) -> StepState:
    """Agent state at the start of `step`, from that step's trace prefix."""
    latest: dict[str, Artifact] = {}
    last_output: dict[str, Artifact] = {}
    for artifact in _artifacts(records):
        latest[artifact.kind] = artifact
        if artifact.producer in ROLES:
            last_output[artifact.producer] = artifact
    observation, action_space = latest.get('observation'), latest.get('action_space')
    task, oracle = latest.get('task'), latest.get('oracle_hint')
    memory, previous_action = latest.get('memory'), latest.get('action')
    if observation is None or action_space is None or task is None:
        raise ValueError('Prefix lacks an observation, action space, or task')
    if observation.step != step or action_space.step != step:
        raise ValueError(f'Latest observation is not from step {step}')
    if oracle is not None and oracle.step != step:
        raise ValueError(f'Latest oracle hint is not from step {step}')
    if step == 0:
        if memory is not None or previous_action is not None:
            raise ValueError('Step 0 prefix contains memory or actions')
    elif (memory is None or previous_action is None
          or memory.step != step - 1 or previous_action.step != step - 1):
        raise ValueError(f'Prefix lacks the memory and action of step {step - 1}')
    calls = sum(r['record'] == 'event' and r['kind'] == 'agent_call_end' for r in records)
    return StepState(step, observation, action_space, oracle, task, memory, previous_action,
                     last_output, calls)


def replay(environment: Environment, records: list[dict]) -> Frame:
    """Reset, then re-execute the prefix's recorded actions, checking every observation."""
    artifacts = _artifacts(records)
    observations = [a for a in artifacts if a.kind == 'observation']
    spaces = [a for a in artifacts if a.kind == 'action_space']
    actions = [a for a in artifacts if a.kind == 'action']
    # An injected environment fault corrupts what agents saw; the world produced the original.
    originals = {r['artifact_ids'][0]: r['details']['original'] for r in records
                 if r['record'] == 'event' and r['kind'] == 'fault_injected'
                 and r['details']['role'] == 'environment'}
    if not len(observations) == len(spaces) == len(actions) + 1:
        raise ValueError('Prefix observations and actions do not align')

    def check(frame: Frame, index: int):
        observation = observations[index]
        expected = originals.get(observation.id, observation.content)
        if frame.text != expected['text']:
            raise ReplayMismatch(f'Observation differs at step {index}')
        if list(frame.commands) != spaces[index].content['commands']:
            raise ReplayMismatch(f'Admissible commands differ at step {index}')
        if frame.done or frame.won:
            raise ReplayMismatch(f'Episode ended during replay at step {index}')

    frame = environment.reset()
    check(frame, 0)
    for index, action in enumerate(actions):
        frame = environment.step(action.content['command'])
        check(frame, index + 1)
    return frame


def restore(source: Path, step: int, trace: Trace, environment: Environment) -> StepState:
    """Copy `source`'s prefix before `step` into the fresh `trace`, replay the environment to
    the same world state, and return the agent state to resume from."""
    records = prefix(read_records(source), step)
    trace.extend(records)
    started = time.monotonic()
    replay(environment, records)
    replay_seconds = time.monotonic() - started
    state = rebuild_state(records, step)
    trace.event('state_restored', step, (state.observation.id,), source=str(source),
                mechanism='reset_replay', replayed_actions=step, replay_seconds=replay_seconds,
                prefix_agent_calls=state.agent_calls)
    return state
