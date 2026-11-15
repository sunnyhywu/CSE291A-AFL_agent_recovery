"""Resume a recorded episode from the start of step k (state restore layers 1 and 2).

Layer 1, agent results and shared memory: artifacts are immutable, so the agent state before
step k is the trace prefix written before step k's first agent call. Rolling back means
pointing the workflow at those artifacts (`rebuild_state`).

Layer 2, a resettable deterministic environment: rolling the agents back does not roll the
world back. The environment is reset and the recorded actions 0..k-1 are re-executed without
querying any agent, and every replayed observation and command list is checked against the
recording (`replay`). A mismatch raises `ReplayMismatch` instead of resuming on a wrong world.

    workflow = Workflow(team, Trace(new_path), ...)
    state = restore(source_path, step, workflow, environment)
    workflow.resume(environment, state)

Selective recovery (`recover`) is given the identity of a faulty artifact in a recorded
episode. It rolls back to the start of the faulty artifact's step, then reruns the episode,
reusing a recorded agent output whenever the same call (role, step, attempt) receives inputs
with identical content and the recorded output is still valid:

* `cutoff` (influence-aware): only the faulty artifact is invalid. A recomputed output equal
  to the recorded one makes its consumers' inputs identical again, so they are reused and
  recomputation stops along that branch.
* `invalidate` (dependency-based): the faulty artifact and all of its recorded descendants
  are invalid and always recomputed.

Agents are stateless: an output depends only on the role, the call seed (derived from the
workflow seed, role, step and attempt) and the input contents, never on artifact IDs. With
`audit=True` every reused output is also recomputed and compared, at no counted cost.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

from .contracts import ROLES
from .environment import Environment, Frame
from .trace import Artifact
from .workflow import StepState, Workflow, extract_task

MODES = ('cutoff', 'invalidate')


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


def restore(source: Path, step: int, workflow: Workflow, environment: Environment,
            invalid=frozenset()) -> StepState:
    """Copy `source`'s prefix before `step` into the workflow's fresh trace, replay the
    environment to the same world state, and return the agent state to resume from.

    If the step's observation is in `invalid` (a corrupted tool result), it is replaced by a
    fresh snapshot of the replayed world, together with its action space and oracle hint.
    """
    if workflow.injector is not None:
        raise ValueError('Fault injection is not supported on a resumed run')
    trace = workflow.trace
    records = prefix(read_records(source), step)
    trace.extend(records)
    started = time.monotonic()
    frame = replay(environment, records)
    replay_seconds = time.monotonic() - started
    state = rebuild_state(records, step)
    if state.observation.id in invalid:
        observation, action_space, oracle = workflow.snapshot(
            frame, step, state.observation.inputs)
        task = state.task
        if step == 0:
            task = trace.add('task', 'task_extractor', (observation.id,), 0,
                             {'goal': extract_task(frame.text)})
        trace.event('observation_refreshed', step, (state.observation.id, observation.id))
        state = replace(state, observation=observation, action_space=action_space,
                        oracle=oracle, task=task)
    trace.event('state_restored', step, (state.observation.id,), source=str(source),
                mechanism='reset_replay', replayed_actions=step, replay_seconds=replay_seconds,
                prefix_agent_calls=state.agent_calls)
    return state


def _signature(inputs: dict) -> dict:
    return {name: (artifact.kind, json.dumps(artifact.content, sort_keys=True))
            for name, artifact in inputs.items()}


def descendants(records: list[dict], artifact_id: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for artifact in _artifacts(records):
        for parent in artifact.inputs:
            children.setdefault(parent, []).append(artifact.id)
    found, pending = set(), list(children.get(artifact_id, ()))
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending.extend(children.get(node, ()))
    return found


class SelectiveReuse:
    """Answers the workflow's reuse lookups from a recorded episode."""

    def __init__(self, records: list[dict], invalid: set[str], audit=False):
        artifacts = {a.id: a for a in _artifacts(records)}
        self.calls = {}
        for record in records:
            if record['record'] == 'event' and record['kind'] == 'agent_call_end':
                details = record['details']
                inputs = {name: artifacts[i] for name, i in details['inputs'].items()}
                self.calls[details['role'], record['step'], details['attempt']] = (
                    _signature(inputs), artifacts[record['artifact_ids'][0]])
        self.invalid, self.audit = set(invalid), audit

    def lookup(self, role, step, attempt, inputs) -> Artifact | None:
        recorded = self.calls.get((role, step, attempt))
        if recorded is None:
            return None
        signature, output = recorded
        if output.id in self.invalid or signature != _signature(inputs):
            return None
        return output


def faulty_artifact(records: list[dict]) -> str | None:
    """ID of the artifact a recorded injected fault produced (the identity given to recovery)."""
    for index, record in enumerate(records):
        if record['record'] == 'event' and record['kind'] == 'fault_injected':
            if record['details']['role'] == 'environment':
                return record['artifact_ids'][0]
            return next(r['artifact_ids'][0] for r in records[index:]
                        if r['record'] == 'event' and r['kind'] == 'agent_call_end')
    return None


def recover(source: Path, faulty_id: str, workflow: Workflow, environment: Environment,
            mode='cutoff', audit=False, configuration=None) -> bool:
    """Selectively recover `source` given its faulty artifact; returns the recovered success."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}')
    records = read_records(source)
    artifacts = {a.id: a for a in _artifacts(records)}
    if faulty_id not in artifacts:
        raise ValueError(f'Unknown faulty artifact: {faulty_id}')
    faulty = artifacts[faulty_id]
    if faulty.producer not in ROLES and faulty.kind != 'observation':
        raise ValueError('The faulty artifact must be an agent output or an observation')
    recorded = records[0]['details']
    expected = {'seed': workflow.seed, 'agents': workflow.team.describe(),
                'oracle_hints': workflow.oracle_hints,
                'max_plan_attempts': workflow.max_plan_attempts}
    mismatched = [key for key, value in expected.items() if recorded.get(key) != value]
    if mismatched:
        # Reused outputs are only valid for the same agents, seeds and attempt structure.
        raise ValueError(f'Recovery workflow differs from the recorded run: {", ".join(mismatched)}')
    invalid = {faulty_id}
    if mode == 'invalidate':
        invalid |= descendants(records, faulty_id)
    state = restore(source, faulty.step, workflow, environment, invalid)
    workflow.reuse = SelectiveReuse(records, invalid, audit)
    workflow.trace.event('recovery_started', faulty.step, mode=mode, source=str(source),
                         faulty_artifact=faulty_id, invalidated=len(invalid), audit=audit)
    return workflow.resume(environment, state, configuration)


def summarize(path: Path) -> dict:
    """Work retained and recomputed by one resumed or recovered episode."""
    records = read_records(path)
    resumed = next(r for r in records if r['record'] == 'event' and r['kind'] == 'run_resumed')
    calls = [r for r in records if r['record'] == 'event' and r['kind'] == 'agent_call_end'
             and r['sequence'] > resumed['sequence']]
    reused = [c for c in calls if 'reused_from' in c['details']]
    recomputed = [c for c in calls if 'reused_from' not in c['details']]
    end = records[-1]['details'] if records[-1].get('kind') == 'run_end' else {}
    return {
        'won': end.get('won'),
        'prefix_calls': resumed['details']['prefix_agent_calls'],
        'reused_calls': len(reused),
        'recomputed_calls': len(recomputed),
        'recomputed_by_role': dict(Counter(c['details']['role'] for c in recomputed)),
        'recomputed_tokens': sum(c['details']['metrics'].get('input_tokens', 0)
                                 + c['details']['metrics'].get('output_tokens', 0)
                                 for c in recomputed),
        'audit_mismatches': sum(c['details'].get('audit_match') is False for c in reused),
    }
