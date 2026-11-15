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

The recovered run acts on the world through `world.RecordedWorld`, which restores the real
world (reset and replay, or compensation with inverse actions) only when a recovered action
first differs from the recorded one, and reports irreversible effects.
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
from .world import (MECHANISMS, RecordedWorld, ReplayMismatch, RestoreFailed,  # noqa: F401
                    artifacts as _artifacts, check, recording)

MODES = ('cutoff', 'invalidate')


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
    record = recording(records)
    frame = environment.reset()
    check(frame, record.frames[0], 0)
    for index, command in enumerate(record.actions):
        frame = environment.step(command)
        check(frame, record.frames[index + 1], index + 1)
    return frame


def restore_agents(records: list[dict], step: int, workflow: Workflow) -> StepState:
    """Layer 1: copy the prefix before `step` into the workflow's fresh trace and rebuild the
    agent state from it."""
    if workflow.injector is not None:
        raise ValueError('Fault injection is not supported on a resumed run')
    records = prefix(records, step)
    workflow.trace.extend(records)
    return rebuild_state(records, step)


def restore(source: Path, step: int, workflow: Workflow, environment: Environment) -> StepState:
    """Restore the agent state before `step` and eagerly reset and replay the environment to the
    same world state; returns the state to resume from."""
    records = read_records(source)
    state = restore_agents(records, step, workflow)
    started = time.monotonic()
    replay(environment, prefix(records, step))
    workflow.trace.event('state_restored', step, (state.observation.id,), source=str(source),
                         mechanism='reset_replay', replayed_actions=step,
                         replay_seconds=time.monotonic() - started,
                         prefix_agent_calls=state.agent_calls)
    return state


def refresh(workflow: Workflow, state: StepState, frame: Frame) -> StepState:
    """Replace a corrupted observation (and what derives from it) with a fresh observation."""
    trace, step = workflow.trace, state.step
    observation, action_space, oracle = workflow.snapshot(frame, step, state.observation.inputs)
    task = state.task
    if step == 0:
        task = trace.add('task', 'task_extractor', (observation.id,), 0,
                         {'goal': extract_task(frame.text)})
    trace.event('observation_refreshed', step, (state.observation.id, observation.id))
    return replace(state, observation=observation, action_space=action_space, oracle=oracle,
                   task=task)


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
            mode='cutoff', world='reset', position: int | None = None, audit=False,
            configuration=None) -> bool:
    """Selectively recover `source` given its faulty artifact; returns the recovered success.

    `world` picks how the real world is restored at the first divergent action ('reset',
    'compensate' or 'auto', see `agent_recovery.world`). For compensation, `position` is how
    many recorded actions `environment` has executed (default: all, i.e. recovery starts after
    the faulty episode ended).
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}')
    if world not in MECHANISMS:
        raise ValueError(f'world must be one of {", ".join(MECHANISMS)}')
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
    world_view = RecordedWorld(environment, recording(records), workflow.trace, world, position)
    state = restore_agents(records, faulty.step, workflow)
    world_view.start(faulty.step)
    workflow.trace.event('recovery_started', faulty.step, mode=mode, world=world,
                         source=str(source), faulty_artifact=faulty_id, invalidated=len(invalid),
                         audit=audit, prefix_agent_calls=state.agent_calls)
    if state.observation.id in invalid:
        try:
            state = refresh(workflow, state, world_view.observe(faulty.step))
        except Exception as exc:
            workflow.trace.event('run_error', faulty.step, error_type=type(exc).__name__,
                                 message=str(exc))
            raise
    workflow.reuse = SelectiveReuse(records, invalid, audit)
    return workflow.resume(world_view, state, configuration)


def summarize(path: Path) -> dict:
    """Work retained and recomputed, and world restoration, of one recovered episode.

    `complete_recovery` is conservative: the task succeeded and no irreversible effect was
    left behind. Whether a reported irreversible effect was harmless (e.g. the task needed it
    anyway) is for the analysis to decide, not for recovery to assume.
    """
    records = read_records(path)
    events = [r for r in records if r['record'] == 'event']
    start = next(e for e in events if e['kind'] in ('recovery_started', 'run_resumed'))
    calls = [e for e in events if e['kind'] == 'agent_call_end' and e['sequence'] > start['sequence']]
    reused = [c for c in calls if 'reused_from' in c['details']]
    recomputed = [c for c in calls if 'reused_from' not in c['details']]
    restores = [e['details'] for e in events if e['kind'] == 'world_restored']
    irreversible = [command for r in restores for command in r['irreversible']]
    end = events[-1]['details'] if events[-1]['kind'] == 'run_end' else {}
    won = end.get('won')
    return {
        'won': won,
        'complete_recovery': bool(won) and not irreversible,
        'prefix_calls': start['details']['prefix_agent_calls'],
        'reused_calls': len(reused),
        'recomputed_calls': len(recomputed),
        'recomputed_by_role': dict(Counter(c['details']['role'] for c in recomputed)),
        'recomputed_tokens': sum(c['details']['metrics'].get('input_tokens', 0)
                                 + c['details']['metrics'].get('output_tokens', 0)
                                 for c in recomputed),
        'audit_mismatches': sum(c['details'].get('audit_match') is False for c in reused),
        'world_restores': restores,
        'world_actions': sum(r['environment_actions'] for r in restores),
        'irreversible_effects': irreversible,
        'fallbacks': sum(e['kind'] == 'world_restore_fallback' for e in events),
        'restore_failed': any(e['kind'] == 'world_restore_failed' for e in events),
    }
