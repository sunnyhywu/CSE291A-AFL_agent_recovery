"""World-state restoration for a recovered episode (state restore layers 2, 4 and 5).

Rolling the agents back to step k does not roll back the world they acted on. A recovered run
therefore sees the world through `RecordedWorld`, which restores it lazily:

* While the recovered actions equal the recorded ones, the recorded observations are what the
  world would return, so they are served from the recording and the real world is not touched.
  A fault that never changes an executed action costs no world restoration at all.
* At the first divergent action d, the real world is brought to its state before d, and from
  then on every action goes to the real environment.

Mechanisms for bringing the world to the state after j recorded actions:

* `reset` (layer 2, a resettable deterministic environment): reset, then re-execute the
  recorded actions 0..j-1, checking each observation and command list against the recording.
* `compensate` (layer 4, a live environment that cannot be reset): undo the recorded actions
  after j, newest first, with registered inverse actions (`policy.inverse`), then check the
  world's fingerprint. An environment that is behind j is moved forward by re-executing the
  recorded actions.
* `auto`: compensate, and fall back to reset if compensation is impossible or its fingerprint
  does not match.

Layer 5: an action with no inverse (heat, cool, clean, slice) cannot be compensated. It is
skipped and reported as an irreversible effect, never hidden, and the recovered episode is not
counted as a complete recovery. A reset undoes it only because this world is a simulator.

The fingerprint is the environment's admissible-command list. In ALFWorld it reflects the
agent's location, what it holds and whether the receptacle in front of it is open. It does not
capture object states elsewhere, such as a heated object, which is why irreversible actions
are reported explicitly rather than inferred from the fingerprint.
"""
from dataclasses import dataclass
import re
import time

from .environment import Environment, Frame
from .policy import effect, inverse
from .trace import Artifact, Trace

MECHANISMS = ('reset', 'compensate', 'auto')


class ReplayMismatch(RuntimeError):
    """The environment did not reproduce the recorded world state."""


class RestoreFailed(RuntimeError):
    """The world could not be brought back to the required recorded state."""

    def __init__(self, message: str, spent: int = 0):
        super().__init__(message)
        self.spent = spent  # environment actions already executed by the failed attempt


def artifacts(records: list[dict]) -> list[Artifact]:
    return [Artifact(r['id'], r['kind'], r['producer'], tuple(r['inputs']), r['step'], r['content'])
            for r in records if r['record'] == 'artifact']


@dataclass(frozen=True)
class Recording:
    """The recorded world history: frames[j] is what the world showed after j actions."""
    actions: tuple[str, ...]
    frames: tuple[Frame, ...]


def recording(records: list[dict]) -> Recording:
    items = artifacts(records)
    observations = [a for a in items if a.kind == 'observation']
    spaces = [a for a in items if a.kind == 'action_space']
    oracles = [a for a in items if a.kind == 'oracle_hint']
    actions = [a.content['command'] for a in items if a.kind == 'action']
    transitions = [r['details'] for r in records
                   if r['record'] == 'event' and r['kind'] == 'transition']
    # An injected environment fault corrupts what agents saw; the world produced the original.
    originals = {r['artifact_ids'][0]: r['details']['original'] for r in records
                 if r['record'] == 'event' and r['kind'] == 'fault_injected'
                 and r['details']['role'] == 'environment'}
    if not len(observations) == len(spaces) == len(actions) + 1 == len(transitions) + 1:
        raise ValueError('Recorded observations and actions do not align')
    if oracles and len(oracles) != len(observations):
        raise ValueError('Recorded oracle hints do not align with observations')
    frames = []
    for j, observation in enumerate(observations):
        content = originals.get(observation.id, observation.content)
        outcome = transitions[j - 1] if j else {'score': 0.0, 'done': False, 'won': False}
        frames.append(Frame(content['text'], tuple(spaces[j].content['commands']),
                            content['gamefile'], outcome['score'], outcome['done'], outcome['won'],
                            tuple(oracles[j].content['commands']) if oracles else None))
    return Recording(tuple(actions), tuple(frames))


def check(frame: Frame, expected: Frame, step: int):
    """A re-executed step must reproduce the recorded observation and command list."""
    if frame.text != expected.text:
        raise ReplayMismatch(f'Observation differs at step {step}')
    if list(frame.commands) != list(expected.commands):
        raise ReplayMismatch(f'Admissible commands differ at step {step}')
    if (frame.done, frame.won) != (expected.done, expected.won):
        raise ReplayMismatch(f'Episode status differs at step {step}')


def location_before(actions, index: int) -> str | None:
    """Where the agent stood before actions[index]: the target of the last earlier 'go to'."""
    for command in reversed(actions[:index]):
        match = re.match(r'^go to (.+)$', command)
        if match:
            return match.group(1)
    return None


class RecordedWorld:
    """The environment a recovered run acts on; restores the real world at the first divergence.

    `position` is how many recorded actions the real environment has already executed (all of
    them when recovery starts after the faulty episode ended). It is only used by compensation.
    """

    def __init__(self, environment: Environment, record: Recording, trace: Trace,
                 mechanism='reset', position: int | None = None):
        if mechanism not in MECHANISMS:
            raise ValueError(f'mechanism must be one of {", ".join(MECHANISMS)}')
        self.environment, self.record, self.trace = environment, record, trace
        self.mechanism = mechanism
        self.position = len(record.actions) if position is None else position
        if not 0 <= self.position <= len(record.actions):
            raise ValueError('position must be within the recorded actions')
        self.step_index = 0       # actions the recovered run has executed so far
        self.live = False         # True once the real world matches the recovered run
        self.restores: list[dict] = []

    def describe(self):
        return self.environment.describe()

    def close(self):
        self.environment.close()

    def start(self, step: int):
        """The recovered run resumes at the start of `step`."""
        self.step_index = step

    def step(self, command: str) -> Frame:
        j = self.step_index
        if not self.live:
            if j < len(self.record.actions) and command == self.record.actions[j]:
                self.step_index += 1
                return self.record.frames[j + 1]
            self.materialize(j)
        self.step_index += 1
        return self.environment.step(command)

    def observe(self, step: int) -> Frame:
        """A fresh observation of the real world after `step` recorded actions, used to replace
        a corrupted tool result: restore the state before the last action and execute it again."""
        if step == 0:
            if self.mechanism == 'compensate':
                self.trace.event('world_restore_failed', 0, mechanism='compensate',
                                 reason='the initial observation can only be re-observed by a reset',
                                 environment_actions=0)
                raise RestoreFailed('The initial observation can only be re-observed by a reset')
            frame = self.environment.reset()
            self._log('reset', 0, 0, 1, [], True)
        else:
            self.materialize(step - 1)
            frame = self.environment.step(self.record.actions[step - 1])
        self.position, self.live = step, True
        return frame

    def materialize(self, j: int):
        """Bring the real world to its recorded state after j actions."""
        if self.mechanism in ('compensate', 'auto'):
            try:
                self._compensate(j)
                self.live = True
                return
            except RestoreFailed as failure:
                kind = 'world_restore_failed' if self.mechanism == 'compensate' else 'world_restore_fallback'
                self.trace.event(kind, j, mechanism='compensate', reason=str(failure),
                                 environment_actions=failure.spent)
                if self.mechanism == 'compensate':
                    raise
        self._reset_replay(j)
        self.live = True

    def _reset_replay(self, j: int):
        started = time.monotonic()
        frames, actions = self.record.frames, self.record.actions
        check(self.environment.reset(), frames[0], 0)
        for i in range(j):
            check(self.environment.step(actions[i]), frames[i + 1], i + 1)
        self.position = j
        self._log('reset', j, None, j + 1, [], True, time.monotonic() - started)

    def _compensate(self, j: int):
        started = time.monotonic()
        actions, frames = self.record.actions, self.record.frames
        origin, spent, irreversible = self.position, 0, []
        commands = list(frames[origin].commands)
        # Behind the target: re-execute recorded actions forward (no reset needed).
        for i in range(origin, j):
            frame = self.environment.step(actions[i])
            check(frame, frames[i + 1], i + 1)
            commands, spent = list(frame.commands), spent + 1
        # Ahead of the target: undo the recorded actions after j, newest first.
        for i in range(origin - 1, j - 1, -1):
            kind = effect(actions[i])
            if kind == 'none':
                continue
            if kind == 'irreversible':
                irreversible.append(actions[i])
                continue
            undo = inverse(actions[i], location_before(actions, i))
            if undo is None or undo not in commands:
                raise RestoreFailed(f'No admissible inverse for {actions[i]!r} at step {i}', spent)
            commands, spent = list(self.environment.step(undo).commands), spent + 1
        matched = sorted(commands) == sorted(frames[j].commands)
        self.position = j
        self._log('compensate', j, origin, spent, irreversible, matched,
                  time.monotonic() - started)
        if not matched:
            # Already logged with its cost by _log; spent=0 avoids counting it twice.
            raise RestoreFailed(f'World fingerprint differs from step {j} after compensation')

    def _log(self, mechanism, target, origin, spent, irreversible, matched, seconds=0.0):
        restore = {'mechanism': mechanism, 'to_step': target, 'from_step': origin,
                   'environment_actions': spent, 'irreversible': irreversible,
                   'fingerprint_match': matched, 'seconds': seconds}
        self.restores.append(restore)
        self.trace.event('world_restored', target, **restore)
