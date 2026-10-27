"""Five replaceable roles, bounded replanning, and a deterministic execution gate."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import re
import time
from types import MappingProxyType

from .contracts import Agent, AgentRequest, AgentResult, OUTPUT_KINDS, ROLES, validate_result
from .environment import Environment, Frame
from .trace import Artifact, Trace, json_copy


@dataclass
class AgentTeam:
    observation: Agent
    memory: Agent
    constraints: Agent
    planner: Agent
    verifier: Agent

    def describe(self):
        return {role: f'{type(getattr(self, role)).__module__}:{type(getattr(self, role)).__qualname__}'
                for role in ROLES}


class Workflow:
    def __init__(self, team: AgentTeam, trace: Trace, seed=42, max_steps=150,
                 max_plan_attempts=2, oracle_hints=True, injector=None):
        if max_steps < 1 or max_plan_attempts < 1:
            raise ValueError('Step and plan-attempt limits must be positive')
        if not oracle_hints and getattr(team.planner, 'requires_oracle', False):
            raise ValueError('Configured planner requires oracle hints; replace it before disabling hints')
        self.team, self.trace = team, trace
        self.seed, self.max_steps = seed, max_steps
        self.max_plan_attempts, self.oracle_hints = max_plan_attempts, oracle_hints
        self.agent_calls = 0
        self.injector = injector
        self.last_output: dict[str, Artifact] = {}

    def call(self, role: str, step: int, inputs: dict[str, Artifact], attempt=0) -> Artifact:
        # Copies prevent a hook from changing artifacts used by other roles or stored in the DAG.
        detached = MappingProxyType(deepcopy(inputs))
        call_seed = int.from_bytes(hashlib.sha256(
            f'{self.seed}:{role}:{step}:{attempt}'.encode()).digest()[:4], 'big')
        request = AgentRequest(role, step, attempt, call_seed, detached)
        ids = tuple(value.id for value in inputs.values())
        named_inputs = {name: value.id for name, value in inputs.items()}
        self.trace.event('agent_call_start', step, ids, role=role, attempt=attempt,
                         seed=call_seed, inputs=named_inputs)
        started = time.monotonic()
        self.agent_calls += 1
        try:
            injector = self.injector
            previous = self.last_output.get(role)
            if injector and previous is not None and injector.times_out(role, step, attempt):
                # Simulated timeout; the workflow's policy reuses the role's last output.
                injector.fired = True
                self.trace.event('fault_injected', step, ids, role=role, attempt=attempt,
                                 fault=injector.spec.to_dict(), original=None,
                                 fallback=previous.id)
                ids, named_inputs = ids + (previous.id,), {**named_inputs, 'timeout_fallback': previous.id}
                # The wait is simulated, so it is reported as logical time rather than slept.
                content, metrics = previous.content, {'simulated_timeout_seconds': injector.timeout_seconds}
            else:
                result = getattr(self.team, role).run(request)
                if not isinstance(result, AgentResult):
                    raise TypeError(f'{role}: run() must return AgentResult')
                content, metrics = json_copy(result.content), result.metrics
                validate_result(role, content)
                if not isinstance(metrics, dict) or any(
                    not isinstance(key, str) or type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0 for key, value in metrics.items()
                ):
                    raise ValueError(f'{role}: metrics must map names to finite nonnegative numbers')
                faulty = injector.apply(role, step, attempt, content, inputs) if injector else None
                if faulty is not None:
                    validate_result(role, faulty)
                    self.trace.event('fault_injected', step, ids, role=role, attempt=attempt,
                                     fault=injector.spec.to_dict(), original=content,
                                     detail=injector.detail)
                    content = faulty
            if injector:
                injector.observe(role, content)
            output = self.trace.add(OUTPUT_KINDS[role], role, ids, step, content)
            self.last_output[role] = output
            self.trace.event('agent_call_end', step, (output.id,), role=role, attempt=attempt,
                             inputs=named_inputs, duration_seconds=time.monotonic() - started,
                             metrics=metrics)
            return output
        except Exception as exc:
            self.trace.event('agent_call_error', step, ids, role=role, attempt=attempt,
                             error_type=type(exc).__name__, message=str(exc),
                             duration_seconds=time.monotonic() - started)
            raise

    def snapshot(self, frame: Frame, step: int, parents=()):
        content = {'text': frame.text, 'gamefile': frame.gamefile}
        injector = self.injector
        faulty = injector.apply('environment', step, 0, content, {}) if injector else None
        # A faulty tool result: agents see the corrupted text; the executor still checks the
        # environment's own command list, and the original is kept in the trace.
        observation = self.trace.add('observation', 'environment', parents, step, faulty or content)
        if faulty is not None:
            self.trace.event('fault_injected', step, (observation.id,), role='environment', attempt=0,
                             fault=injector.spec.to_dict(), original=content, detail=injector.detail)
        if injector:
            injector.observe('environment', content)
        action_space = self.trace.add('action_space', 'environment', (observation.id,), step,
                                      {'commands': list(frame.commands)})
        oracle = None
        if self.oracle_hints:
            if frame.oracle_commands is None:
                raise ValueError('Oracle hints enabled but environment supplied none')
            oracle = self.trace.add('oracle_hint', 'environment_expert', (observation.id,), step,
                                    {'commands': list(frame.oracle_commands)})
        return observation, action_space, oracle

    def finish(self, step, won, reason):
        self.trace.event('run_end', step, won=won, reason=reason, agent_calls=self.agent_calls,
                         environment_actions=step,
                         tracking={'bytes': self.trace.bytes_written, 'records': self.trace.sequence
                                   + len(self.trace.artifacts), 'write_seconds': self.trace.write_seconds})
        return won

    def run(self, environment: Environment, configuration=None) -> bool:
        """Caller owns environment/trace cleanup. Every episode uses a fresh Workflow."""
        step = 0
        try:
            self.trace.event('run_start', 0, schema_version=2, seed=self.seed,
                             environment=environment.describe(), agents=self.team.describe(),
                             workflow=configuration or {}, oracle_hints=self.oracle_hints,
                             max_steps=self.max_steps, max_plan_attempts=self.max_plan_attempts)
            frame = environment.reset()
            observation, action_space, oracle = self.snapshot(frame, 0)
            match = re.search(r'Your task is to:\s*([^\n]+)', frame.text)
            if match is None:
                raise ValueError('Initial observation contains no task goal')
            task = self.trace.add('task', 'task_extractor', (observation.id,), 0,
                                  {'goal': match.group(1).strip()})
            memory = previous_action = None
            if frame.done or frame.won:
                return self.finish(0, frame.won, 'terminal')
            for step in range(self.max_steps):
                state = self.call('observation', step, {'observation': observation})
                memory_inputs = {'state': state}
                if memory is not None:
                    memory_inputs['previous_memory'] = memory
                if previous_action is not None:
                    memory_inputs['previous_action'] = previous_action
                memory = self.call('memory', step, memory_inputs)
                constraints = self.call('constraints', step,
                                        {'task': task, 'action_space': action_space})
                planner_inputs = {'state': state, 'memory': memory, 'constraints': constraints}
                if oracle is not None:
                    planner_inputs['oracle'] = oracle
                for attempt in range(self.max_plan_attempts):
                    plan = self.call('planner', step, planner_inputs, attempt)
                    # The raw observation lets a verifier check derived state against its source.
                    verification = self.call('verifier', step,
                                             {'plan': plan, 'state': state, 'constraints': constraints,
                                              'observation': observation}, attempt)
                    command = plan.content['action']
                    # Check the environment's command list independently of plugin-produced constraints.
                    admissible = command in action_space.content['commands']
                    approved = verification.content['approved'] and admissible
                    reason = verification.content['reason'] if admissible else 'Executor: inadmissible command'
                    gate = self.trace.add('execution_gate', 'deterministic_executor',
                                          (plan.id, verification.id, action_space.id), step,
                                          {'approved': approved, 'reason': reason,
                                           'action': command})
                    if approved:
                        break
                    self.trace.event('plan_rejected', step, (plan.id, verification.id, gate.id),
                                     attempt=attempt, reason=reason)
                    planner_inputs['feedback'] = gate
                    planner_inputs['previous_plan'] = plan
                else:
                    return self.finish(step, False, 'verification_rejected')
                action = self.trace.add('action', 'deterministic_executor',
                                        (plan.id, verification.id, gate.id), step,
                                        {'command': command})
                self.trace.event('action_requested', step, (action.id,), command=command)
                frame = environment.step(command)
                observation, action_space, oracle = self.snapshot(
                    frame, step + 1, (action.id, observation.id))
                self.trace.event('transition', step + 1, (action.id, observation.id),
                                 score=frame.score, done=frame.done, won=frame.won)
                previous_action = action
                if frame.done or frame.won:
                    return self.finish(step + 1, frame.won, 'terminal')
            return self.finish(self.max_steps, False, 'step_limit')
        except Exception as exc:
            self.trace.event('run_error', step, error_type=type(exc).__name__, message=str(exc))
            raise
