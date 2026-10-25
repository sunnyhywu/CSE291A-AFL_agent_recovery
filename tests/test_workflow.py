import json
from pathlib import Path
import tempfile
import unittest

from agent_recovery import AgentRequest, AgentResult, AgentTeam, Workflow
from agent_recovery.agents import (
    ObservationAgent, MemoryAgent, ConstraintsAgent, OraclePlanner, AdmissibilityVerifier)
from agent_recovery.environment import Frame
from agent_recovery.loading import load_team
from agent_recovery.trace import Trace


class FakeEnvironment:
    def __init__(self, oracle=True):
        self.actions = []
        self.oracle = oracle

    def describe(self):
        return {'backend': 'fake', 'oracle_hints': self.oracle}

    def frame(self):
        count = len(self.actions)
        return Frame('Your task is to: finish.\nRoom contents.', ('look', 'finish'), 'fake-game',
                     score=float(count == 2), done=count == 2, won=count == 2,
                     oracle_commands=('look' if count == 0 else 'finish',) if self.oracle else None)

    def reset(self):
        self.actions = []
        return self.frame()

    def step(self, command):
        self.actions.append(command)
        return self.frame()

    def close(self):
        pass


class FixedAgent:
    def __init__(self, content):
        self.content = content

    def run(self, request):
        return AgentResult(self.content)


class RetryPlanner:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return AgentResult({'action': 'look' if request.attempt else 'nonexistent command'})


class RecordingAgent:
    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def run(self, request):
        self.requests.append(request)
        return self.inner.run(request)


def team():
    return AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(),
                     OraclePlanner(), AdmissibilityVerifier())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'trace.jsonl'
        self.trace = Trace(self.path)
        self.addCleanup(self.trace.close)

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_five_roles_branch_and_carry_memory_across_steps(self):
        agents = team()
        for role in agents.describe():
            setattr(agents, role, RecordingAgent(getattr(agents, role)))
        env = FakeEnvironment()
        workflow = Workflow(agents, self.trace)
        self.assertTrue(workflow.run(env))
        self.assertEqual(env.actions, ['look', 'finish'])
        self.assertEqual(workflow.agent_calls, 10)
        self.assertEqual(set(agents.planner.requests[0].inputs),
                         {'state', 'memory', 'constraints', 'oracle'})
        for role in ('observation', 'memory', 'constraints', 'verifier'):
            self.assertNotIn('oracle', getattr(agents, role).requests[0].inputs)
        state = next(a for a in self.trace.artifacts.values() if a.kind == 'state')
        constraints = next(a for a in self.trace.artifacts.values() if a.kind == 'constraints')
        self.assertNotIn(constraints.id, self.trace.descendants(state.id))
        memory_inputs = agents.memory.requests[1].inputs
        self.assertEqual(memory_inputs['previous_action'].content['command'], 'look')
        self.assertIn('previous_memory', memory_inputs)
        for action in (a for a in self.trace.artifacts.values() if a.kind == 'action'):
            kinds = [self.trace.artifacts[i].kind for i in action.inputs]
            self.assertEqual(kinds, ['plan', 'verification', 'execution_gate'])

    def test_rejected_plans_reach_planner_feedback_without_changing_world(self):
        planner = RetryPlanner()
        agents = team()
        agents.planner = planner
        env = FakeEnvironment(oracle=False)
        workflow = Workflow(agents, self.trace, max_steps=1, oracle_hints=False)
        self.assertFalse(workflow.run(env))  # One valid action, then our step limit.
        self.assertEqual(env.actions, ['look'])
        self.assertEqual(workflow.agent_calls, 7)  # Three upstream + two planner/verifier calls.
        feedback = planner.requests[1].inputs['feedback']
        self.assertFalse(feedback.content['approved'])
        self.assertIn('previous_plan', planner.requests[1].inputs)
        self.assertFalse(any(a.kind == 'oracle_hint' for a in self.trace.artifacts.values()))
        self.assertEqual(sum(r.get('kind') == 'plan_rejected' for r in self.records()), 1)

    def test_executor_checks_raw_commands_even_if_two_agents_approve_bad_action(self):
        agents = team()
        agents.planner = FixedAgent({'action': 'delete everything'})
        agents.constraints = FixedAgent({'goal': 'finish', 'allowed_actions': ['delete everything']})
        agents.verifier = FixedAgent({'approved': True, 'reason': 'Approved'})
        env = FakeEnvironment(oracle=False)
        workflow = Workflow(agents, self.trace, max_plan_attempts=2, oracle_hints=False)
        self.assertFalse(workflow.run(env))
        self.assertEqual(env.actions, [])
        self.assertFalse(any(a.kind == 'action' for a in self.trace.artifacts.values()))
        self.assertEqual(self.records()[-1]['details']['reason'], 'verification_rejected')

    def test_verifier_can_reject_an_admissible_plan(self):
        agents = team()
        agents.verifier = FixedAgent({'approved': False, 'reason': 'Conflicts with task'})
        env = FakeEnvironment()
        self.assertFalse(Workflow(agents, self.trace, max_plan_attempts=1).run(env))
        self.assertEqual(env.actions, [])

    def test_invalid_hook_output_logs_error_before_any_action(self):
        agents = team()
        agents.verifier = FixedAgent({'approved': 'yes', 'reason': 'Wrong type'})
        env = FakeEnvironment()
        with self.assertRaisesRegex(ValueError, 'approved must be bool'):
            Workflow(agents, self.trace).run(env)
        self.assertEqual(env.actions, [])
        self.assertEqual(self.records()[-1]['kind'], 'run_error')
        self.assertTrue(any(r.get('kind') == 'agent_call_error' for r in self.records()))

    def test_hooks_cannot_mutate_stored_artifacts(self):
        class MutatingObservation:
            def run(self, request):
                request.inputs['observation'].content['text'] = 'Changed in hook'
                return AgentResult({'summary': 'Changed in hook', 'facts': []})
        agents = team()
        agents.observation = MutatingObservation()
        Workflow(agents, self.trace).run(FakeEnvironment())
        original = next(a for a in self.trace.artifacts.values() if a.kind == 'observation')
        self.assertIn('Your task is to:', original.content['text'])

    def test_yaml_factory_override_and_oracle_guard(self):
        config = Path(__file__).resolve().parents[1] / 'configs/agents.yaml'
        with self.assertRaisesRegex(ValueError, 'Replace OraclePlanner'):
            load_team(config, oracle_override=False)
        agents, effective = load_team(config,
                                     ['planner=examples.my_agents:LookPlanner'], False)
        self.assertFalse(effective['oracle_hints'])
        self.assertEqual(agents.planner.run(AgentRequest('planner', 0, 0, 42, {})).content['action'],
                         'look')
        with self.assertRaisesRegex(ValueError, '--agent must'):
            load_team(config, ['unknown=x:y'])

    def test_trace_detaches_result_content(self):
        content = {'nested': ['original']}
        artifact = self.trace.add('test', 'test', (), 0, content)
        content['nested'].append('changed')
        self.assertEqual(artifact.content, {'nested': ['original']})


if __name__ == '__main__':
    unittest.main()
