import json
from pathlib import Path
import tempfile
import unittest

from agent_recovery import AgentTeam, Workflow
from agent_recovery.agents import (
    AdmissibilityVerifier, ConstraintsAgent, MemoryAgent, ObservationAgent, OraclePlanner)
from agent_recovery.contracts import AgentResult
from agent_recovery.injection import FaultSpec, Injector
from agent_recovery.recovery import ReplayMismatch, prefix, read_records, rebuild_state, restore
from agent_recovery.trace import Trace
from test_injection import RoomEnvironment
from test_workflow import FakeEnvironment
from verify_trace import verify


class StepPlanner:
    """Deterministic non-oracle planner for RoomEnvironment: look, look, finish."""
    def run(self, request):
        return AgentResult({'action': 'finish' if request.step == 2 else 'look'})


def oracle_team():
    return AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(),
                     OraclePlanner(), AdmissibilityVerifier())


def room_team():
    return AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(),
                     StepPlanner(), AdmissibilityVerifier())


def artifacts(path):
    return [r for r in read_records(path) if r['record'] == 'artifact']


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def record(self, name, team, environment, oracle=True, injector=None):
        path = self.root / name
        trace = Trace(path)
        try:
            Workflow(team, trace, oracle_hints=oracle, injector=injector).run(environment)
        finally:
            trace.close()
        return path

    def resume(self, source, step, name, team, environment, oracle=True):
        path = self.root / name
        trace = Trace(path)
        try:
            workflow = Workflow(team, trace, oracle_hints=oracle)
            state = restore(source, step, workflow, environment)
            won = workflow.resume(environment, state)
        finally:
            trace.close()
        return path, state, won, workflow

    def test_resume_reproduces_a_deterministic_episode_at_every_step(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        for step in (0, 1):
            with self.subTest(step=step):
                env = FakeEnvironment()
                path, state, won, workflow = self.resume(
                    source, step, f'resumed-{step}.jsonl', oracle_team(), env)
                self.assertTrue(won)
                self.assertEqual(env.actions, ['look', 'finish'])
                self.assertEqual(artifacts(path), artifacts(source))
                self.assertEqual(state.agent_calls, 5 * step)
                self.assertEqual(workflow.agent_calls, 10)
                verify(path)
                kinds = [r['kind'] for r in read_records(path) if r['record'] == 'event']
                self.assertEqual(kinds.count('state_restored'), 1)
                self.assertEqual(kinds.count('run_resumed'), 1)

    def test_rebuilt_state_points_at_previous_memory_and_action(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        records = read_records(source)
        state = rebuild_state(prefix(records, 1), 1)
        self.assertEqual((state.memory.step, state.previous_action.step), (0, 0))
        self.assertEqual(state.previous_action.content['command'], 'look')
        self.assertEqual((state.observation.step, state.action_space.step, state.oracle.step),
                         (1, 1, 1))
        self.assertEqual(set(state.last_output), {'observation', 'memory', 'constraints',
                                                  'planner', 'verifier'})
        first = rebuild_state(prefix(records, 0), 0)
        self.assertIsNone(first.memory)
        self.assertIsNone(first.previous_action)
        self.assertEqual(first.last_output, {})

    def test_resumed_agents_receive_the_restored_memory_and_action(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        team = oracle_team()
        requests = []
        inner = team.memory

        class Recording:
            def run(self, request):
                requests.append(request)
                return inner.run(request)
        team.memory = Recording()
        _, state, _, _ = self.resume(source, 1, 'resumed.jsonl', team, FakeEnvironment())
        self.assertEqual(requests[0].step, 1)
        self.assertEqual(requests[0].inputs['previous_memory'], state.memory)
        self.assertEqual(requests[0].inputs['previous_action'].content['command'], 'look')

    def test_replay_rejects_a_world_that_differs_from_the_recording(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())

        class Drifted(FakeEnvironment):
            def frame(self):
                frame = super().frame()
                if self.actions:
                    return frame.__class__(frame.text + ' Changed.', *[
                        getattr(frame, name) for name in
                        ('commands', 'gamefile', 'score', 'done', 'won', 'oracle_commands')])
                return frame
        trace = Trace(self.root / 'resumed.jsonl')
        self.addCleanup(trace.close)
        with self.assertRaisesRegex(ReplayMismatch, 'step 1'):
            restore(source, 1, Workflow(oracle_team(), trace), Drifted())

    def test_replay_compares_against_the_original_of_a_corrupted_observation(self):
        injector = Injector(FaultSpec(step=1, role='environment', operator='incorrect', seed=3))
        source = self.record('faulty.jsonl', room_team(), RoomEnvironment(), oracle=False,
                             injector=injector)
        self.assertTrue(injector.fired)
        # The agents saw a corrupted step-1 observation; the replayed world must match the
        # original, and the restored state keeps the corrupted artifact the agents saw.
        path, state, won, _ = self.resume(source, 2, 'resumed.jsonl', room_team(),
                                          RoomEnvironment(), oracle=False)
        self.assertTrue(won)
        verify(path)
        faulty = next(r for r in read_records(source) if r.get('kind') == 'fault_injected')
        self.assertIn(faulty['artifact_ids'][0], {a['id'] for a in artifacts(path)})

    def test_resume_requires_a_step_with_agent_calls(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        trace = Trace(self.root / 'resumed.jsonl')
        self.addCleanup(trace.close)
        with self.assertRaisesRegex(ValueError, 'no agent call at step 2'):
            restore(source, 2, Workflow(oracle_team(), trace), FakeEnvironment())

    def test_resume_rejects_injection_and_oracle_mismatch(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        trace = Trace(self.root / 'resumed.jsonl')
        self.addCleanup(trace.close)
        env = FakeEnvironment()
        injector = Injector(FaultSpec(step=1, role='planner', operator='incorrect'))
        with self.assertRaisesRegex(ValueError, 'Fault injection'):
            restore(source, 1, Workflow(oracle_team(), trace, injector=injector), env)
        state = restore(source, 1, Workflow(oracle_team(), trace), env)
        with self.assertRaisesRegex(ValueError, 'Fault injection'):
            Workflow(oracle_team(), trace, injector=injector).resume(env, state)
        team = oracle_team()
        team.planner = StepPlanner()
        with self.assertRaisesRegex(ValueError, 'oracle hints'):
            Workflow(team, trace, oracle_hints=False).resume(env, state)

    def test_extend_only_accepts_a_fresh_trace(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        trace = Trace(self.root / 'other.jsonl')
        self.addCleanup(trace.close)
        trace.add('test', 'test', (), 0, {})
        with self.assertRaisesRegex(ValueError, 'fresh trace'):
            trace.extend(prefix(read_records(source), 1))

    def test_resumed_trace_records_new_calls_separately(self):
        source = self.record('source.jsonl', oracle_team(), FakeEnvironment())
        path, _, _, _ = self.resume(source, 1, 'resumed.jsonl', oracle_team(), FakeEnvironment())
        records = read_records(path)
        resumed = next(r for r in records if r.get('kind') == 'run_resumed')
        end = records[-1]
        self.assertEqual(resumed['details']['prefix_agent_calls'], 5)
        self.assertEqual(end['details']['agent_calls'], 10)
        starts = [r for r in records if r.get('kind') == 'run_start']
        self.assertEqual(len(starts), 1)
        self.assertEqual(json.dumps(starts[0], sort_keys=True),
                         json.dumps(read_records(source)[0], sort_keys=True))


if __name__ == '__main__':
    unittest.main()
