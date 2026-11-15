from pathlib import Path
import tempfile
import unittest

from agent_recovery import AgentTeam, Workflow
from agent_recovery.agents import (
    AdmissibilityVerifier, ConstraintsAgent, MemoryAgent, ObservationAgent)
from agent_recovery.contracts import AgentResult
from agent_recovery.environment import Frame
from agent_recovery.injection import FaultSpec, Injector
from agent_recovery.recovery import faulty_artifact, read_records, recover, summarize
from agent_recovery.trace import Trace
from verify_trace import verify

TEXT = 'You are at countertop 1. On the countertop 1, you see a apple 1, a bread 1, and a knife 2.'
COMMANDS = ('look', 'take apple 1 from countertop 1', 'go to fridge 1')


class KitchenEnvironment:
    """The world shows the last command and what is held, so a different action diverges it.
    Three actions end the episode; it is won only if the apple was never taken."""
    def describe(self):
        return {'backend': 'fake-kitchen'}

    def frame(self):
        text = f'{TEXT}\nLast: {self.last}. Holding: {self.holding}.'
        if not self.count:
            text += '\nYour task is to: go to the fridge without taking anything.'
        done = self.count == 3
        return Frame(text, COMMANDS, 'kitchen', score=float(done and not self.holding),
                     done=done, won=done and not self.holding)

    def reset(self):
        self.count, self.last, self.holding = 0, 'none', False
        return self.frame()

    def step(self, command):
        self.count, self.last = self.count + 1, command
        self.holding = self.holding or command.startswith('take')
        return self.frame()


class KitchenPlanner:
    def run(self, request):
        return AgentResult({'action': 'go to fridge 1' if request.step == 2 else 'look'})


def team():
    return AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(),
                     KitchenPlanner(), AdmissibilityVerifier())


def outputs(path):
    """(role, step, attempt) -> output content of every agent call in a trace."""
    records = read_records(path)
    contents = {r['id']: r['content'] for r in records if r['record'] == 'artifact'}
    return {(r['details']['role'], r['step'], r['details']['attempt']):
            contents[r['artifact_ids'][0]]
            for r in records if r['record'] == 'event' and r['kind'] == 'agent_call_end'}


def actions(path):
    return [r['content']['command'] for r in read_records(path)
            if r['record'] == 'artifact' and r['kind'] == 'action']


class SelectiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.clean = self.run_episode('clean.jsonl')

    def run_episode(self, name, fault=None):
        path = self.root / name
        trace = Trace(path)
        injector = Injector(FaultSpec(**fault)) if fault else None
        try:
            Workflow(team(), trace, oracle_hints=False, injector=injector).run(KitchenEnvironment())
        finally:
            trace.close()
        if injector:
            self.assertTrue(injector.fired)
        return path

    def recover(self, source, name, mode='cutoff', audit=False, **workflow):
        path = self.root / name
        trace = Trace(path)
        try:
            workflow = Workflow(team(), trace, oracle_hints=False, **workflow)
            faulty = faulty_artifact(read_records(source))
            won = recover(source, faulty, workflow, KitchenEnvironment(), mode=mode, audit=audit)
        finally:
            trace.close()
        verify(path, require_success=False)
        return path, won

    def test_recovered_outputs_match_the_clean_run_for_every_fault_stage(self):
        faults = {'environment': dict(step=1, role='environment', operator='incorrect', seed=1),
                  'observation': dict(step=1, role='observation', operator='contradictory'),
                  'memory': dict(step=1, role='memory', operator='incorrect', seed=2),
                  'planner': dict(step=1, role='planner', operator='incorrect', seed=1)}
        for stage, fault in faults.items():
            source = self.run_episode(f'{stage}.jsonl', fault)
            self.assertNotEqual(outputs(source), outputs(self.clean))
            for mode in ('cutoff', 'invalidate'):
                with self.subTest(stage=stage, mode=mode):
                    path, won = self.recover(source, f'{stage}-{mode}.jsonl', mode)
                    self.assertTrue(won)
                    self.assertEqual(outputs(path), outputs(self.clean))
                    self.assertEqual(actions(path), actions(self.clean))

    def test_cutoff_reuses_unaffected_calls_that_invalidation_recomputes(self):
        source = self.run_episode('observation.jsonl',
                                  dict(step=1, role='observation', operator='contradictory'))
        cutoff = summarize(self.recover(source, 'cutoff.jsonl', 'cutoff')[0])
        invalidate = summarize(self.recover(source, 'invalidate.jsonl', 'invalidate')[0])
        # Same total work: the five step-0 calls are kept as prefix, ten calls follow.
        for summary in (cutoff, invalidate):
            self.assertEqual(summary['prefix_calls'], 5)
            self.assertEqual(summary['reused_calls'] + summary['recomputed_calls'], 10)
        # The actions never diverge, so step 2's observation agent sees the same world and
        # the step-2 verifier sees an identical (recomputed) plan; only memory carries the fault.
        self.assertEqual(cutoff['recomputed_by_role'],
                         {'observation': 1, 'memory': 2, 'planner': 2, 'verifier': 1})
        self.assertEqual(cutoff['reused_calls'], 4)
        # Structurally, everything after the faulty state descends from it except the
        # step-1 constraints, which read only the task and the action space.
        self.assertEqual(invalidate['reused_calls'], 1)
        self.assertGreater(cutoff['reused_calls'], invalidate['reused_calls'])

    def test_world_divergence_is_rolled_back_and_recomputed(self):
        source = self.run_episode('planner.jsonl',
                                  dict(step=1, role='planner', operator='incorrect', seed=1))
        self.assertIn('take apple 1 from countertop 1', actions(source))
        self.assertEqual(read_records(source)[-1]['details']['won'], False)
        path, won = self.recover(source, 'recovered.jsonl')
        self.assertTrue(won)
        summary = summarize(path)
        # After the faulty action the recorded world differs, so the recorded step-2 calls
        # saw different observations and cannot be reused.
        self.assertEqual(summary['recomputed_by_role'].get('observation'), 1)

    def test_environment_fault_refreshes_the_corrupted_observation(self):
        source = self.run_episode('environment.jsonl',
                                  dict(step=1, role='environment', operator='incorrect', seed=1))
        path, _ = self.recover(source, 'recovered.jsonl')
        kinds = [r['kind'] for r in read_records(path) if r['record'] == 'event']
        self.assertEqual(kinds.count('observation_refreshed'), 1)
        self.assertEqual(summarize(path)['recomputed_by_role'].get('observation'), 1)

    def test_audit_confirms_every_reused_output(self):
        source = self.run_episode('memory.jsonl',
                                  dict(step=1, role='memory', operator='incorrect', seed=2))
        path, _ = self.recover(source, 'audited.jsonl', audit=True)
        reused = [r for r in read_records(path) if r.get('kind') == 'agent_call_end'
                  and 'reused_from' in r['details']]
        self.assertTrue(reused)
        self.assertTrue(all(r['details']['audit_match'] for r in reused))
        self.assertEqual(summarize(path)['audit_mismatches'], 0)

    def test_audit_reports_a_reused_output_that_no_longer_holds(self):
        source = self.run_episode('memory.jsonl',
                                  dict(step=1, role='memory', operator='incorrect', seed=2))

        class ChangedConstraints:
            def run(self, request):
                return AgentResult({'goal': 'changed', 'allowed_actions': list(COMMANDS)})
        path = self.root / 'drift.jsonl'
        trace = Trace(path)
        agents = team()
        workflow = Workflow(agents, trace, oracle_hints=False)
        # Same recorded agent identity, but the constraints agent now behaves differently.
        agents.constraints.run = ChangedConstraints().run
        recover(source, faulty_artifact(read_records(source)), workflow, KitchenEnvironment(),
                audit=True)
        trace.close()
        self.assertGreater(summarize(path)['audit_mismatches'], 0)

    def test_recovery_rejects_an_incompatible_workflow_or_artifact(self):
        source = self.run_episode('memory.jsonl',
                                  dict(step=1, role='memory', operator='incorrect', seed=2))
        records = read_records(source)
        trace = Trace(self.root / 'rejected.jsonl')
        self.addCleanup(trace.close)
        with self.assertRaisesRegex(ValueError, 'max_plan_attempts'):
            recover(source, faulty_artifact(records),
                    Workflow(team(), trace, oracle_hints=False, max_plan_attempts=3),
                    KitchenEnvironment())
        workflow = Workflow(team(), trace, oracle_hints=False)
        with self.assertRaisesRegex(ValueError, 'Unknown faulty artifact'):
            recover(source, 'a999999', workflow, KitchenEnvironment())
        action = next(r['id'] for r in records if r.get('kind') == 'action')
        with self.assertRaisesRegex(ValueError, 'agent output or an observation'):
            recover(source, action, workflow, KitchenEnvironment())
        with self.assertRaisesRegex(ValueError, 'mode'):
            recover(source, faulty_artifact(records), workflow, KitchenEnvironment(), mode='x')


if __name__ == '__main__':
    unittest.main()
