import json
from pathlib import Path
import tempfile
import unittest

from agent_recovery import AgentTeam, Workflow
from agent_recovery.agents import (
    AdmissibilityVerifier, ConstraintsAgent, MemoryAgent, ObservationAgent)
from agent_recovery.contracts import AgentResult
from agent_recovery.environment import Frame
from agent_recovery.injection import FaultSpec, Injector, can_fire
from agent_recovery.trace import Artifact, Trace

TEXT = ('You arrive at countertop 1. On the countertop 1, you see a apple 1, a bread 1, '
        'and a knife 2.')


class RoomEnvironment:
    """Three-step fake: look, look, then 'finish' wins."""
    def __init__(self):
        self.count = 0

    def describe(self):
        return {'backend': 'fake'}

    def frame(self):
        text = (f'{TEXT}\nYour task is to: put a apple in fridge.' if self.count == 0
                else f'{TEXT} (step {self.count})')
        won = self.count == 3
        return Frame(text, ('look', 'finish', 'take apple 1 from countertop 1'), 'fake',
                     score=float(won), done=won, won=won)

    def reset(self):
        self.count = 0
        return self.frame()

    def step(self, command):
        self.count += 1
        return self.frame()

    def close(self):
        pass


class EchoPlanner:
    """Finishes at step 2; records the observation each step to detect propagation."""
    def __init__(self):
        self.seen = []

    def run(self, request):
        self.seen.append(request.inputs['state'].content['summary'])
        return AgentResult({'action': 'finish' if request.step == 2 else 'look'})


def state(text):
    return Artifact('s', 'state', 'observation', (), 0, {'summary': text, 'facts': [text]})


class InjectionTests(unittest.TestCase):
    def run_with(self, spec):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 't.jsonl'
        trace = Trace(path)
        planner = EchoPlanner()
        team = AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(), planner,
                         AdmissibilityVerifier())
        workflow = Workflow(team, trace, oracle_hints=False,
                            injector=Injector(spec) if spec else None)
        won = workflow.run(RoomEnvironment())
        trace.close()
        return won, planner, [json.loads(line) for line in path.read_text().splitlines()]

    def test_observation_fault_reaches_planner_and_is_logged_with_original(self):
        won, planner, records = self.run_with(FaultSpec(1, 'observation', 'incorrect', seed=3))
        faults = [r for r in records if r.get('kind') == 'fault_injected']
        self.assertEqual(len(faults), 1)
        self.assertIn('apple 1', faults[0]['details']['original']['summary'])
        self.assertNotEqual(planner.seen[1], faults[0]['details']['original']['summary'])
        self.assertEqual(faults[0]['details']['fault']['category'], 'accidental')
        self.assertTrue(won)

    def test_fault_fires_once_and_clean_run_has_no_fault(self):
        _, clean_planner, clean = self.run_with(None)
        _, planner, faulty = self.run_with(FaultSpec(1, 'observation', 'contradictory', seed=1))
        self.assertFalse(any(r.get('kind') == 'fault_injected' for r in clean))
        self.assertEqual(planner.seen[0], clean_planner.seen[0])
        self.assertIn('Looking again, there is no', planner.seen[1])
        self.assertEqual(planner.seen[2], clean_planner.seen[2])

    def test_timeout_reuses_previous_output_as_explicit_dependency(self):
        _, planner, records = self.run_with(FaultSpec(1, 'observation', 'timeout'))
        self.assertEqual(planner.seen[1], planner.seen[0])
        fault = next(r for r in records if r.get('kind') == 'fault_injected')
        state_ids = [r['id'] for r in records if r.get('kind') == 'state']
        self.assertEqual(fault['details']['fallback'], state_ids[0])
        self.assertIn(state_ids[0], [r for r in records if r.get('id') == state_ids[1]][0]['inputs'])

    def test_text_operators(self):
        inputs = {'state': state(TEXT)}
        missing = Injector(FaultSpec(0, 'observation', 'missing', seed=0))._missing(TEXT, inputs)
        self.assertLess(missing.count(' 1') + missing.count(' 2'), TEXT.count(' 1') + TEXT.count(' 2'))
        incorrect = Injector(FaultSpec(0, 'observation', 'incorrect', seed=0))._incorrect(TEXT, inputs)
        self.assertNotEqual(incorrect, TEXT)
        malicious = Injector(FaultSpec(0, 'observation', 'malicious', seed=0))._malicious(TEXT, inputs)
        self.assertIn('garbagecan', malicious)
        self.assertEqual(FaultSpec(0, 'observation', 'malicious').category, 'adversarial')

    def test_memory_fault_corrupts_shared_state_for_later_steps(self):
        _, planner, records = self.run_with(FaultSpec(1, 'memory', 'contradictory', seed=2))
        memories = [r for r in records if r.get('kind') == 'memory']
        self.assertIn('Looking again', memories[1]['content']['observations'][-1])
        self.assertIn('Looking again', memories[2]['content']['summary'])

    def test_invalid_spec_rejected(self):
        with self.assertRaises(ValueError):
            FaultSpec(0, 'planner', 'malicious')

    def test_can_fire_requires_object_mentions_for_text_rewrites(self):
        closed = 'You arrive at drawer 2. The drawer 2 is closed.'
        self.assertFalse(can_fire('observation', 'incorrect', closed))
        self.assertFalse(can_fire('memory', 'missing', closed))
        self.assertTrue(can_fire('observation', 'incorrect', TEXT))
        self.assertTrue(can_fire('observation', 'malicious', closed))
        self.assertTrue(can_fire('planner', 'incorrect', closed))

    def test_malicious_fault_logs_payload_target(self):
        injector = Injector(FaultSpec(1, 'observation', 'malicious', seed=3))
        inputs = {'observation': Artifact('o', 'observation', 'environment', (), 1,
                                          {'text': TEXT, 'gamefile': 'fake'})}
        faulty = injector.apply('observation', 1, 0, {'summary': TEXT, 'facts': [TEXT]}, inputs)
        target = injector.detail['payload_target']
        self.assertIn(target, ('apple 1', 'bread 1', 'knife 2'))  # Objects seen; no task text given.
        self.assertIn(f'take {target}', faulty['summary'])

    def test_environment_fault_corrupts_raw_observation_before_any_agent(self):
        won, planner, records = self.run_with(FaultSpec(1, 'environment', 'contradictory', seed=1))
        fault = next(r for r in records if r.get('kind') == 'fault_injected')
        observation = next(r for r in records if r.get('id') == fault['artifact_ids'][0])
        self.assertEqual((observation['kind'], observation['step']), ('observation', 1))
        self.assertIn('Looking again, there is no', observation['content']['text'])
        self.assertNotIn('Looking again', fault['details']['original']['text'])
        self.assertIn('Looking again, there is no', planner.seen[1])
        self.assertTrue(won)

    def test_fact_operators_work_on_one_fact_per_line(self):
        facts = 'apple 1 on countertop 1\nat countertop 1\nknife 2 on countertop 1'
        missing = Injector(FaultSpec(0, 'observation', 'missing', seed=0))._missing(facts, {})
        self.assertEqual(len(missing.splitlines()), 2)
        self.assertIn('at countertop 1', missing)
        incorrect = Injector(FaultSpec(0, 'observation', 'incorrect', seed=0))._incorrect(facts, {})
        self.assertIn('countertop 1', incorrect)
        self.assertNotEqual(incorrect, facts)
        self.assertNotIn(' a ', incorrect)

    def test_belief_memory_missing_forgets_half_the_facts(self):
        content = {'summary': 'p', 'format': 'beliefs', 'actions': ['look'],
                   'observations': ['a 1 on b 1', 'c 1 on b 1', 'at b 1', 'holding d 1']}
        faulty = Injector(FaultSpec(2, 'memory', 'missing', seed=0)).apply('memory', 2, 0, content, {})
        self.assertEqual(len(faulty['observations']), 2)
        self.assertEqual(faulty['actions'], ['look'])

    def test_alter_payload_targets_a_seen_foreign_object_with_a_present_appliance(self):
        injector = Injector(FaultSpec(1, 'environment', 'malicious_alter', seed=0))
        injector.observe('environment', {'text': 'You see a fridge 1, a microwave 1.\n'
                                                 'Your task is to: put a apple in fridge.'})
        faulty = injector.apply('environment', 1, 0, {'text': TEXT, 'gamefile': 'g'}, {})
        self.assertIn(injector.detail['payload_target'], ('bread 1', 'knife 2'))
        verb = injector.detail['payload_action'].split()[0]
        self.assertIn(verb, ('heat', 'cool'))
        self.assertIn(f"{verb} it with {'microwave 1' if verb == 'heat' else 'fridge 1'}", faulty['text'])
        self.assertEqual(FaultSpec(1, 'environment', 'malicious_alter').category, 'adversarial')


if __name__ == '__main__':
    unittest.main()
