from pathlib import Path
import tempfile
import unittest

from agent_recovery import AgentTeam, Workflow
from agent_recovery.agents import (
    AdmissibilityVerifier, ConstraintsAgent, MemoryAgent, ObservationAgent)
from agent_recovery.contracts import AgentResult
from agent_recovery.environment import Frame
from agent_recovery.injection import FaultSpec, Injector
from agent_recovery.recovery import (
    RestoreFailed, faulty_artifact, read_records, recover, summarize)
from agent_recovery.trace import Trace
from agent_recovery.world import location_before
from verify_trace import verify

PLACES = ('countertop 1', 'fridge 1', 'microwave 1')


class House:
    """ALFWorld-like: observations are action feedback, commands depend on the location, what is
    held and whether the fridge is open, and heating (allowed anywhere while holding) has no
    inverse. The episode ends once the apple is in the fridge; it is won only if not heated."""
    def __init__(self):
        self.executed = []

    def describe(self):
        return {'backend': 'fake-house'}

    def reset(self):
        self.location, self.apple, self.fridge_open, self.hot = None, 'countertop 1', False, False
        self.executed = []
        return self.frame('You are in the middle of a room. Your task is to: put a apple in fridge.')

    def commands(self):
        commands = ['inventory', 'look'] + [f'go to {p}' for p in PLACES if p != self.location]
        holding = self.apple == 'held'
        if holding:
            commands += ['examine apple 1', 'heat apple 1 with microwave 1']
            if self.location == 'countertop 1' or (self.location == 'fridge 1' and self.fridge_open):
                commands.append(f'move apple 1 to {self.location}')
        elif self.apple == self.location and (self.location != 'fridge 1' or self.fridge_open):
            commands.append(f'take apple 1 from {self.location}')
        if self.location == 'fridge 1':
            commands.append('close fridge 1' if self.fridge_open else 'open fridge 1')
        return tuple(sorted(commands))

    def frame(self, text):
        done = self.apple == 'fridge 1'
        return Frame(text, self.commands(), 'house', score=float(done and not self.hot),
                     done=done, won=done and not self.hot)

    def step(self, command):
        if command not in self.commands():
            raise AssertionError(f'inadmissible command {command!r}')
        self.executed.append(command)
        verb, _, rest = command.partition(' ')
        if verb == 'go':
            self.location = rest[len('to '):]
            text = f'You arrive at {self.location}.'
        elif verb == 'take':
            self.apple = 'held'
            text = f'You pick up the apple 1 from the {self.location}.'
        elif verb == 'move':
            self.apple = self.location
            text = f'You move the apple 1 to the {self.location}.'
        elif verb in ('open', 'close'):
            self.fridge_open = verb == 'open'
            text = f'You {verb} the fridge 1.'
        elif verb == 'heat':
            self.hot = True
            text = 'You heat the apple 1 using the microwave 1.'
        else:
            text = 'Nothing happens.'
        return self.frame(text)


class HousePlanner:
    """Chooses from the admissible commands only, so it also recovers from a detour."""
    def run(self, request):
        allowed = request.inputs['constraints'].content['allowed_actions']
        for command in ('move apple 1 to fridge 1', 'take apple 1 from countertop 1'):
            if command in allowed:
                return AgentResult({'action': command})
        holding = 'examine apple 1' in allowed
        if holding and 'open fridge 1' in allowed:
            return AgentResult({'action': 'open fridge 1'})
        target = 'go to fridge 1' if holding else 'go to countertop 1'
        return AgentResult({'action': target if target in allowed else 'look'})


def team():
    return AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(),
                     HousePlanner(), AdmissibilityVerifier())


def actions(path):
    return [r['content']['command'] for r in read_records(path)
            if r['record'] == 'artifact' and r['kind'] == 'action']


CLEAN = ['go to countertop 1', 'take apple 1 from countertop 1', 'go to fridge 1',
         'open fridge 1', 'move apple 1 to fridge 1']


class WorldRestoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def faulty_run(self, name, **fault):
        """Run a faulty episode; returns the trace and the live house it left behind."""
        path, house = self.root / name, House()
        trace = Trace(path)
        injector = Injector(FaultSpec(**fault))
        try:
            Workflow(team(), trace, oracle_hints=False, injector=injector).run(house)
        finally:
            trace.close()
        self.assertTrue(injector.fired)
        return path, house

    def recover(self, source, house, name, world, mode='cutoff'):
        path = self.root / name
        trace = Trace(path)
        try:
            workflow = Workflow(team(), trace, oracle_hints=False)
            recover(source, faulty_artifact(read_records(source)), workflow, house,
                    mode=mode, world=world)
        finally:
            trace.close()
        return path

    def test_clean_episode(self):
        path, house = self.root / 'clean.jsonl', House()
        trace = Trace(path)
        Workflow(team(), trace, oracle_hints=False).run(house)
        trace.close()
        self.assertEqual(actions(path), CLEAN)

    def test_compensation_undoes_a_detour_without_reset(self):
        # Step 1: a wrong 'go to microwave 1' instead of taking the apple.
        source, house = self.faulty_run('detour.jsonl', step=1, role='planner',
                                        operator='incorrect', seed=0)
        self.assertEqual(actions(source)[1], 'go to microwave 1')
        self.assertEqual(house.apple, 'fridge 1')  # The faulty episode still finished.

        class NoReset(House):
            def reset(self):
                raise AssertionError('compensation must not reset')
        live = NoReset()
        live.__dict__.update(house.__dict__)
        live.executed = []
        path = self.recover(source, live, 'compensated.jsonl', 'compensate')
        verify(path)
        self.assertEqual(actions(path), CLEAN)
        summary = summarize(path)
        self.assertTrue(summary['complete_recovery'])
        (restore,) = summary['world_restores']
        self.assertEqual((restore['mechanism'], restore['from_step'], restore['to_step']),
                         ('compensate', 7, 1))
        self.assertTrue(restore['fingerprint_match'])
        # Six recorded actions after step 1 were undone, newest first, then the run continued.
        self.assertEqual(live.executed[:6], [
            'take apple 1 from fridge 1', 'close fridge 1', 'go to countertop 1',
            'move apple 1 to countertop 1', 'go to microwave 1', 'go to countertop 1'])
        self.assertEqual(live.executed[6:], CLEAN[1:])

    def test_reset_restores_the_same_world_by_replay(self):
        source, house = self.faulty_run('detour.jsonl', step=1, role='planner',
                                        operator='incorrect', seed=0)
        path = self.recover(source, House(), 'reset.jsonl', 'reset')
        self.assertEqual(actions(path), CLEAN)
        (restore,) = summarize(path)['world_restores']
        self.assertEqual((restore['mechanism'], restore['environment_actions']), ('reset', 2))

    def test_irreversible_action_is_reported_not_recovered(self):
        source, house = self.faulty_run('heated.jsonl', step=2, role='planner',
                                        operator='incorrect', seed=5)
        self.assertEqual(actions(source)[2], 'heat apple 1 with microwave 1')
        self.assertTrue(house.hot)
        path = self.recover(source, house, 'compensated.jsonl', 'compensate')
        summary = summarize(path)
        self.assertEqual(actions(path), CLEAN)
        # The actions are right again, but the apple stays heated: the task is lost and the
        # irreversible effect is reported instead of counting this as recovered.
        self.assertFalse(summary['won'])
        self.assertFalse(summary['complete_recovery'])
        self.assertEqual(summary['irreversible_effects'], ['heat apple 1 with microwave 1'])
        # A reset (a simulator, or a snapshot) does undo it.
        reset = summarize(self.recover(source, House(), 'reset.jsonl', 'reset'))
        self.assertTrue(reset['complete_recovery'])
        self.assertEqual(reset['irreversible_effects'], [])

    def test_compensation_fails_without_an_inverse_and_auto_falls_back_to_reset(self):
        # Step 0: the first 'go to' has no previous location to return to.
        source, house = self.faulty_run('first.jsonl', step=0, role='planner',
                                        operator='incorrect', seed=0)
        self.assertNotEqual(actions(source)[0], CLEAN[0])
        snapshot = dict(house.__dict__, executed=list(house.executed))
        with self.assertRaises(RestoreFailed):
            self.recover(source, house, 'failed.jsonl', 'compensate')
        events = [r['kind'] for r in read_records(self.root / 'failed.jsonl')
                  if r['record'] == 'event']
        self.assertIn('world_restore_failed', events)
        self.assertEqual(events[-1], 'run_error')
        house.__dict__.update(snapshot)
        path = self.recover(source, house, 'auto.jsonl', 'auto')
        summary = summarize(path)
        self.assertTrue(summary['complete_recovery'])
        self.assertEqual(summary['fallbacks'], 1)
        # The failed attempt's undo actions are counted as well as the reset replay.
        failed = sum(1 for r in read_records(path) if r.get('kind') == 'world_restore_fallback'
                     for _ in range(r['details']['environment_actions']))
        self.assertGreater(failed, 0)
        self.assertEqual(summary['world_actions'],
                         failed + summary['world_restores'][0]['environment_actions'])
        self.assertEqual(actions(path), CLEAN)

    def test_no_world_restore_when_actions_never_diverge(self):
        source, house = self.faulty_run('reasoning.jsonl', step=2, role='observation',
                                        operator='contradictory')
        self.assertEqual(actions(source), CLEAN)
        before = list(house.executed)
        path = self.recover(source, house, 'recovered.jsonl', 'compensate')
        summary = summarize(path)
        self.assertTrue(summary['complete_recovery'])
        self.assertEqual(summary['world_restores'], [])
        self.assertEqual(house.executed, before)  # The real world was never touched.

    def test_environment_fault_is_reobserved_by_compensation(self):
        source, house = self.faulty_run('corrupted.jsonl', step=2, role='environment',
                                        operator='incorrect', seed=1)
        path = self.recover(source, house, 'recovered.jsonl', 'compensate')
        verify(path)
        records = read_records(path)
        refreshed = next(r for r in records if r.get('kind') == 'observation_refreshed')
        fresh = next(r for r in records if r.get('id') == refreshed['artifact_ids'][1])
        self.assertEqual(fresh['content']['text'], 'You pick up the apple 1 from the countertop 1.')
        self.assertEqual(actions(path), CLEAN)

    def test_irreversible_effects_of_a_superseded_compensation_are_not_counted(self):
        import json
        path = self.root / 'synthetic.jsonl'

        def event(sequence, kind, **details):
            return {'record': 'event', 'sequence': sequence, 'kind': kind, 'step': 1,
                    'artifact_ids': [], 'elapsed_seconds': 0.0, 'details': details}
        compensated = dict(mechanism='compensate', to_step=1, from_step=4, environment_actions=3,
                           irreversible=['heat apple 1 with microwave 1'], seconds=0.0)
        records = [
            event(0, 'recovery_started', prefix_agent_calls=5),
            event(1, 'world_restored', **compensated, fingerprint_match=False),
            event(2, 'world_restore_fallback', mechanism='compensate', reason='fingerprint',
                  environment_actions=0),
            event(3, 'world_restored', mechanism='reset', to_step=1, from_step=None,
                  environment_actions=2, irreversible=[], fingerprint_match=True, seconds=0.0),
            event(4, 'run_end', won=True)]
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        summary = summarize(path)
        self.assertEqual(summary['irreversible_effects'], [])
        self.assertTrue(summary['complete_recovery'])
        self.assertEqual(summary['world_actions'], 5)  # both attempts cost real actions
        # A compensation that took effect keeps its irreversible effects.
        records[1]['details']['fingerprint_match'] = True
        path.write_text(''.join(json.dumps(r) + '\n' for r in records[:2] + records[4:]))
        self.assertEqual(summarize(path)['irreversible_effects'], ['heat apple 1 with microwave 1'])

    def test_location_before(self):
        steps = ['go to a 1', 'take x 1 from a 1', 'go to b 1', 'open b 1']
        self.assertIsNone(location_before(steps, 0))
        self.assertEqual(location_before(steps, 2), 'a 1')
        self.assertEqual(location_before(steps, 4), 'b 1')


if __name__ == '__main__':
    unittest.main()
