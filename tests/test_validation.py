from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from agent_recovery import AgentTeam, Workflow
from agent_recovery.agents import (
    ObservationAgent, MemoryAgent, ConstraintsAgent, OraclePlanner, AdmissibilityVerifier)
from agent_recovery.trace import Trace
from test_workflow import FakeEnvironment, FixedAgent
from verify_trace import verify


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'trace.jsonl'

    def make_trace(self, rejected=False):
        team = AgentTeam(ObservationAgent(), MemoryAgent(), ConstraintsAgent(),
                         OraclePlanner(), AdmissibilityVerifier())
        if rejected:
            team.verifier = FixedAgent({'approved': False, 'reason': 'Rejected'})
        trace = Trace(self.path)
        try:
            Workflow(team, trace).run(FakeEnvironment())
        finally:
            trace.close()
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def check(self, success=True):
        with redirect_stdout(StringIO()):
            return verify(self.path, require_success=success)

    def rewrite(self, records):
        self.path.write_text(''.join(json.dumps(record) + '\n' for record in records))

    def test_complete_successful_five_role_trace(self):
        self.make_trace()
        self.assertEqual(self.check()['actions'], 2)

    def test_rejection_trace_valid_only_when_failure_allowed(self):
        self.make_trace(rejected=True)
        with self.assertRaisesRegex(ValueError, 'Mission did not succeed'):
            self.check()
        self.assertFalse(self.check(success=False)['won'])

    def test_tampered_verification_cannot_authorize_action(self):
        records = self.make_trace()
        verification = next(r for r in records if r.get('kind') == 'verification')
        verification['content']['approved'] = False
        self.rewrite(records)
        with self.assertRaisesRegex(ValueError, 'Rejected action was executed'):
            self.check()

    def test_missing_named_input_edge_is_detected(self):
        records = self.make_trace()
        plan = next(r for r in records if r.get('kind') == 'plan')
        plan['inputs'] = plan['inputs'][1:]
        self.rewrite(records)
        with self.assertRaisesRegex(ValueError, 'Call inputs missing from DAG'):
            self.check()

    def test_missing_transition_is_detected(self):
        records = self.make_trace()
        records = [r for r in records if not (r.get('kind') == 'transition' and r['step'] == 1)]
        for sequence, event in enumerate(r for r in records if r['record'] == 'event'):
            event['sequence'] = sequence
        self.rewrite(records)
        with self.assertRaisesRegex(ValueError, 'Incomplete action/transition history'):
            self.check()


if __name__ == '__main__':
    unittest.main()
