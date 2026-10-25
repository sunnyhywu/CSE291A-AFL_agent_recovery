import json
from pathlib import Path
import tempfile
import unittest
from baseline import Trace, StateAgent, MemoryAgent, OraclePlanner


class TraceTests(unittest.TestCase):
    def test_branching_dependencies_survive_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = Trace(path)
            root = trace.add('observation', 'environment', (), 0,
                             {'text': 'apple on table', 'admissible_commands': ['look']})
            hint = trace.add('oracle_hint', 'expert', (root.id,), 0, {'commands': ['look']})
            state = StateAgent().run(trace, 0, root)
            memory = MemoryAgent().run(trace, 0, state)
            plan = OraclePlanner().run(trace, 0, state, memory, hint)
            self.assertEqual(trace.descendants(hint.id), {plan.id})
            self.assertEqual(trace.descendants(root.id), {hint.id, state.id, memory.id, plan.id})
            with self.assertRaises(ValueError):
                trace.add('bad', 'test', ('missing',), 0, {})
            trace.event('run_end', 0, (plan.id,), won=True)
            trace.file.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            seen = set()
            for record in records:
                if record['record'] == 'artifact':
                    self.assertTrue(set(record['inputs']) <= seen)
                    seen.add(record['id'])
            self.assertTrue(records[-1]['details']['won'])
            with self.assertRaises(FileExistsError):
                Trace(path)


if __name__ == '__main__':
    unittest.main()
