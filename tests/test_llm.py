from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent_recovery import AgentRequest
from datetime import datetime, timezone

from agent_recovery import llm
from agent_recovery.llm import CachedChat, rate_limit_delay
from agent_recovery.llm_agents import (
    LLMMemory, LLMObservation, LLMPlanner, LLMVerifier, history_lines, memory_lines,
    parse_action, parse_verdict)
from agent_recovery.trace import Artifact


class FakeCompletions:
    def __init__(self, text):
        self.text, self.requests = text, []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            model=request['model'], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text),
                                     finish_reason='stop')])


def fake_client(text):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(text)))


def artifact(kind, content):
    return Artifact(f'{kind}-id', kind, 'test', (), 0, content)


def planner_request(feedback=None):
    inputs = {'state': artifact('state', {'summary': 'You see a apple 1.', 'facts': []}),
              'memory': artifact('memory', {'summary': '', 'observations': ['start', 'You see a apple 1.'],
                                            'actions': ['go to countertop 1']}),
              'constraints': artifact('constraints', {'goal': 'put an apple in fridge.',
                                                      'allowed_actions': ['look', 'take apple 1 from countertop 1']})}
    if feedback:
        inputs['feedback'] = artifact('execution_gate', feedback)
    return AgentRequest('planner', 1, 0, 7, inputs)


class LLMTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)

    def test_cache_returns_original_metrics_without_new_call(self):
        client = fake_client('{"action": "look"}')
        chat = CachedChat('m', cache_dir=self.cache, client=client)
        messages = [{'role': 'user', 'content': 'hi'}]
        first, first_metrics = chat.complete(messages, seed=1)
        second, second_metrics = chat.complete(messages, seed=1)
        self.assertEqual(first, second)
        self.assertEqual(len(client.chat.completions.requests), 1)
        self.assertEqual((first_metrics['cache_hit'], second_metrics['cache_hit']), (0, 1))
        self.assertEqual(first_metrics['input_tokens'], second_metrics['input_tokens'])
        chat.complete(messages, seed=2)
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_parse_maps_case_and_falls_back_without_inventing(self):
        allowed = ['look', 'take apple 1 from countertop 1']
        self.assertEqual(parse_action('```json\n{"action": "Look", "rationale": "r"}\n```', allowed),
                         ('look', 'r', 'json'))
        self.assertEqual(parse_action('I will take apple 1 from countertop 1.', allowed)[2], 'fallback')
        self.assertEqual(parse_action('fly away', allowed), ('fly away', '', 'failed'))

    def test_history_pairs_observations_with_following_actions(self):
        memory = {'observations': ['o1', 'o2', 'o3'], 'actions': ['a0', 'a1', 'a2']}
        self.assertEqual(history_lines(memory),
                         ['Observation: o1', '> a1', 'Observation: o2', '> a2'])

    def test_planner_uses_inputs_feedback_and_seed(self):
        client = fake_client('{"action": "take apple 1 from countertop 1", "rationale": "need it"}')
        planner = LLMPlanner('m', client=client)
        planner.chat.cache_dir = self.cache
        result = planner.run(planner_request({'action': 'fly', 'reason': 'not allowed'}))
        self.assertEqual(result.content['action'], 'take apple 1 from countertop 1')
        request = client.chat.completions.requests[0]
        prompt = request['messages'][1]['content']
        self.assertIn('put an apple in fridge.', prompt)
        self.assertIn('Observation: start\n> go to countertop 1', prompt)
        self.assertIn('"fly" was rejected', prompt)
        self.assertEqual(request['seed'], 7)
        self.assertEqual(result.metrics['output_tokens'], 3)

    def test_rate_limit_waits_for_reset_and_reports_throttle_separately(self):
        class RateLimited(Exception):
            status_code = 429

        client = fake_client('{"action": "look"}')
        create, calls = client.chat.completions.create, []

        def flaky(**request):
            calls.append(request)
            if len(calls) == 1:
                raise RateLimited('Limit resets at: 2026-10-26 03:14:31 UTC')
            return create(**request)

        client.chat.completions.create = flaky
        sleeps = []
        original_sleep, original_delay = llm.time.sleep, llm.rate_limit_delay
        llm.time.sleep, llm.rate_limit_delay = sleeps.append, lambda message: 5.0
        try:
            _, metrics = CachedChat('m', cache_dir=self.cache, client=client).complete(
                [{'role': 'user', 'content': 'x'}], seed=1)
        finally:
            llm.time.sleep, llm.rate_limit_delay = original_sleep, original_delay
        self.assertEqual((len(calls), sleeps, metrics['throttle_seconds']), (2, [5.0], 5.0))

    def test_rate_limit_delay_parses_reset_time(self):
        now = datetime(2026, 10, 26, 3, 14, 0, tzinfo=timezone.utc)
        delay = rate_limit_delay('Limit resets at: 2026-10-26 03:14:31 UTC', now)
        self.assertTrue(32 <= delay <= 33)
        self.assertEqual(rate_limit_delay('no reset given'), 60.0)

    def test_verdict_parse_fails_open_and_is_recorded(self):
        self.assertEqual(parse_verdict('{"approved": false, "reason": "unrelated object"}'),
                         (False, 'unrelated object', 'json'))
        self.assertEqual(parse_verdict('{"approved": "no"}')[::2], (True, 'failed'))

    def test_verifier_sees_raw_observation_and_skips_inadmissible_plans(self):
        client = fake_client('{"approved": false, "reason": "state contradicts observation"}')
        verifier = LLMVerifier('m', client=client)
        verifier.chat.cache_dir = self.cache
        inputs = dict(planner_request().inputs)
        inputs['observation'] = artifact('observation', {'text': 'You see a mug 1.', 'gamefile': 'g'})
        inputs['plan'] = artifact('plan', {'action': 'take apple 1 from countertop 1'})
        result = verifier.run(AgentRequest('verifier', 1, 0, 7, inputs))
        self.assertFalse(result.content['approved'])
        prompt = client.chat.completions.requests[0]['messages'][1]['content']
        self.assertIn('You see a mug 1.', prompt)
        self.assertIn('Proposed action: take apple 1 from countertop 1', prompt)
        inputs['plan'] = artifact('plan', {'action': 'fly'})
        skipped = verifier.run(AgentRequest('verifier', 1, 0, 7, inputs))
        self.assertEqual((skipped.content['approved'], skipped.content['parse']), (False, 'skipped'))
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_observation_agent_returns_sorted_facts_and_degrades_to_raw_lines(self):
        client = fake_client('{"facts": ["at countertop 1", "apple 1 on countertop 1", "at countertop 1"]}')
        agent = LLMObservation('m', client=client)
        agent.chat.cache_dir = self.cache
        raw = artifact('observation', {'text': 'You arrive at countertop 1.\nYou see a apple 1.',
                                       'gamefile': 'g'})
        result = agent.run(AgentRequest('observation', 0, 0, 1, {'observation': raw}))
        self.assertEqual(result.content['facts'], ['apple 1 on countertop 1', 'at countertop 1'])
        self.assertEqual(result.content['summary'], 'apple 1 on countertop 1\nat countertop 1')
        broken = LLMObservation('m', client=fake_client('no json'))
        broken.chat.cache_dir = self.cache / 'b'
        degraded = broken.run(AgentRequest('observation', 0, 0, 1, {'observation': raw}))
        self.assertEqual((degraded.content['parse'], len(degraded.content['facts'])), ('failed', 2))

    def test_memory_agent_updates_beliefs_and_keeps_action_window(self):
        client = fake_client('{"facts": ["holding apple 1", "at countertop 1"], "progress": "Took the apple."}')
        agent = LLMMemory('m', window=2, client=client)
        agent.chat.cache_dir = self.cache
        previous = artifact('memory', {'summary': 'Found the apple.', 'format': 'beliefs',
                                       'observations': ['apple 1 on countertop 1'],
                                       'actions': ['look', 'go to countertop 1']})
        state = artifact('state', {'summary': 'holding apple 1', 'facts': ['holding apple 1']})
        action = artifact('action', {'command': 'take apple 1 from countertop 1'})
        result = agent.run(AgentRequest('memory', 2, 0, 1, {'state': state, 'previous_memory': previous,
                                                             'previous_action': action}))
        self.assertEqual(result.content['observations'], ['at countertop 1', 'holding apple 1'])
        self.assertEqual(result.content['actions'], ['go to countertop 1', 'take apple 1 from countertop 1'])
        self.assertEqual(result.content['summary'], 'Took the apple.')
        prompt = client.chat.completions.requests[0]['messages'][1]['content']
        self.assertIn('- apple 1 on countertop 1', prompt)
        self.assertIn('Last action: take apple 1 from countertop 1', prompt)
        lines = memory_lines(result.content)
        self.assertIn('- holding apple 1', lines)
        self.assertIn('Progress notes: Took the apple.', lines)


if __name__ == '__main__':
    unittest.main()
