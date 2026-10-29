import random
import unittest

from agent_recovery.injection import TARGETS, FaultSpec
from analysis.compare import signature
from experiments.fuzz import key, mutate, random_fault

TEXT = 'You arrive at countertop 1. On the countertop 1, you see a apple 1.'
GAMES = {f'valid_unseen-t1-g00{i}-s42.jsonl': {
    'split': 'valid_unseen', 'task_type': 1, 'game_index': i, 'seed': 42, 'steps': steps,
    'texts': {s: TEXT if s % 2 else 'Nothing happens.' for s in range(steps + 1)}}
    for i, steps in enumerate((4, 12, 30))}


def row(**changes):
    base = {'fault_fired': True, 'propagation_path': 'memory>planner', 'propagation': 'actions_changed',
            'changed_calls': 5, 'steps_to_divergence': 0, 'unauthorized_actions': 1,
            'clean_unauthorized_actions': 1, 'irreversible_actions': 0}
    return {**base, **changes}


class FuzzTests(unittest.TestCase):
    def test_mutants_are_valid_faults_inside_their_game(self):
        rng = random.Random(0)
        item = random_fault(rng, GAMES, 'fuzz')
        mutations = set()
        for _ in range(500):
            item, mutation = mutate(rng, GAMES, item)
            mutations.add(mutation)
            fault = item['fault']
            FaultSpec(fault['step'], fault['role'], fault['operator'], seed=fault['seed'])
            self.assertIn(fault['operator'], TARGETS[fault['role']])
            self.assertTrue(1 <= fault['step'] < GAMES[item['clean_trace']]['steps'])
        self.assertEqual(mutations, {'step', 'operator', 'role', 'game', 'seed'})

    def test_reseeding_changes_the_fault_key(self):
        item = random_fault(random.Random(1), GAMES, 'fuzz')
        reseeded = {**item, 'fault': {**item['fault'], 'seed': item['fault']['seed'] + 1}}
        self.assertNotEqual(key(item), key(reseeded))

    def test_signature_buckets_and_flags(self):
        self.assertEqual(signature(row()), 'memory>planner|actions_changed|spread:4-15|delay:now')
        self.assertEqual(signature(row(changed_calls=0, steps_to_divergence=None, propagation='masked',
                                       propagation_path='memory')),
                         'memory|masked|spread:none|delay:none')
        self.assertTrue(signature(row(unauthorized_actions=2, irreversible_actions=1, steps_to_divergence=7))
                        .endswith('delay:latent|unauthorized|irreversible'))
        self.assertIsNone(signature({'fault_fired': False}))


if __name__ == '__main__':
    unittest.main()
