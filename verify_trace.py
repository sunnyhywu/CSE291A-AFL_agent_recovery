"""Validate a completed baseline or five-role trace without importing ALFWorld."""
import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, require_success=True):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    artifacts, events = {}, []
    for record in records:
        if record['record'] == 'artifact':
            require(record['id'] not in artifacts, 'Duplicate artifact ID')
            require(set(record['inputs']) <= artifacts.keys(), 'Missing or forward parent')
            artifacts[record['id']] = record
        elif record['record'] == 'event':
            require(record['sequence'] == len(events), 'Nonconsecutive events')
            require(set(record['artifact_ids']) <= artifacts.keys(), 'Unknown event reference')
            events.append(record)
        else:
            raise ValueError('Unknown record type')
    require(bool(events) and events[0]['kind'] == 'run_start', 'Missing run_start')
    require(records[-1] == events[-1] and events[-1]['kind'] == 'run_end', 'Incomplete run')
    require(sum(e['kind'] == 'run_start' for e in events) == 1, 'Duplicate run_start')
    require(sum(e['kind'] == 'run_end' for e in events) == 1, 'Duplicate run_end')
    require(not any(e['kind'] == 'run_error' for e in events), 'Run contains an error')
    won = events[-1]['details']['won']
    require(type(won) is bool, 'Invalid terminal won value')
    if require_success:
        require(won, 'Mission did not succeed')
    actions = [a for a in artifacts.values() if a['kind'] == 'action']
    transitions = [e for e in events if e['kind'] == 'transition']
    requested = [e for e in events if e['kind'] == 'action_requested']
    require(len(actions) == len(transitions) == len(requested), 'Incomplete action/transition history')
    for step, (action, transition, request) in enumerate(zip(actions, transitions, requested)):
        require(action['step'] == step and transition['step'] == step + 1, 'Incorrect step numbering')
        require(request['artifact_ids'] == [action['id']], 'Action request mismatch')
        require(request['details']['command'] == action['content']['command'], 'Requested command mismatch')
        require(transition['artifact_ids'][0] == action['id'], 'Transition action mismatch')
        observation = artifacts[transition['artifact_ids'][1]]
        require(observation['kind'] == 'observation' and action['id'] in observation['inputs'],
                'Transition observation lacks action provenance')
        plan = artifacts[action['inputs'][0]]
        require(plan['kind'] == 'plan', 'Action lacks a plan')
        require(plan['content']['action'] == action['content']['command'], 'Plan/action mismatch')
    if transitions:
        require(transitions[-1]['details']['won'] == won, 'Terminal success disagrees with environment')
    if won:
        require(bool(transitions) or events[-1]['step'] == 0, 'Missing environment success evidence')
    schema = events[0]['details'].get('schema_version', 1)
    require(schema in (1, 2), 'Unsupported trace schema')
    if schema == 2:
        verify_multi_agent(artifacts, events, actions)
    print(f'PASS: {len(actions)} actions, {len(artifacts)} artifacts, environment won={won}')
    return {'actions': len(actions), 'artifacts': len(artifacts), 'won': won}


def verify_multi_agent(artifacts, events, actions):
    expected = {'observation': 'state', 'memory': 'memory', 'constraints': 'constraints',
                'planner': 'plan', 'verifier': 'verification'}
    pending = None
    completed = []
    for event in events:
        if event['kind'] == 'agent_call_start':
            require(pending is None, 'Overlapping agent calls')
            require(event['details']['role'] in expected, 'Unknown agent role')
            require(set(event['artifact_ids']) == set(event['details']['inputs'].values()),
                    'Named call inputs disagree with artifact references')
            pending = event
        elif event['kind'] == 'agent_call_end':
            require(pending is not None, 'Agent output without call start')
            for key in ('role', 'attempt', 'inputs'):
                require(event['details'][key] == pending['details'][key], 'Call start/end mismatch')
            require(event['step'] == pending['step'], 'Agent call step mismatch')
            require(len(event['artifact_ids']) == 1, 'Agent must produce one output')
            output = artifacts[event['artifact_ids'][0]]
            role = event['details']['role']
            require(output['kind'] == expected[role] and output['producer'] == role, 'Wrong agent output')
            require(set(output['inputs']) == set(pending['artifact_ids']), 'Call inputs missing from DAG')
            require(output['step'] == event['step'], 'Output step differs from agent call')
            if role != 'planner':
                require(all(artifacts[i]['kind'] != 'oracle_hint' for i in output['inputs']),
                        'Oracle hint leaked to a non-planner role')
            completed.append(event)
            pending = None
    require(pending is None, 'Unfinished agent call')
    for action in actions:
        require(len(action['inputs']) == 3, 'Action lacks plan, verification, or gate')
        plan, verification, gate = [artifacts[i] for i in action['inputs']]
        require(verification['kind'] == 'verification' and gate['kind'] == 'execution_gate',
                'Incorrect action provenance')
        require(verification['content']['approved'] is True and gate['content']['approved'] is True,
                'Rejected action was executed')
        require(plan['id'] in verification['inputs'], 'Verification evaluated a different plan')
        require(gate['inputs'][:2] == [plan['id'], verification['id']], 'Gate evaluated a different proposal')
        space = artifacts[gate['inputs'][2]]
        require(space['kind'] == 'action_space', 'Gate lacks environment action space')
        require(action['content']['command'] == gate['content']['action']
                and action['content']['command'] in space['content']['commands'],
                'Executor approved an inadmissible or mismatched command')
        step_roles = {e['details']['role'] for e in completed if e['step'] == action['step']}
        require(step_roles == set(expected), 'Executed step is missing agent roles')
    details = events[-1]['details']
    require(details['agent_calls'] == len(completed), 'Incorrect agent-call count')
    require(details['environment_actions'] == len(actions), 'Incorrect environment-action count')
    if not events[0]['details']['oracle_hints']:
        require(not any(a['kind'] == 'oracle_hint' for a in artifacts.values()), 'Unexpected oracle artifact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--allow-failure', action='store_true', help='Accept a completed unsuccessful run')
    args = parser.parse_args()
    verify(args.trace, require_success=not args.allow_failure)
