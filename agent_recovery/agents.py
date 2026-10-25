"""Cheap deterministic role implementations. Replace any with your own hook."""
from .contracts import AgentRequest, AgentResult


class ObservationAgent:
    def run(self, request: AgentRequest) -> AgentResult:
        text = request.inputs['observation'].content['text']
        return AgentResult({'summary': text,
                            'facts': [line.strip() for line in text.splitlines() if line.strip()]})


class MemoryAgent:
    def __init__(self, window: int = 8):
        if type(window) is not int or window < 1:
            raise ValueError('memory window must be a positive integer')
        self.window = window

    def run(self, request: AgentRequest) -> AgentResult:
        previous = request.inputs.get('previous_memory')
        observations = list(previous.content['observations']) if previous else []
        actions = list(previous.content['actions']) if previous else []
        observations.append(request.inputs['state'].content['summary'])
        if 'previous_action' in request.inputs:
            actions.append(request.inputs['previous_action'].content['command'])
        observations, actions = observations[-self.window:], actions[-self.window:]
        return AgentResult({'summary': '\n'.join(observations),
                            'observations': observations, 'actions': actions})


class ConstraintsAgent:
    def run(self, request: AgentRequest) -> AgentResult:
        return AgentResult({'goal': request.inputs['task'].content['goal'],
                            'allowed_actions': request.inputs['action_space'].content['commands']})


class OraclePlanner:
    """Explicit infrastructure control, not an independent reasoning policy."""
    requires_oracle = True

    def run(self, request: AgentRequest) -> AgentResult:
        oracle = request.inputs.get('oracle')
        if oracle is None or not oracle.content['commands']:
            raise RuntimeError('OraclePlanner requires oracle hints; configure a custom planner')
        return AgentResult({'action': oracle.content['commands'][0],
                            'rationale': 'ALFWorld handcoded expert control'})


class AdmissibilityVerifier:
    """Checks syntax/availability only; semantic task checking belongs in a hook."""
    def run(self, request: AgentRequest) -> AgentResult:
        action = request.inputs['plan'].content['action']
        approved = action in request.inputs['constraints'].content['allowed_actions']
        return AgentResult({'approved': approved,
                            'reason': 'Action is admissible' if approved else
                                      'Action is absent from allowed_actions'})
