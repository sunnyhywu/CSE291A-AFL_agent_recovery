"""Replace the method body with your local-model or API-model call."""
from agent_recovery import AgentRequest, AgentResult


class MyPlanner:
    def __init__(self, model_name='your-model'):
        self.model_name = model_name
        # Initialize your client here. Read API keys from environment variables.

    def run(self, request: AgentRequest) -> AgentResult:
        state = request.inputs['state'].content
        memory = request.inputs['memory'].content
        constraints = request.inputs['constraints'].content
        feedback = request.inputs.get('feedback')  # execution_gate artifact on retry
        # Build a prompt from these inputs. Use request.seed for deterministic sampling.
        # response = your_client(...)
        # return AgentResult(
        #     content={'action': response.action, 'rationale': response.reason},
        #     metrics={'input_tokens': response.input_tokens,
        #              'output_tokens': response.output_tokens})
        raise NotImplementedError('Connect your model here and return AgentResult')


class LookPlanner:
    """Runnable hook example only: demonstrates wiring, usually hits the step limit."""
    def run(self, request: AgentRequest) -> AgentResult:
        return AgentResult({'action': 'look', 'rationale': 'Hook smoke test'})
