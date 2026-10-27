"""List TritonAI models and optionally send one tiny request. Never prints the API key."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_recovery.llm import make_client  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Send one short test request to this model')
args = parser.parse_args()
client = make_client(timeout=60)
for model in sorted(m.id for m in client.models.list().data):
    print(model)
if args.model:
    started = time.monotonic()
    response = client.chat.completions.create(
        model=args.model, temperature=0, max_tokens=20,
        messages=[{'role': 'user', 'content': 'Reply with the single word: ready'}])
    print(f'\n{args.model}: {response.choices[0].message.content!r} '
          f'({time.monotonic() - started:.2f}s, usage={response.usage})')
