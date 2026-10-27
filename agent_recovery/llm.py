"""OpenAI-compatible chat client (UCSD TritonAI) with a content-addressed response cache.

The cache makes recomputation reproducible: an identical request returns the stored
response. Metrics always report the original call's tokens and latency, so logical
cost is unchanged by a cache hit; `cache_hit` distinguishes actual API usage.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / '.env'):
    """Fill missing variables from a git-ignored KEY=VALUE file; real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[len('export '):]
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def make_client(timeout: float):
    from openai import OpenAI
    load_env()
    missing = [name for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL') if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f'Set {", ".join(missing)} in the environment or in {ROOT / ".env"}')
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'],
                  timeout=timeout, max_retries=0)


class CachedChat:
    def __init__(self, model: str, cache_dir: Path = ROOT / '.llm_cache', temperature=0.0,
                 max_tokens=512, timeout=60.0, retries=3, send_seed=True, client=None,
                 max_throttles=20):
        self.model, self.cache_dir = model, Path(cache_dir)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.timeout, self.retries, self.send_seed = timeout, retries, send_seed
        self.max_throttles = max_throttles
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.timeout)
        return self._client

    def complete(self, messages: list[dict], seed: int) -> tuple[str, dict]:
        request = {'model': self.model, 'messages': messages, 'temperature': self.temperature,
                   'max_tokens': self.max_tokens}
        if self.send_seed:
            request['seed'] = seed
        # A namespace (LLM_CACHE_NAMESPACE) isolates runs that must make live calls, such as
        # an executed full restart; it is part of the key but never sent to the API.
        namespace = os.environ.get('LLM_CACHE_NAMESPACE', '')
        keyed = {**request, 'cache_namespace': namespace} if namespace else request
        key = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f'{key}.json'
        hit = path.exists()
        if hit:
            record = json.loads(path.read_text())
        else:
            record = self._call(request)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Unique per writer: concurrent episodes can issue the same request.
            partial = path.with_suffix(f'.{os.getpid()}.{random.randrange(2 ** 32)}.tmp')
            partial.write_text(json.dumps(record, sort_keys=True))
            partial.replace(path)
        return record['text'], {'input_tokens': record['input_tokens'],
                                'output_tokens': record['output_tokens'],
                                'llm_latency_seconds': record['latency_seconds'],
                                'throttle_seconds': 0.0 if hit else record.get('throttle_seconds', 0.0),
                                'cache_hit': int(hit)}

    def _call(self, request: dict) -> dict:
        failures = throttles = 0
        throttle_seconds = 0.0
        while True:
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(**request)
                break
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                if status == 429 and throttles < self.max_throttles:
                    # Wait for the provider's window reset; this is not model latency.
                    throttles += 1
                    delay = rate_limit_delay(str(exc))
                    throttle_seconds += delay
                    time.sleep(delay)
                    continue
                failures += 1
                # Retry transient transport/server failures; client errors fail fast.
                if failures == self.retries or (status is not None and status < 500):
                    raise
                time.sleep(2 ** failures)
        usage = response.usage
        return {'request': request, 'response_model': response.model,
                'text': response.choices[0].message.content or '',
                'finish_reason': response.choices[0].finish_reason,
                'input_tokens': usage.prompt_tokens if usage else 0,
                'output_tokens': usage.completion_tokens if usage else 0,
                'latency_seconds': time.monotonic() - started,
                'throttle_seconds': throttle_seconds}


def rate_limit_delay(message: str, now=None) -> float:
    """Seconds until 'Limit resets at: YYYY-MM-DD HH:MM:SS UTC', else a conservative minute."""
    match = re.search(r'resets at: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC', message)
    if not match:
        return 60.0
    reset = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return min(max((reset - now).total_seconds() + 1 + random.random(), 1.0), 900.0)
