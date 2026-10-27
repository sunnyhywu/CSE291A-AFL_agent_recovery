"""Append-only JSONL artifacts and execution events, shared with the baseline."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any


def json_copy(value: Any) -> Any:
    """Detach mutable data and reject non-JSON values, NaN, and infinity."""
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    producer: str
    inputs: tuple[str, ...]
    step: int
    content: dict


@dataclass(frozen=True)
class ExecutionEvent:
    sequence: int
    kind: str
    step: int
    artifact_ids: tuple[str, ...]
    elapsed_seconds: float
    details: dict


class Trace:
    def __init__(self, path: Path):
        self.file = path.open('x')
        self.artifacts: dict[str, Artifact] = {}
        self.children: dict[str, set[str]] = {}
        self.sequence = 0
        self.started = time.monotonic()
        # Bookkeeping overhead: serialized bytes and time spent recording.
        self.bytes_written, self.write_seconds = 0, 0.0

    def write(self, record: dict):
        started = time.monotonic()
        line = json.dumps(record, sort_keys=True, allow_nan=False) + '\n'
        self.file.write(line)
        self.file.flush()
        self.bytes_written += len(line.encode())
        self.write_seconds += time.monotonic() - started

    def add(self, kind: str, producer: str, inputs: tuple[str, ...], step: int, content: dict):
        for parent in inputs:
            if parent not in self.artifacts:
                raise ValueError(f'Unknown parent: {parent}')
        artifact = Artifact(f'a{len(self.artifacts):06d}', kind, producer, tuple(inputs), step,
                            json_copy(content))
        self.write({'record': 'artifact', **asdict(artifact)})
        self.artifacts[artifact.id] = artifact
        self.children[artifact.id] = set()
        for parent in inputs:
            self.children[parent].add(artifact.id)
        return artifact

    def event(self, kind: str, step: int, ids=(), **details):
        if not set(ids) <= self.artifacts.keys():
            raise ValueError('Unknown artifact reference in event')
        event = ExecutionEvent(self.sequence, kind, step, tuple(ids),
                               time.monotonic() - self.started, json_copy(details))
        self.write({'record': 'event', **asdict(event)})
        self.sequence += 1

    def descendants(self, artifact_id: str) -> set[str]:
        pending = list(self.children[artifact_id])
        found = set()
        while pending:
            node = pending.pop()
            if node not in found:
                found.add(node)
                pending.extend(self.children[node])
        return found

    def close(self):
        self.file.close()
