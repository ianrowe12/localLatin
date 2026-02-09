from __future__ import annotations

import re
from typing import Iterable, List


def parse_layers(value: str) -> List[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    layers: List[int] = []
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def extract_layer_numbers(paths: Iterable[str], pattern: str) -> List[int]:
    regex = re.compile(pattern)
    layers = []
    for path in paths:
        match = regex.search(path)
        if match:
            layers.append(int(match.group(1)))
    return sorted(set(layers))
