"""Approximate token counting using tiktoken.

We use cl100k_base — close enough to the Alem tokenizer for budget purposes.
TASK.md §3 says *"approximate is fine; don't blow past it by 2x"* — we aim
for 0.95 × max_tokens with this counter as a safety margin.
"""

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text))


def fits(text: str, budget: int) -> bool:
    return count(text) <= budget
