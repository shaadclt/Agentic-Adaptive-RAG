"""
Prompt injection detection for user input.

This is a defensive heuristic layer. It is not intended to provide
perfect classification of all prompt injection attacks.
"""

import re
from dataclasses import dataclass
from typing import List

from security.security_config import (
    MAX_QUESTION_LENGTH,
    PROMPT_INJECTION_PATTERNS,
)


@dataclass
class PromptGuardResult:
    allowed: bool
    reason: str = ""
    matched_patterns: List[str] | None = None


def check_prompt(prompt: str) -> PromptGuardResult:
    """
    Inspect user input for common direct prompt-injection attempts.
    """

    if not isinstance(prompt, str):
        return PromptGuardResult(
            allowed=False,
            reason="Question must be a string.",
            matched_patterns=[],
        )

    normalized = prompt.strip()

    if not normalized:
        return PromptGuardResult(
            allowed=False,
            reason="Question cannot be empty.",
            matched_patterns=[],
        )

    if len(normalized) > MAX_QUESTION_LENGTH:
        return PromptGuardResult(
            allowed=False,
            reason=(
                f"Question exceeds the maximum allowed length "
                f"of {MAX_QUESTION_LENGTH} characters."
            ),
            matched_patterns=[],
        )

    matches: List[str] = []

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):
            matches.append(pattern)

    if matches:
        return PromptGuardResult(
            allowed=False,
            reason=(
                "The request contains patterns associated "
                "with prompt-injection or instruction-override attacks."
            ),
            matched_patterns=matches,
        )

    return PromptGuardResult(
        allowed=True,
        reason="No high-confidence prompt-injection pattern detected.",
        matched_patterns=[],
    )