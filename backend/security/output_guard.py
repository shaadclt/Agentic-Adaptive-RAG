"""
Output security layer.

Detects and redacts common credentials, tokens, and sensitive
configuration values before an answer is returned to the user.
"""

import re
from dataclasses import dataclass

from backend.security.security_config import (
    MAX_OUTPUT_LENGTH,
    SECRET_PATTERNS,
    SENSITIVE_ENVIRONMENT_VARIABLES,
)


@dataclass
class OutputGuardResult:
    safe: bool
    output: str
    reason: str = ""
    redactions: int = 0


def sanitize_output(
    output: str,
) -> OutputGuardResult:
    """
    Detect and redact common secrets from model output.
    """

    if output is None:
        return OutputGuardResult(
            safe=True,
            output="",
        )

    text = str(output)

    redactions = 0

    # Protect known environment variable names.
    for variable in SENSITIVE_ENVIRONMENT_VARIABLES:
        pattern = (
            rf"({re.escape(variable)}\s*=\s*)"
            rf"([^\s\"'`]+)"
        )

        text, count = re.subn(
            pattern,
            r"\1[REDACTED]",
            text,
            flags=re.IGNORECASE,
        )

        redactions += count

    # Protect known credential/token formats.
    for pattern in SECRET_PATTERNS:
        text, count = re.subn(
            pattern,
            "[REDACTED_SECRET]",
            text,
            flags=re.IGNORECASE,
        )

        redactions += count

    # Avoid returning unexpectedly huge model output.
    if len(text) > MAX_OUTPUT_LENGTH:
        text = (
            text[:MAX_OUTPUT_LENGTH]
            + "\n\n[Output truncated by security policy.]"
        )

    if redactions:
        return OutputGuardResult(
            safe=False,
            output=text,
            reason=(
                "Potential credential or sensitive configuration "
                "data was detected and redacted."
            ),
            redactions=redactions,
        )

    return OutputGuardResult(
        safe=True,
        output=text,
        reason="No known secret pattern detected.",
        redactions=0,
    )