from __future__ import annotations

import os
import time


def generate_outputs(
    prompt: str,
    n: int,
    model: str = "llama-3.3-70b-versatile",
    api_key: str | None = None,
    system_prompt: str | None = None,
    delay: float = 2.2,
) -> list[str]:
    """Generate n outputs from Groq using the groq SDK.

    Respects the 30 RPM free-tier limit with a configurable delay between calls.
    Returns a list of output strings.
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(
            "groq SDK not installed — run: pip install spaniq[groq]"
        ) from exc

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=key)
    outputs: list[str] = []

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for i in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        outputs.append(resp.choices[0].message.content or "")
        if i < n - 1:
            time.sleep(delay)

    return outputs
