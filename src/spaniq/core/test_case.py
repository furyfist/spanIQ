from dataclasses import dataclass, field


@dataclass
class LLMTestCase:
    """A single evaluation test case.

    baseline_outputs defines the reference distribution — what normal looks like.
    Collect 10-50 outputs from your LLM when it was working correctly, once.
    Every future eval compares actual_output statistically against this distribution.
    """

    input: str
    actual_output: str
    expected_output: str | None = None
    baseline_outputs: list[str] = field(default_factory=list)
    retrieval_context: list[str] | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if not self.input:
            raise ValueError("input cannot be empty")
        if not self.actual_output:
            raise ValueError("actual_output cannot be empty")
