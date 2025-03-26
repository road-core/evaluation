from dataclasses import dataclass
from langchain_core.runnables import Runnable


@dataclass
class MockResult:
    """Mock of LLM response"""

    content: str


class MockLLM(Runnable):
    """Mock LLM class"""

    def invoke(self, input, config=None, **kwargs):
        return MockResult("5")
