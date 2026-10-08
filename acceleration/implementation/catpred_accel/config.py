"""Runtime policy; independent of torch and available hardware."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    numeric: str = "exact"
    memory: str = "stream"
    backend: str = "auto"
    input_budget_bytes: int = 2 << 30
    fallback: str = "stock"
    allow_unvalidated: bool = False

    def __post_init__(self):
        if self.numeric not in ("off", "exact"):
            raise ValueError("Only off/exact are implemented; fast has no accepted contract")
        if self.memory != "stream":
            raise ValueError("This release supports bounded stream memory policy")
        if self.backend not in ("auto", "g4", "h100", "generic", "cpu"):
            raise ValueError("Unknown backend profile")
        if isinstance(self.input_budget_bytes, bool) or not isinstance(self.input_budget_bytes, int) or self.input_budget_bytes <= 0:
            raise ValueError("Input budget must be positive integer bytes")
        if not isinstance(self.allow_unvalidated, bool):
            raise ValueError("allow_unvalidated must be an explicit boolean")
        if self.fallback not in ("stock", "raise"):
            raise ValueError("Fallback must be stock or raise")
