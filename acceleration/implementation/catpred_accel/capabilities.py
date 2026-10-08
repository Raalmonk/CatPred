"""Pure metadata dispatch. Eligibility is not a numerical validation result."""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from .config import RuntimeConfig


class UnsupportedCapability(RuntimeError):
    pass


@dataclass(frozen=True)
class Capability:
    device_type: str
    compute_capability: Optional[Tuple[int, int]] = None
    device_name: str = "unknown"
    source_compatible: bool = False
    rust_available: bool = False
    dtype: str = "float32"
    software_key: str = "unknown"
    validated_context: bool = False


@dataclass(frozen=True)
class BackendDecision:
    requested_backend: str
    selected_backend: str
    hardware_profile: str
    numeric: str
    engaged: bool
    reason: str
    validation_status: str = "not_run"

    def as_dict(self):
        return asdict(self)


def resolve(config, capability):
    if not isinstance(config, RuntimeConfig) or not isinstance(capability, Capability):
        raise TypeError("Expected RuntimeConfig and Capability metadata")
    profile = ({(12, 0): "g4_sm120", (9, 0): "h100_sm90"}.get(capability.compute_capability, "generic_cuda")
               if capability.device_type == "cuda" else "cpu" if capability.device_type == "cpu" else "unknown")
    reason = None
    if config.numeric == "off":
        return BackendDecision(config.backend, "stock", profile, "off", False, "Numeric policy explicitly off")
    if config.backend == "g4" and profile != "g4_sm120":
        reason = "Requested G4 requires observed compute capability 12.0"
    elif config.backend == "h100" and profile != "h100_sm90":
        reason = "Requested H100 requires observed compute capability 9.0"
    elif config.backend == "cpu" and capability.device_type != "cpu":
        reason = "Requested CPU does not match the model device"
    elif capability.device_type != "cuda":
        reason = "CUDA streaming unavailable; original CPU/non-CUDA path retained"
    elif capability.dtype != "float32":
        reason = "Accelerated exact path requires unchanged FP32 models"
    elif not capability.source_compatible:
        reason = "Pinned scientific source or input configuration is unsupported"
    elif not capability.rust_available:
        reason = "Accepted native Rust packing extension unavailable"
    elif capability.software_key == "unknown":
        reason = "Complete software/context identity unavailable"
    elif profile == "generic_cuda" and config.backend != "generic":
        reason = "Unknown CUDA architecture; select generic explicitly to investigate"
    elif not capability.validated_context and not config.allow_unvalidated:
        reason = "Hardware/software/source/model context has no matching exact validation certificate"
    if reason:
        if config.fallback == "raise":
            raise UnsupportedCapability(reason)
        return BackendDecision(config.backend, "stock", profile, "exact", False, reason)
    return BackendDecision(config.backend, "accepted_stream", profile, "exact", True,
                           "Exact validated context" if capability.validated_context else "Explicit benchmark-only unvalidated execution",
                           "verified" if capability.validated_context else "experimental_unvalidated")
