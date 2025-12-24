"""Shared utilities and components for FortunePIA.

This package contains modules and functions used across different bounded contexts.
The shared kernel follows strict dependency rules: contexts depend on shared,
but shared NEVER depends on contexts.
"""

# Use relative imports within shared module
from .core.logger import setup_logger
from .llm.cost_logger import CostLogger
from .llm.llm_manager import LLMManager

__all__ = [
    "setup_logger",
    "LLMManager",
    "CostLogger",
]
