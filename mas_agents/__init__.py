"""
Multi-Agent System (MAS) architectures for BrowseComp-Plus evaluation.

Implements 5 canonical architectures from Kim et al. (2025),
"Towards a Science of Scaling Agent Systems":
  - Single-Agent System (SAS)
  - Independent MAS
  - Centralized MAS
  - Decentralized MAS
  - Hybrid MAS
"""

from .single_agent import CopilotSingleAgent
from .multi_agents import (
    CopilotIndependentAgent,
    CopilotCentralizedAgent,
    CopilotDecentralizedAgent,
    CopilotHybridAgent,
)

__all__ = [
    "CopilotSingleAgent",
    "CopilotIndependentAgent",
    "CopilotCentralizedAgent",
    "CopilotDecentralizedAgent",
    "CopilotHybridAgent",
]
