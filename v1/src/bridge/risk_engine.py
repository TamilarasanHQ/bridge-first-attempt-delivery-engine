"""
BRIDGE Risk Decision Engine (src/bridge/risk_engine.py)
Re-exports the core risk engine implementation for packaged usage.
"""

from pathlib import Path
import sys

# Ensure root is on path for direct imports
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from risk_engine import (
    RiskEngine,
    classify_risk,
    recommend_intervention,
    generate_explanations,
    evaluate_delivery,
)

__all__ = [
    "RiskEngine",
    "classify_risk",
    "recommend_intervention",
    "generate_explanations",
    "evaluate_delivery",
]
