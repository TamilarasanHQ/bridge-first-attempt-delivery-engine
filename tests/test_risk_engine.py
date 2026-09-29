"""
Unit Tests for BRIDGE Risk Decision Engine (Phase 2.3)
Verifies threshold boundaries, deterministic interventions, context handling, and strict leakage safety.
"""

from pathlib import Path
import pytest
import numpy as np
from risk_engine import (
    RiskEngine,
    classify_risk,
    recommend_intervention,
    generate_explanations,
    evaluate_delivery,
)


@pytest.fixture
def engine():
    return RiskEngine()


# ------------------------------------------------------------------------------
# 1. Classification & Threshold Boundary Tests
# ------------------------------------------------------------------------------
def test_risk_classification_standard(engine):
    assert engine.classify_risk(0.045) == "LOW"
    assert engine.classify_risk(0.095) == "MEDIUM"
    assert engine.classify_risk(0.135) == "HIGH"


def test_exact_threshold_boundaries(engine):
    # LOW < 0.080
    assert engine.classify_risk(0.079999) == "LOW"
    assert engine.classify_risk(0.080000) == "MEDIUM"

    # MEDIUM < 0.120
    assert engine.classify_risk(0.119999) == "MEDIUM"
    assert engine.classify_risk(0.120000) == "HIGH"

    # Extremes
    assert engine.classify_risk(0.0) == "LOW"
    assert engine.classify_risk(1.0) == "HIGH"


def test_invalid_probabilities(engine):
    with pytest.raises(ValueError):
        engine.classify_risk(-0.01)

    with pytest.raises(ValueError):
        engine.classify_risk(1.01)

    with pytest.raises(TypeError):
        engine.classify_risk("0.05")  # type: ignore


# ------------------------------------------------------------------------------
# 2. Missing & Optional Context Handling
# ------------------------------------------------------------------------------
def test_missing_context_evaluation(engine):
    # Engine should safely evaluate with None or empty context dict
    res_none = engine.evaluate_delivery(0.05, None)
    res_empty = engine.evaluate_delivery(0.05, {})

    assert res_none["risk_level"] == "LOW"
    assert res_none["recommended_action"] == "STANDARD_DISPATCH"
    assert res_empty["recommended_action"] == "STANDARD_DISPATCH"
    assert isinstance(res_none["reason_codes"], list)
    assert isinstance(res_none["explanation"], str)


def test_partial_context_evaluation(engine):
    context = {"customer_delivery_preference": "Anytime"}
    res = engine.evaluate_delivery(0.10, context)
    assert res["risk_level"] == "MEDIUM"
    assert res["recommended_action"] == "CUSTOMER_NOTIFICATION"


# ------------------------------------------------------------------------------
# 3. Deterministic Intervention Logic
# ------------------------------------------------------------------------------
def test_low_risk_interventions(engine):
    # Standard low risk
    action, _ = engine.recommend_intervention("LOW", {"signature_required": 0})
    assert action == "STANDARD_DISPATCH"

    # Low risk with signature required
    action_sig, _ = engine.recommend_intervention("LOW", {"signature_required": 1})
    assert action_sig == "STANDARD_DISPATCH_SIGNATURE_PROMPT"


def test_medium_risk_interventions(engine):
    # Generic medium risk
    action, _ = engine.recommend_intervention("MEDIUM", {})
    assert action == "CUSTOMER_NOTIFICATION"

    # Medium risk with gate access difficulty
    action_gate, _ = engine.recommend_intervention("MEDIUM", {"address_access_difficulty": 5})
    assert action_gate == "REQUEST_GATE_CODE_INSTRUCTIONS"

    # Medium risk with preference mismatch
    action_mismatch, _ = engine.recommend_intervention("MEDIUM", {"preference_departure_mismatch": 1})
    assert action_mismatch == "APPOINTMENT_TIMING_CONFIRMATION"

    # Medium risk with hand to customer and low availability
    action_handoff, _ = engine.recommend_intervention("MEDIUM", {
        "preferred_delivery_method": "Hand to Customer",
        "customer_availability_rate": 0.50
    })
    assert action_handoff == "CONFIRM_PRESENCE_AND_BACKUP_LOCATION"


def test_high_risk_interventions(engine):
    # High risk with preference mismatch -> Reschedule
    action_resched, _ = engine.recommend_intervention("HIGH", {"preference_departure_mismatch": 1})
    assert action_resched == "RESCHEDULE_DELIVERY_WINDOW"

    # High risk with Evening Only and early departure -> Hold for evening
    action_evening, _ = engine.recommend_intervention("HIGH", {
        "customer_delivery_preference": "Evening Only",
        "departure_hour": 10
    })
    assert action_evening == "HOLD_FOR_EVENING_SHIFT"

    # High risk with Hand to Customer or Signature -> Access Point Redirect
    action_locker, _ = engine.recommend_intervention("HIGH", {"preferred_delivery_method": "Hand to Customer"})
    assert action_locker == "ACCESS_POINT_REDIRECT"

    action_sig_high, _ = engine.recommend_intervention("HIGH", {"signature_required": 1})
    assert action_sig_high == "ACCESS_POINT_REDIRECT"

    # High risk with severe access difficulty -> Reroute to building guard/locker
    action_guard, _ = engine.recommend_intervention("HIGH", {"address_access_difficulty": 4})
    assert action_guard == "REROUTE_TO_BUILDING_MANAGEMENT_OR_LOCKER"

    # High risk generic -> Expedited confirmation or locker
    action_gen, _ = engine.recommend_intervention("HIGH", {})
    assert action_gen == "EXPEDITED_CONFIRMATION_OR_LOCKER"


def test_unknown_risk_level_raises(engine):
    with pytest.raises(ValueError):
        engine.recommend_intervention("CRITICAL", {})


# ------------------------------------------------------------------------------
# 4. Explanations & Reason Codes
# ------------------------------------------------------------------------------
def test_reason_codes_generation(engine):
    context = {
        "customer_availability_rate": 0.45,
        "signature_required": 1,
        "address_access_difficulty": 5,
        "traffic_risk_score": 0.85,
        "preference_departure_mismatch": 1,
    }
    codes, explanation = engine.generate_explanations(0.14, context)

    assert "LOW_CUSTOMER_AVAILABILITY" in codes
    assert "SIGNATURE_REQUIRED" in codes
    assert "HIGH_ADDRESS_ACCESS_DIFFICULTY" in codes
    assert "ELEVATED_TRAFFIC_RISK" in codes
    assert "PREFERENCE_DEPARTURE_MISMATCH" in codes

    # Non-causal phrasing assertion
    assert "Model-associated risk factors" in explanation
    assert "causes" not in explanation.lower()


# ------------------------------------------------------------------------------
# 5. Full evaluate_delivery Output Schema
# ------------------------------------------------------------------------------
def test_evaluate_delivery_schema(engine):
    context = {
        "customer_availability_rate": 0.60,
        "preferred_delivery_method": "Hand to Customer",
        "address_access_difficulty": 2,
    }
    result = engine.evaluate_delivery(0.125, context)

    assert isinstance(result, dict)
    assert result["failure_probability"] == 0.125
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "ACCESS_POINT_REDIRECT"
    assert isinstance(result["action_details"], str)
    assert isinstance(result["reason_codes"], list)
    assert "HAND_TO_CUSTOMER_METHOD" in result["reason_codes"]
    assert "LOW_CUSTOMER_AVAILABILITY" in result["reason_codes"]
    assert result["policy_version"] == "v1.0.0"


# ------------------------------------------------------------------------------
# 6. Leakage & TEST Isolation Assertions
# ------------------------------------------------------------------------------
def test_test_set_isolation():
    base_dir = Path(__file__).resolve().parent.parent
    val_idx_path = base_dir / "v1" / "models" / "val_indices.npy" if (base_dir / "v1" / "models" / "val_indices.npy").exists() else base_dir / "models" / "val_indices.npy"
    test_idx_path = base_dir / "v1" / "models" / "test_indices.npy" if (base_dir / "v1" / "models" / "test_indices.npy").exists() else base_dir / "models" / "test_indices.npy"

    assert val_idx_path.exists(), "Validation indices file missing!"
    assert test_idx_path.exists(), "Test indices file missing!"

    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    # Disjointness check
    overlap = set(val_idx).intersection(set(test_idx))
    assert len(overlap) == 0, f"LEAKAGE DEFECT: {len(overlap)} indices overlap between val and test!"
    assert len(val_idx) == 246432
    assert len(test_idx) == 224881
