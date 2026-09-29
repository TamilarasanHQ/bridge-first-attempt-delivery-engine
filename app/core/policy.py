"""
Deterministic Risk Policy Engine.
Maps continuous failure probabilities to operational risk bands and prescribes interventions.
"""
from typing import Dict, Any, List
from app.core.config import settings


class RiskPolicyEngine:
    """
    Deterministic operational policy mapping continuous failure probabilities
    into LOW / MEDIUM / HIGH risk bands with prescribed intervention actions.
    """
    def __init__(
        self,
        low_threshold: float = settings.LOW_RISK_THRESHOLD,
        high_threshold: float = settings.HIGH_RISK_THRESHOLD
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(self, prob: float, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a parcel's predicted probability against operational thresholds.
        """
        if prob >= self.high_threshold:
            tier = "HIGH"
            priority = "URGENT"
            intervention_required = True
            action = "CUSTOMER_PREFERENCE_REQUIRED"
            allowed_options = [
                {
                    "id": "ACCESS_POINT",
                    "title": "Deliver to Nearby Access Point / Locker",
                    "description": "Secure automated pickup locker within 1.5 miles. Guaranteed successful first attempt."
                },
                {
                    "id": "HOLD_PICKUP",
                    "title": "Hold for Pickup at Fulfillment Depot",
                    "description": "Hold package at station for direct customer collection."
                },
                {
                    "id": "RESCHEDULE",
                    "title": "Deliver on Another Date",
                    "description": "Customer chooses a preferred day when guaranteed to be available."
                },
                {
                    "id": "NEIGHBOR",
                    "title": "Deliver to Designated Neighbor",
                    "description": "Authorize handoff to adjacent neighbor or apartment building manager."
                },
                {
                    "id": "CONFIRM_AVAILABLE",
                    "title": "Confirm Availability & Provide Gate Code",
                    "description": "Customer confirms physical presence and provides building/gate code."
                }
            ]
            summary = "High failure probability detected. Pre-dispatch customer preference engagement triggered."
        elif prob >= self.low_threshold:
            tier = "MEDIUM"
            priority = "ADVISORY"
            intervention_required = False
            action = "PROACTIVE_SMS_REMINDER"
            allowed_options = [
                {
                    "id": "CONFIRM_AVAILABLE",
                    "title": "Confirm Availability",
                    "description": "Soft check-in confirming recipient is available for the delivery window."
                },
                {
                    "id": "ACCESS_POINT",
                    "title": "Optional Locker Reroute",
                    "description": "Optional convenience transfer to nearby pickup point."
                }
            ]
            summary = "Moderate failure probability. Send proactive tracking alert to recipient."
        else:
            tier = "LOW"
            priority = "ROUTINE"
            intervention_required = False
            action = "STANDARD_DISPATCH"
            allowed_options = []
            summary = "Low failure probability. Proceed with automated standard vehicle loading."

        return {
            "risk_tier": tier,
            "priority": priority,
            "intervention_required": intervention_required,
            "operational_action": action,
            "allowed_options": allowed_options,
            "policy_summary": summary
        }


policy_engine = RiskPolicyEngine()
