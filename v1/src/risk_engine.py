"""
BRIDGE Risk Decision Engine (Phase 2.3)
Deterministic operational policy mapping continuous failure probabilities and delivery context
into LOW / MEDIUM / HIGH risk bands with prescribed intervention actions and non-causal explanations.
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional


DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "models" / "risk_policy_v1.json"
if not DEFAULT_POLICY_PATH.exists():
    # If running from src/bridge/
    DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "risk_policy_v1.json"


class RiskEngine:
    """
    Deterministic operational risk decision engine for BRIDGE.
    Consumes calibrated first-attempt delivery failure probability and delivery/customer context.
    """

    POLICY_VERSION = "v1.0.0"
    DEFAULT_LOW_CUTOFF = 0.080
    DEFAULT_HIGH_CUTOFF = 0.120

    def __init__(self, policy_path: Optional[Path] = None, policy_dict: Optional[Dict[str, Any]] = None):
        self.policy = {}
        if policy_dict is not None:
            self.policy = policy_dict
        elif policy_path is not None and Path(policy_path).exists():
            with open(policy_path, "r", encoding="utf-8") as f:
                self.policy = json.load(f)
        elif DEFAULT_POLICY_PATH.exists():
            with open(DEFAULT_POLICY_PATH, "r", encoding="utf-8") as f:
                self.policy = json.load(f)

        # Extract thresholds with safe fallbacks
        thresholds = self.policy.get("thresholds", {})
        self.low_cutoff = float(thresholds.get("low_cutoff", self.DEFAULT_LOW_CUTOFF))
        self.high_cutoff = float(thresholds.get("high_cutoff", self.DEFAULT_HIGH_CUTOFF))
        self.version = self.policy.get("policy_version", self.POLICY_VERSION)

    def classify_risk(self, probability: float) -> str:
        """
        Classify continuous probability into operational risk bands: LOW / MEDIUM / HIGH.
        Raises ValueError if probability is outside [0.0, 1.0] or non-numeric.
        """
        if not isinstance(probability, (int, float)):
            raise TypeError(f"Failure probability must be a numeric float, got {type(probability)}")
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"Failure probability must be between 0.0 and 1.0 inclusive, got {probability}")

        if probability < self.low_cutoff:
            return "LOW"
        elif probability < self.high_cutoff:
            return "MEDIUM"
        else:
            return "HIGH"

    def generate_explanations(self, probability: float, context: Optional[Dict[str, Any]] = None) -> Tuple[List[str], str]:
        """
        Generate human-readable reason codes and non-causal risk explanations.
        Phrasing explicitly adheres to: 'Model-associated risk factors include...'
        """
        if context is None:
            context = {}

        reason_codes: List[str] = []
        explanation_phrases: List[str] = []

        # 1. Customer Availability & Historical Friction
        avail = context.get("customer_availability_rate")
        if avail is not None and float(avail) < 0.70:
            reason_codes.append("LOW_CUSTOMER_AVAILABILITY")
            explanation_phrases.append("lower estimated customer availability")

        prev_fails = context.get("customer_previous_failed_attempts")
        if prev_fails is not None and int(prev_fails) >= 2:
            reason_codes.append("HIGH_PREVIOUS_FAILED_ATTEMPTS")
            explanation_phrases.append(f"history of prior delivery failures ({int(prev_fails)} attempts)")

        succ_rate = context.get("customer_previous_success_rate")
        attempts = context.get("customer_previous_delivery_attempts", 1)
        if succ_rate is not None and float(succ_rate) < 0.85 and attempts and int(attempts) > 0:
            if "HIGH_PREVIOUS_FAILED_ATTEMPTS" not in reason_codes:
                reason_codes.append("LOW_PREVIOUS_SUCCESS_RATE")
                explanation_phrases.append("below-average historical delivery success rate")

        # 2. Access Difficulty & Delivery Method
        access = context.get("address_access_difficulty")
        if access is not None and int(access) >= 4:
            reason_codes.append("HIGH_ADDRESS_ACCESS_DIFFICULTY")
            explanation_phrases.append(f"elevated premises access difficulty (level {int(access)}/5)")

        method = context.get("preferred_delivery_method")
        if method == "Hand to Customer":
            reason_codes.append("HAND_TO_CUSTOMER_METHOD")
            explanation_phrases.append("recipient in-person handoff requirement")

        sig = context.get("signature_required")
        if sig is not None and int(sig) == 1:
            reason_codes.append("SIGNATURE_REQUIRED")
            explanation_phrases.append("mandatory physical signature requirement")

        # 3. Preference Mismatch
        mismatch = context.get("preference_departure_mismatch")
        pref = context.get("customer_delivery_preference")
        dep_hour = context.get("departure_hour")
        is_weekend = context.get("is_weekend")

        is_mismatch = False
        if mismatch is not None and int(mismatch) == 1:
            is_mismatch = True
        elif pref == "Morning Only" and dep_hour is not None and int(dep_hour) >= 13:
            is_mismatch = True
        elif pref == "Weekday Only" and is_weekend is not None and int(is_weekend) == 1:
            is_mismatch = True

        if is_mismatch:
            reason_codes.append("PREFERENCE_DEPARTURE_MISMATCH")
            explanation_phrases.append(f"dispatch timing mismatch with '{pref}' preference")

        # 4. Environmental & Spatial Factors
        zone_rate = context.get("historical_zone_failure_rate")
        if zone_rate is not None and float(zone_rate) >= 0.060:
            reason_codes.append("HIGH_HISTORICAL_ZONE_FAILURE")
            explanation_phrases.append(f"elevated zone failure friction ({float(zone_rate)*100:.1f}%)")

        traffic = context.get("traffic_risk_score")
        if traffic is not None and float(traffic) >= 0.60:
            reason_codes.append("ELEVATED_TRAFFIC_RISK")
            explanation_phrases.append("elevated traffic corridor risk")

        weather = context.get("weather_risk_score")
        if weather is not None and float(weather) >= 0.50:
            reason_codes.append("ELEVATED_WEATHER_RISK")
            explanation_phrases.append("weather disturbance risk in target sector")

        vol = context.get("package_volume_cm3")
        if vol is not None and float(vol) >= 25000:
            reason_codes.append("LARGE_PACKAGE_VOLUME")
            explanation_phrases.append("large package volume")

        # Synthesize Human-Readable Non-Causal Explanation
        risk_level = self.classify_risk(probability)
        if explanation_phrases:
            factors_str = ", ".join(explanation_phrases)
            explanation = f"Model-associated risk factors for {risk_level} tier ({probability*100:.1f}%) include: {factors_str}."
        else:
            if risk_level == "LOW":
                explanation = f"Low first-attempt failure risk ({probability*100:.1f}%). Favorable customer availability and standard access indicators."
            elif risk_level == "MEDIUM":
                explanation = f"Moderate first-attempt failure risk ({probability*100:.1f}%). Reflects aggregate routing and baseline operational variance."
            else:
                explanation = f"Elevated first-attempt failure risk ({probability*100:.1f}%). High aggregate risk score across multiple routing dimensions."

        return reason_codes, explanation

    def recommend_intervention(self, risk_level: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Determine deterministic intervention action and descriptive details
        based on risk tier, customer preference, and physical delivery context.
        """
        if context is None:
            context = {}

        pref = context.get("customer_delivery_preference")
        method = context.get("preferred_delivery_method")
        sig = int(context.get("signature_required", 0))
        mismatch = int(context.get("preference_departure_mismatch", 0))
        dep_hour = context.get("departure_hour")
        is_weekend = int(context.get("is_weekend", 0))
        access = int(context.get("address_access_difficulty", 1))
        avail = float(context.get("customer_availability_rate", 0.85))

        # Check explicit mismatch conditions
        if pref == "Morning Only" and dep_hour is not None and int(dep_hour) >= 13:
            mismatch = 1
        elif pref == "Weekday Only" and is_weekend == 1:
            mismatch = 1

        if risk_level == "LOW":
            if sig == 1:
                return (
                    "STANDARD_DISPATCH_SIGNATURE_PROMPT",
                    "Normal dispatch to van; unattended delivery prohibited. Driver will prompt recipient for physical signature upon delivery."
                )
            else:
                return (
                    "STANDARD_DISPATCH",
                    "Normal dispatch to van; unattended delivery authorized to designated drop point without dispatcher intervention."
                )

        elif risk_level == "MEDIUM":
            if mismatch == 1:
                return (
                    "APPOINTMENT_TIMING_CONFIRMATION",
                    "Trigger automated SMS to confirm recipient availability outside requested delivery window before departure."
                )
            elif access >= 4:
                return (
                    "REQUEST_GATE_CODE_INSTRUCTIONS",
                    "Send automated SMS requesting property gate code, access badge notes, or specific drop instructions prior to van departure."
                )
            elif method == "Hand to Customer" or sig == 1 or avail < 0.70:
                return (
                    "CONFIRM_PRESENCE_AND_BACKUP_LOCATION",
                    "Send proactive SMS asking recipient to confirm in-person presence or authorize a designated neighbor / safe drop place."
                )
            else:
                return (
                    "CUSTOMER_NOTIFICATION",
                    "Send proactive customer SMS notification with estimated delivery arrival window to encourage recipient availability."
                )

        elif risk_level == "HIGH":
            if mismatch == 1:
                return (
                    "RESCHEDULE_DELIVERY_WINDOW",
                    "Pre-dispatch hold: Prompt customer via SMS/app to reschedule delivery to a compatible time window or preferred delivery day."
                )
            elif pref == "Evening Only" and dep_hour is not None and int(dep_hour) < 15:
                return (
                    "HOLD_FOR_EVENING_SHIFT",
                    "Pre-dispatch hold: Hold parcel at fulfillment station and re-stage for late-afternoon / evening delivery route."
                )
            elif method == "Hand to Customer" or sig == 1:
                return (
                    "ACCESS_POINT_REDIRECT",
                    "High presence friction: Recommend pre-dispatch rerouting to nearby Amazon Locker / Access Point for secure customer collection."
                )
            elif access >= 4:
                return (
                    "REROUTE_TO_BUILDING_MANAGEMENT_OR_LOCKER",
                    "Severe premises access barrier: Redirect parcel to property management, building leasing office, or on-site locker."
                )
            else:
                return (
                    "EXPEDITED_CONFIRMATION_OR_LOCKER",
                    "Mandatory pre-dispatch customer verification: Require customer authorization for alternate safe place, neighbor, or 1-click locker reroute."
                )

        else:
            raise ValueError(f"Unknown risk level '{risk_level}'. Expected 'LOW', 'MEDIUM', or 'HIGH'.")

    def evaluate_delivery(self, probability: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete end-to-end evaluation pipeline:
        Classifies risk, selects deterministic intervention, and generates explainability factors.
        Returns a clean, JSON-serializable dictionary.
        """
        if context is None:
            context = {}

        risk_level = self.classify_risk(probability)
        action, action_details = self.recommend_intervention(risk_level, context)
        reason_codes, explanation = self.generate_explanations(probability, context)

        return {
            "failure_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "recommended_action": action,
            "action_details": action_details,
            "reason_codes": reason_codes,
            "explanation": explanation,
            "policy_version": self.version,
        }


# Module-level convenience functions mirroring the singleton engine
_default_engine = RiskEngine()

def classify_risk(probability: float) -> str:
    return _default_engine.classify_risk(probability)

def recommend_intervention(risk_level: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    return _default_engine.recommend_intervention(risk_level, context)

def generate_explanations(probability: float, context: Optional[Dict[str, Any]] = None) -> Tuple[List[str], str]:
    return _default_engine.generate_explanations(probability, context)

def evaluate_delivery(probability: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _default_engine.evaluate_delivery(probability, context)
