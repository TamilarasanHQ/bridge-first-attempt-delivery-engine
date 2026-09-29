"""
Customer preference engagement and intervention resolution endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.schemas.preference import CustomerPreferenceSubmission, PreferenceConfirmation
from app.core.state import (
    interventions_db,
    latest_manifest_assessments,
    record_customer_preference,
    trigger_sms
)

router = APIRouter()


@router.get("/package/{package_id}")
def get_package_for_portal(package_id: str):
    """
    Retrieves package details and allowed preference options for the customer-facing portal.
    """
    if package_id in latest_manifest_assessments:
        data = latest_manifest_assessments[package_id]
        intervention_info = interventions_db.get(package_id, {})
        return {
            "found": True,
            "package_id": package_id,
            "recipient_name": data.get("recipient_name", "Valued Customer"),
            "address_line": data.get("address_line", "Destination Address"),
            "risk_tier": data.get("risk_tier", "MEDIUM"),
            "policy_summary": data.get("policy_summary", "Delivery check-in"),
            "allowed_options": data.get("allowed_options", []),
            "current_status": intervention_info.get("status", data.get("status", "PENDING_DISPATCH")),
            "selected_preference": intervention_info.get("selected_preference")
        }
    
    # Fallback default demo package if not in recent manifest
    return {
        "found": True,
        "package_id": package_id,
        "recipient_name": "Valued Customer",
        "address_line": "888 S Hope St, High-Rise Unit 1804, Los Angeles, CA",
        "risk_tier": "HIGH",
        "policy_summary": "High risk of delivery failure detected prior to vehicle dispatch.",
        "allowed_options": [
            {
                "id": "ACCESS_POINT",
                "title": "Deliver to Nearby Access Point / Locker",
                "description": "Secure 24/7 automated pickup locker (0.8 mi away, Ralphs Market Hub)."
            },
            {
                "id": "HOLD_PICKUP",
                "title": "Hold for Pickup at Fulfillment Depot",
                "description": "Hold package at station DLA3 for direct customer pickup."
            },
            {
                "id": "RESCHEDULE",
                "title": "Deliver on Another Date",
                "description": "Select tomorrow or a weekend date when you will be home."
            },
            {
                "id": "NEIGHBOR",
                "title": "Deliver to Designated Neighbor",
                "description": "Authorize handoff to building manager, front desk, or neighbor."
            },
            {
                "id": "CONFIRM_AVAILABLE",
                "title": "Confirm Availability & Provide Gate Code",
                "description": "Confirm you will be home and provide door or gate buzzer instructions."
            }
        ],
        "current_status": "NOTIFICATION_SENT",
        "selected_preference": None
    }


@router.post("/trigger-notification/{package_id}")
def trigger_notification(package_id: str):
    """
    Simulates sending an automated SMS / Push alert to the recipient of a high-risk package.
    """
    res = trigger_sms(package_id)
    if not res:
        # Fallback simulation
        return {
            "success": True,
            "package_id": package_id,
            "status": "NOTIFICATION_SENT",
            "message": f"Pre-dispatch SMS alert sent to customer for {package_id}. Preference portal link activated."
        }
    return {
        "success": True,
        "package_id": package_id,
        "status": res.get("status", "NOTIFICATION_SENT"),
        "message": f"Pre-dispatch notification dispatched for {package_id}."
    }


@router.post("/preferences", response_model=PreferenceConfirmation)
def submit_customer_preference(submission: CustomerPreferenceSubmission):
    """
    Customer submits their desired delivery preference via the customer portal.
    """
    res = record_customer_preference(submission.package_id, submission.model_dump())
    
    routing_map = {
        "ACCESS_POINT": "REROUTED: Package assigned to Hub Locker #418",
        "HOLD_PICKUP": "HELD: Package staged at Hub DLA3 Will-Call",
        "RESCHEDULE": f"RESCHEDULED: Scheduled for dispatch on {submission.reschedule_date or 'Next Business Day'}",
        "NEIGHBOR": f"AUTHORIZED: Driver instructed to handoff to {submission.neighbor_notes or 'Neighbor/Front Desk'}",
        "CONFIRM_AVAILABLE": f"CONFIRMED: Driver briefed with gate code '{submission.gate_code or 'VERIFIED'}'"
    }
    
    routing_msg = routing_map.get(submission.selected_option, "ROUTING UPDATED: Operational manifest updated.")
    
    return PreferenceConfirmation(
        package_id=submission.package_id,
        selected_option=submission.selected_option,
        status=res.get("status", "RESOLVED"),
        confirmation_code=f"BRG-{submission.package_id[-4:]}-OK",
        new_operational_routing=routing_msg,
        estimated_delivery_update="Updated on Driver Manifest prior to dispatch.",
        message="Preference recorded successfully. First-attempt delivery risk mitigated."
    )


@router.get("/interventions")
def list_interventions():
    """
    Returns active high-risk interventions and real-time customer response statuses.
    """
    return {
        "total_active_interventions": len(interventions_db),
        "interventions": list(interventions_db.values())
    }
