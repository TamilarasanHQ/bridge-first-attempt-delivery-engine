"""
In-Memory State Store for Live Session Manifests and Customer Preferences.
Enables real-time synchronization between the Dispatch Manager and the Customer Portal.
"""
from typing import Dict, Any, Optional

# Active interventions indexed by package_id
interventions_db: Dict[str, Dict[str, Any]] = {}

# Scored manifests store
latest_manifest_assessments: Dict[str, Dict[str, Any]] = {}


def register_assessment(assessment: Dict[str, Any]):
    pkg_id = assessment["package_id"]
    latest_manifest_assessments[pkg_id] = assessment
    if assessment["intervention_required"]:
        if pkg_id not in interventions_db:
            interventions_db[pkg_id] = {
                **assessment,
                "notification_sent": False,
                "preference_resolved": False,
                "selected_preference": None,
                "resolution_notes": None
            }


def trigger_sms(package_id: str) -> Dict[str, Any]:
    if package_id in interventions_db:
        interventions_db[package_id]["notification_sent"] = True
        interventions_db[package_id]["status"] = "NOTIFICATION_SENT"
        return interventions_db[package_id]
    elif package_id in latest_manifest_assessments:
        assessment = latest_manifest_assessments[package_id]
        interventions_db[package_id] = {
            **assessment,
            "notification_sent": True,
            "preference_resolved": False,
            "selected_preference": None,
            "resolution_notes": None,
            "status": "NOTIFICATION_SENT"
        }
        return interventions_db[package_id]
    return {}


def record_customer_preference(package_id: str, preference_data: Dict[str, Any]) -> Dict[str, Any]:
    opt = preference_data.get("selected_option")
    status_map = {
        "ACCESS_POINT": "REROUTED_TO_LOCKER",
        "HOLD_PICKUP": "HELD_AT_DEPOT",
        "RESCHEDULE": "DELIVERY_RESCHEDULED",
        "NEIGHBOR": "NEIGHBOR_HANDOFF_AUTHORIZED",
        "CONFIRM_AVAILABLE": "CONFIRMED_AVAILABLE"
    }
    new_status = status_map.get(opt, "RESOLVED")
    
    if package_id in interventions_db:
        interventions_db[package_id]["preference_resolved"] = True
        interventions_db[package_id]["selected_preference"] = opt
        interventions_db[package_id]["preference_details"] = preference_data
        interventions_db[package_id]["status"] = new_status
        return interventions_db[package_id]
    else:
        interventions_db[package_id] = {
            "package_id": package_id,
            "preference_resolved": True,
            "selected_preference": opt,
            "preference_details": preference_data,
            "status": new_status
        }
        return interventions_db[package_id]
