"""
Prediction endpoints for single parcel and batch manifest evaluation.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas.delivery import PackageInput, BatchManifestInput, RiskAssessment, BatchPredictionResponse
from app.core.engine import engine
from app.core.policy import policy_engine
from app.core.state import register_assessment, interventions_db

router = APIRouter()


@router.post("/predict", response_model=RiskAssessment)
def predict_single_package(package: PackageInput):
    """
    Evaluates failure risk for an individual parcel prior to dispatch.
    """
    record = package.model_dump()
    prob = float(engine.predict_proba([record])[0])
    policy = policy_engine.evaluate(prob, record)
    explanations = engine.explain_risk(record, prob)
    
    assessment = {
        "package_id": package.package_id,
        "failure_probability": round(prob, 4),
        "failure_percentage": f"{prob * 100:.2f}%",
        "risk_tier": policy["risk_tier"],
        "priority": policy["priority"],
        "intervention_required": policy["intervention_required"],
        "operational_action": policy["operational_action"],
        "top_risk_factors": explanations,
        "allowed_options": policy["allowed_options"],
        "policy_summary": policy["policy_summary"],
        "recipient_name": package.recipient_name,
        "recipient_phone": package.recipient_phone,
        "address_line": package.address_line,
        "status": "PENDING_DISPATCH"
    }
    
    register_assessment(assessment)
    return assessment


@router.post("/predict-manifest", response_model=BatchPredictionResponse)
def predict_batch_manifest(manifest: BatchManifestInput):
    """
    Scores an entire morning vehicle manifest batch before dispatch.
    """
    if not manifest.packages:
        raise HTTPException(status_code=400, detail="Manifest contains no packages.")

    records = [pkg.model_dump() for pkg in manifest.packages]
    probs = engine.predict_proba(records)
    
    assessments: List[RiskAssessment] = []
    high_count = 0
    med_count = 0
    low_count = 0
    
    for pkg, prob_val in zip(manifest.packages, probs):
        prob = float(prob_val)
        record = pkg.model_dump()
        policy = policy_engine.evaluate(prob, record)
        explanations = engine.explain_risk(record, prob)
        
        if policy["risk_tier"] == "HIGH":
            high_count += 1
        elif policy["risk_tier"] == "MEDIUM":
            med_count += 1
        else:
            low_count += 1

        assessment_dict = {
            "package_id": pkg.package_id,
            "failure_probability": round(prob, 4),
            "failure_percentage": f"{prob * 100:.2f}%",
            "risk_tier": policy["risk_tier"],
            "priority": policy["priority"],
            "intervention_required": policy["intervention_required"],
            "operational_action": policy["operational_action"],
            "top_risk_factors": explanations,
            "allowed_options": policy["allowed_options"],
            "policy_summary": policy["policy_summary"],
            "recipient_name": pkg.recipient_name,
            "recipient_phone": pkg.recipient_phone,
            "address_line": pkg.address_line,
            "status": "PENDING_DISPATCH"
        }
        register_assessment(assessment_dict)
        assessments.append(RiskAssessment(**assessment_dict))

    # Business impact estimations:
    # Industry average: Failed delivery costs ~$15.50 (re-attempt fuel, labor, depot re-processing)
    # Intervening on top high-risk packages prevents ~65% of those flagged failures.
    preventable_failures = int(round(high_count * 0.65))
    cost_savings = preventable_failures * 15.50

    return BatchPredictionResponse(
        manifest_id=manifest.manifest_id,
        total_packages=len(manifest.packages),
        high_risk_count=high_count,
        medium_risk_count=med_count,
        low_risk_count=low_count,
        high_risk_percentage=f"{(high_count / len(manifest.packages)) * 100:.1f}%",
        estimated_failures_preventable=preventable_failures,
        projected_cost_savings_usd=cost_savings,
        assessments=assessments
    )


@router.get("/sample-manifest")
def get_sample_manifest():
    """
    Returns realistic Amazon Last Mile manifest records across urban, suburban, and high-dwell routes.
    """
    samples = [
        {
            "package_id": "PKG-DLA3-9941",
            "station_code": "DLA3",
            "day_of_week": "Friday",
            "departure_hour": 13,
            "is_weekend": 0,
            "month": 8,
            "latitude": 34.0407,
            "longitude": -118.2468,
            "depth_cm": 45.0,
            "height_cm": 35.0,
            "width_cm": 30.0,
            "package_volume_cm3": 47250.0,
            "planned_service_time_seconds": 290.0,
            "has_time_window": 1,
            "time_window_duration_hours": 1.5,
            "executor_capacity_cm3": 8500000.0,
            "planned_route_package_count": 165,
            "planned_route_stop_count": 140,
            "planned_route_volume_cm3": 2100000.0,
            "route_volume_utilization": 0.82,
            "recipient_name": "Jordan Hayes",
            "recipient_phone": "+1 (213) 555-0142",
            "address_line": "888 S Hope St, High-Rise Unit 1804, Los Angeles, CA"
        },
        {
            "package_id": "PKG-DLA8-5120",
            "station_code": "DLA8",
            "day_of_week": "Friday",
            "departure_hour": 14,
            "is_weekend": 0,
            "month": 8,
            "latitude": 34.0210,
            "longitude": -118.2890,
            "depth_cm": 50.0,
            "height_cm": 40.0,
            "width_cm": 35.0,
            "package_volume_cm3": 70000.0,
            "planned_service_time_seconds": 260.0,
            "has_time_window": 1,
            "time_window_duration_hours": 2.0,
            "executor_capacity_cm3": 9000000.0,
            "planned_route_package_count": 180,
            "planned_route_stop_count": 155,
            "planned_route_volume_cm3": 2400000.0,
            "route_volume_utilization": 0.85,
            "recipient_name": "Marcus Vance",
            "recipient_phone": "+1 (310) 555-0199",
            "address_line": "1240 W 29th St, Gated Complex Apt 12, Los Angeles, CA"
        },
        {
            "package_id": "PKG-DBO1-2319",
            "station_code": "DBO1",
            "day_of_week": "Wednesday",
            "departure_hour": 9,
            "is_weekend": 0,
            "month": 8,
            "latitude": 42.3601,
            "longitude": -71.0589,
            "depth_cm": 20.0,
            "height_cm": 15.0,
            "width_cm": 15.0,
            "package_volume_cm3": 4500.0,
            "planned_service_time_seconds": 45.0,
            "has_time_window": 0,
            "time_window_duration_hours": 0.0,
            "executor_capacity_cm3": 12000000.0,
            "planned_route_package_count": 120,
            "planned_route_stop_count": 105,
            "planned_route_volume_cm3": 950000.0,
            "route_volume_utilization": 0.25,
            "recipient_name": "Sarah Lin",
            "recipient_phone": "+1 (617) 555-0184",
            "address_line": "15 Commonwealth Ave, Single Family Home, Boston, MA"
        },
        {
            "package_id": "PKG-DSE4-8812",
            "station_code": "DSE4",
            "day_of_week": "Thursday",
            "departure_hour": 10,
            "is_weekend": 0,
            "month": 8,
            "latitude": 47.6062,
            "longitude": -122.3321,
            "depth_cm": 25.0,
            "height_cm": 18.0,
            "width_cm": 20.0,
            "package_volume_cm3": 9000.0,
            "planned_service_time_seconds": 65.0,
            "has_time_window": 0,
            "time_window_duration_hours": 0.0,
            "executor_capacity_cm3": 10000000.0,
            "planned_route_package_count": 140,
            "planned_route_stop_count": 115,
            "planned_route_volume_cm3": 1200000.0,
            "route_volume_utilization": 0.35,
            "recipient_name": "David Miller",
            "recipient_phone": "+1 (206) 555-0131",
            "address_line": "412 Pine St, Residential Entryway, Seattle, WA"
        },
        {
            "package_id": "PKG-DCH2-7744",
            "station_code": "DCH2",
            "day_of_week": "Monday",
            "departure_hour": 11,
            "is_weekend": 0,
            "month": 8,
            "latitude": 41.8781,
            "longitude": -87.6298,
            "depth_cm": 35.0,
            "height_cm": 25.0,
            "width_cm": 25.0,
            "package_volume_cm3": 21875.0,
            "planned_service_time_seconds": 150.0,
            "has_time_window": 1,
            "time_window_duration_hours": 3.0,
            "executor_capacity_cm3": 9500000.0,
            "planned_route_package_count": 150,
            "planned_route_stop_count": 130,
            "planned_route_volume_cm3": 1600000.0,
            "route_volume_utilization": 0.55,
            "recipient_name": "Elena Rostova",
            "recipient_phone": "+1 (312) 555-0177",
            "address_line": "500 N Michigan Ave, Suite 300, Chicago, IL"
        },
        {
            "package_id": "PKG-DLA3-6621",
            "station_code": "DLA3",
            "day_of_week": "Friday",
            "departure_hour": 15,
            "is_weekend": 0,
            "month": 8,
            "latitude": 34.0550,
            "longitude": -118.2500,
            "depth_cm": 38.0,
            "height_cm": 30.0,
            "width_cm": 28.0,
            "package_volume_cm3": 31920.0,
            "planned_service_time_seconds": 210.0,
            "has_time_window": 1,
            "time_window_duration_hours": 2.0,
            "executor_capacity_cm3": 8500000.0,
            "planned_route_package_count": 170,
            "planned_route_stop_count": 145,
            "planned_route_volume_cm3": 2200000.0,
            "route_volume_utilization": 0.88,
            "recipient_name": "Carlos Gomez",
            "recipient_phone": "+1 (213) 555-0188",
            "address_line": "1111 S Figueroa St, Loft 5A, Los Angeles, CA"
        }
    ]
    return {"manifest_id": "AMAZON-MORNING-ROUTE-0821", "packages": samples}
