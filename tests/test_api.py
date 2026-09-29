"""
Integration unit tests for the BRIDGE FastAPI Service.
Verifies health status, batch manifest scoring, single parcel prediction, and customer preferences.
"""
from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 23


def test_sample_manifest_and_batch_scoring():
    # 1. Fetch sample manifest
    r_sample = client.get("/api/v1/sample-manifest")
    assert r_sample.status_code == 200
    manifest = r_sample.json()
    assert len(manifest["packages"]) > 0

    # 2. Score manifest
    r_pred = client.post("/api/v1/predict-manifest", json=manifest)
    assert r_pred.status_code == 200
    result = r_pred.json()
    assert result["total_packages"] == len(manifest["packages"])
    assert result["high_risk_count"] > 0
    assert result["projected_cost_savings_usd"] > 0
    assert len(result["assessments"]) == len(manifest["packages"])


def test_single_parcel_prediction():
    payload = {
        "package_id": "TEST-PARCEL-001",
        "station_code": "DLA3",
        "day_of_week": "Friday",
        "departure_hour": 14,
        "is_weekend": 0,
        "month": 8,
        "latitude": 34.05,
        "longitude": -118.25,
        "depth_cm": 45.0,
        "height_cm": 35.0,
        "width_cm": 30.0,
        "package_volume_cm3": 47250.0,
        "planned_service_time_seconds": 280.0,
        "has_time_window": 1,
        "time_window_duration_hours": 1.5,
        "executor_capacity_cm3": 8500000.0,
        "planned_route_package_count": 160,
        "planned_route_stop_count": 140,
        "planned_route_volume_cm3": 2100000.0,
        "route_volume_utilization": 0.85
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["package_id"] == "TEST-PARCEL-001"
    assert "failure_probability" in data
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(data["top_risk_factors"]) > 0


def test_customer_preference_lifecycle():
    pkg_id = "PKG-LIFECYCLE-TEST"
    
    # 1. Trigger SMS notification
    r_sms = client.post(f"/api/v1/trigger-notification/{pkg_id}")
    assert r_sms.status_code == 200
    assert r_sms.json()["status"] == "NOTIFICATION_SENT"

    # 2. Submit customer preference for Access Point
    pref_payload = {
        "package_id": pkg_id,
        "selected_option": "ACCESS_POINT",
        "access_point_name": "Ralphs Downtown Locker Hub #418"
    }
    r_pref = client.post("/api/v1/preferences", json=pref_payload)
    assert r_pref.status_code == 200
    pref_data = r_pref.json()
    assert pref_data["status"] == "REROUTED_TO_LOCKER"
    assert "REROUTED" in pref_data["new_operational_routing"]


def test_html_ui_routes():
    r_dash = client.get("/")
    assert r_dash.status_code == 200
    assert "BRIDGE v2" in r_dash.text

    r_portal = client.get("/portal")
    assert r_portal.status_code == 200
    assert "Customer Concierge" in r_portal.text
