"""
Pydantic schemas for parcel manifest inputs and risk prediction outputs.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PackageInput(BaseModel):
    package_id: str = Field(..., description="Unique package identifier (e.g. Package_0a1b2c)")
    station_code: str = Field(default="DLA3", description="Origin fulfillment hub (e.g. DLA3, DBO1, DSE4)")
    day_of_week: str = Field(default="Wednesday", description="Day of dispatch (Monday-Sunday)")
    departure_hour: int = Field(default=9, ge=0, le=23, description="Dispatch departure hour (0-23)")
    is_weekend: int = Field(default=0, ge=0, le=1, description="1 if Saturday/Sunday, else 0")
    month: int = Field(default=8, ge=1, le=12, description="Month of dispatch (1-12)")
    latitude: float = Field(default=34.0522, description="Destination latitude")
    longitude: float = Field(default=-118.2437, description="Destination longitude")
    depth_cm: float = Field(default=25.0, gt=0, description="Package depth in centimeters")
    height_cm: float = Field(default=15.0, gt=0, description="Package height in centimeters")
    width_cm: float = Field(default=20.0, gt=0, description="Package width in centimeters")
    package_volume_cm3: float = Field(default=7500.0, gt=0, description="Volume in cubic centimeters")
    planned_service_time_seconds: float = Field(default=180.0, ge=0, description="Estimated driver dwell time at address")
    has_time_window: int = Field(default=1, ge=0, le=1, description="1 if delivery has restricted time window, else 0")
    time_window_duration_hours: float = Field(default=2.0, ge=0, description="Duration of delivery window in hours")
    executor_capacity_cm3: float = Field(default=10000000.0, gt=0, description="Vehicle cargo volume capacity in cm3")
    planned_route_package_count: int = Field(default=150, gt=0, description="Total packages on route")
    planned_route_stop_count: int = Field(default=120, gt=0, description="Total delivery stops on route")
    planned_route_volume_cm3: float = Field(default=1500000.0, gt=0, description="Total route package volume in cm3")
    route_volume_utilization: float = Field(default=0.15, ge=0, le=1.5, description="Ratio of package volume to van capacity")
    
    # Customer and Contextual Metadata (Non-ML metadata for dispatch management)
    recipient_name: Optional[str] = Field(default="Alex Mercer", description="Recipient name")
    recipient_phone: Optional[str] = Field(default="+1 (555) 019-2834", description="Recipient contact phone")
    address_line: Optional[str] = Field(default="742 Evergreen Terrace, Apt 4B", description="Street address")


class BatchManifestInput(BaseModel):
    manifest_id: str = Field(default="MANIFEST-2026-09", description="Manifest batch identifier")
    packages: List[PackageInput] = Field(..., description="List of parcels in the route manifest")


class RiskAssessment(BaseModel):
    package_id: str
    failure_probability: float
    failure_percentage: str
    risk_tier: str
    priority: str
    intervention_required: bool
    operational_action: str
    top_risk_factors: List[str]
    allowed_options: List[Dict[str, Any]]
    policy_summary: str
    recipient_name: Optional[str] = None
    recipient_phone: Optional[str] = None
    address_line: Optional[str] = None
    status: str = Field(default="PENDING_DISPATCH", description="Current status in workflow")


class BatchPredictionResponse(BaseModel):
    manifest_id: str
    total_packages: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    high_risk_percentage: str
    estimated_failures_preventable: int
    projected_cost_savings_usd: float
    assessments: List[RiskAssessment]
