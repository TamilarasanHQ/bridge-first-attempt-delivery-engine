"""
Pydantic schemas for customer delivery preference capture and updates.
"""
from pydantic import BaseModel, Field
from typing import Optional


class CustomerPreferenceSubmission(BaseModel):
    package_id: str = Field(..., description="Target package identifier")
    selected_option: str = Field(
        ...,
        description="Option: ACCESS_POINT, HOLD_PICKUP, RESCHEDULE, NEIGHBOR, CONFIRM_AVAILABLE"
    )
    access_point_name: Optional[str] = Field(default=None, description="Locker name or store location")
    reschedule_date: Optional[str] = Field(default=None, description="Preferred future delivery date (YYYY-MM-DD)")
    neighbor_notes: Optional[str] = Field(default=None, description="Neighbor unit number or authorized name")
    gate_code: Optional[str] = Field(default=None, description="Gate code or access instructions")
    customer_notes: Optional[str] = Field(default=None, description="Special instructions")


class PreferenceConfirmation(BaseModel):
    package_id: str
    selected_option: str
    status: str
    confirmation_code: str
    new_operational_routing: str
    estimated_delivery_update: str
    message: str
