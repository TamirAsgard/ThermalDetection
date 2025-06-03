from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TemperatureUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

class DetectionStatus(str, Enum):
    NO_PERSON = "no_person"
    PERSON_DETECTED = "person_detected"
    MEASURING = "measuring"
    MEASUREMENT_COMPLETE = "measurement_complete"
    ERROR = "error"

class FaceDetectionData(BaseModel):
    """Face detection information"""
    x: int = Field(..., description="Face rectangle X coordinate")
    y: int = Field(..., description="Face rectangle Y coordinate")
    width: int = Field(..., description="Face rectangle width")
    height: int = Field(..., description="Face rectangle height")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Face detection confidence")

class ForeheadDetectionData(BaseModel):
    """Forehead region information"""
    x: int = Field(..., description="Forehead rectangle X coordinate")
    y: int = Field(..., description="Forehead rectangle Y coordinate")
    width: int = Field(..., description="Forehead rectangle width")
    height: int = Field(..., description="Forehead rectangle height")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Forehead detection confidence")

class ThermalReading(BaseModel):
    """Enhanced temperature reading with detection data"""
    temperature: float = Field(..., description="Average forehead temperature")
    max_temperature: float = Field(..., description="Maximum temperature in region")
    min_temperature: float = Field(..., description="Minimum temperature in region")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Measurement confidence")
    is_fever: bool = Field(..., description="Whether temperature indicates fever")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    person_detected: bool = Field(..., description="Whether a person was detected")
    pixel_count: int = Field(..., description="Number of pixels used in measurement")

    # Detection data
    face_detection: Optional[FaceDetectionData] = Field(None, description="Face detection data")
    forehead_detection: Optional[ForeheadDetectionData] = Field(None, description="Forehead detection data")
    frame_width: int = Field(640, description="Video frame width for coordinate reference")
    frame_height: int = Field(480, description="Video frame height for coordinate reference")

class SystemStatus(BaseModel):
    """System status response model"""
    is_running: bool
    camera_connected: bool
    last_measurement: Optional[datetime]
    total_measurements: int
    error_count: int
    uptime_seconds: float

class CameraInfo(BaseModel):
    """Camera information model"""
    thermal_device_type: str
    frame_width: int
    frame_height: int
    fps: int
    rgb_connected: bool
    thermal_connected: bool

class ConfigUpdate(BaseModel):
    """Configuration update request model"""
    fever_threshold: Optional[float] = Field(None, ge=30.0, le=45.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    calibration_offset: Optional[float] = Field(None, ge=-5.0, le=5.0)
    averaging_samples: Optional[int] = Field(None, ge=1, le=50)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, bool]