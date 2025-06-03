import logging
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import cv2
from scipy import ndimage

from src.core.face_detector import FaceDetection


@dataclass
class ForeheadRegion:
    """Forehead region coordinates"""
    x: int
    y: int
    width: int
    height: int


@dataclass
class TemperatureReading:
    """Temperature measurement result"""
    avg_temperature: float
    max_temperature: float
    min_temperature: float
    pixel_count: int
    confidence: float
    timestamp: datetime
    is_fever: bool


class TemperatureAnalyzer:
    """Analyze temperature from thermal data"""

    def __init__(self, config):
        self.config = config
        self.calibration_offset = config.temperature.calibration_offset
        self.fever_threshold = config.temperature.fever_threshold

    def extract_forehead_region(self, face: FaceDetection) -> ForeheadRegion:
        """Extract forehead region from detected face"""
        forehead_height = int(face.height * self.config.detection.forehead_region_ratio)
        forehead_y = face.y + int(face.height * 0.1)  # Start slightly below top

        return ForeheadRegion(
            x=face.x,
            y=forehead_y,
            width=face.width,
            height=forehead_height
        )

    def extract_forehead_region_from_coordinates(self, x: int, y: int, width: int, height: int) -> ForeheadRegion:
        """Extract forehead region from given coordinates"""
        return ForeheadRegion(
            x=x,
            y=y,
            width=width,
            height=height
        )

    def calculate_temperature(self,
                              thermal_frame: np.ndarray,
                              forehead_region: ForeheadRegion) -> Optional[TemperatureReading]:
        """Calculate temperature from thermal data"""
        try:
            # Ensure coordinates are within frame bounds
            h, w = thermal_frame.shape[:2]
            x = max(0, min(forehead_region.x, w - 1))
            y = max(0, min(forehead_region.y, h - 1))
            x2 = max(0, min(x + forehead_region.width, w))
            y2 = max(0, min(y + forehead_region.height, h))

            # Extract forehead region from thermal frame
            roi = thermal_frame[y:y2, x:x2]

            if roi.size == 0:
                return None

            # Apply smoothing filter
            kernel_size = max(1, min(self.config.temperature.smoothing_kernel_size, min(roi.shape) // 2))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size

            roi_smooth = cv2.GaussianBlur(
                roi,
                (kernel_size, kernel_size),
                0
            )

            # Convert thermal values to temperature
            # This conversion depends on your thermal camera's calibration
            # For AT300/AT600, you'll need the specific conversion formula
            temperatures = self._thermal_to_celsius(roi_smooth)

            # Filter out invalid temperatures
            valid_temps = temperatures[
                (temperatures >= self.config.temperature.temp_min) &
                (temperatures <= self.config.temperature.temp_max)
                ]

            if len(valid_temps) == 0:
                return None

            # Calculate statistics
            avg_temp = np.mean(valid_temps) + self.calibration_offset
            max_temp = np.max(valid_temps) + self.calibration_offset
            min_temp = np.min(valid_temps) + self.calibration_offset

            # Calculate confidence based on measurement consistency
            temp_std = np.std(valid_temps)
            confidence = max(0.0, 1.0 - (temp_std / 2.0))

            return TemperatureReading(
                avg_temperature=avg_temp,
                max_temperature=max_temp,
                min_temperature=min_temp,
                pixel_count=len(valid_temps),
                confidence=confidence,
                timestamp=datetime.now(),
                is_fever=avg_temp >= self.fever_threshold
            )

        except Exception as e:
            logging.error(f"Temperature calculation failed: {e}")
            return None

    def _thermal_to_celsius(self, thermal_data: np.ndarray) -> np.ndarray:
        """Convert thermal sensor values to Celsius

        Note: This is a placeholder. You need the actual conversion
        formula from your AT300/AT600 documentation.
        """
        # Placeholder conversion - replace with actual calibration
        # For many thermal cameras: T = (raw_value - offset) * scale
        return (thermal_data.astype(np.float32) - 1000) / 10.0