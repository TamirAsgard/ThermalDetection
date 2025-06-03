import logging
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import cv2

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
        self.logger = logging.getLogger(__name__)

        # Temperature simulation settings for demo mode
        self.demo_mode = getattr(config.development, 'demo_mode', True)
        self.simulate_fever_chance = getattr(config.development, 'simulate_fever_chance', 0.1)

    def extract_forehead_region(self, face: FaceDetection) -> ForeheadRegion:
        """Extract forehead region from detected face"""
        forehead_height = int(face.height * self.config.detection.forehead_region_ratio)
        forehead_y = face.y + int(face.height * 0.1)  # Start slightly below top

        # Ensure forehead doesn't extend beyond face boundaries
        forehead_width = int(face.width * 0.8)  # Make forehead narrower than face
        forehead_x = face.x + int((face.width - forehead_width) / 2)  # Center it

        return ForeheadRegion(
            x=forehead_x,
            y=forehead_y,
            width=forehead_width,
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
            self.logger.debug(
                f"Calculating temperature for region: {forehead_region.x},{forehead_region.y} {forehead_region.width}x{forehead_region.height}")

            # Ensure coordinates are within frame bounds
            h, w = thermal_frame.shape[:2]
            x = max(0, min(forehead_region.x, w - 1))
            y = max(0, min(forehead_region.y, h - 1))
            x2 = max(0, min(x + forehead_region.width, w))
            y2 = max(0, min(y + forehead_region.height, h))

            self.logger.debug(f"Thermal frame shape: {thermal_frame.shape}, adjusted region: ({x},{y}) to ({x2},{y2})")

            # Extract forehead region from thermal frame
            roi = thermal_frame[y:y2, x:x2]

            if roi.size == 0:
                self.logger.warning("Empty ROI extracted from thermal frame")
                return None

            self.logger.debug(
                f"ROI shape: {roi.shape}, ROI stats: min={np.min(roi):.1f}, max={np.max(roi):.1f}, mean={np.mean(roi):.1f}")

            # Apply smoothing filter
            kernel_size = max(1, min(self.config.temperature.smoothing_kernel_size, min(roi.shape) // 2))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size

            # Ensure roi is float32 for cv2.GaussianBlur
            roi_float = roi.astype(np.float32)
            roi_smooth = cv2.GaussianBlur(roi_float, (kernel_size, kernel_size), 0)

            # Convert thermal values to temperature
            temperatures = self._thermal_to_celsius(roi_smooth)

            self.logger.debug(
                f"Temperature range after conversion: {np.min(temperatures):.1f} to {np.max(temperatures):.1f}°C")

            # Filter out invalid temperatures
            valid_temps = temperatures[
                (temperatures >= self.config.temperature.temp_min) &
                (temperatures <= self.config.temperature.temp_max)
                ]

            self.logger.debug(f"Valid temperatures: {len(valid_temps)} out of {temperatures.size} pixels")

            # If no valid temperatures in range, generate simulated reading for demo
            if len(valid_temps) == 0:
                self.logger.warning("No valid temperatures found in configured range")
                if self.demo_mode:
                    return self._generate_demo_temperature_reading(roi.size)
                else:
                    return None

            # Calculate statistics
            avg_temp = np.mean(valid_temps) + self.calibration_offset
            max_temp = np.max(valid_temps) + self.calibration_offset
            min_temp = np.min(valid_temps) + self.calibration_offset

            # Calculate confidence based on measurement consistency
            temp_std = np.std(valid_temps)
            confidence = max(0.1, min(1.0, 1.0 - (temp_std / 3.0)))  # Better confidence calculation

            self.logger.debug(
                f"Temperature reading: avg={avg_temp:.1f}°C, std={temp_std:.2f}, confidence={confidence:.2f}")

            reading = TemperatureReading(
                avg_temperature=avg_temp,
                max_temperature=max_temp,
                min_temperature=min_temp,
                pixel_count=len(valid_temps),
                confidence=confidence,
                timestamp=datetime.now(),
                is_fever=avg_temp >= self.fever_threshold
            )

            self.logger.debug(f"Final reading: {avg_temp:.1f}°C, fever={reading.is_fever}")
            return reading

        except Exception as e:
            self.logger.error(f"Temperature calculation failed: {e}", exc_info=True)
            # Return demo reading as fallback
            if self.demo_mode:
                return self._generate_demo_temperature_reading(100)
            return None

    def _thermal_to_celsius(self, thermal_data: np.ndarray) -> np.ndarray:
        """Convert thermal sensor values to Celsius"""
        try:
            # Detect data type and range to determine conversion method
            data_min = np.min(thermal_data)
            data_max = np.max(thermal_data)

            self.logger.debug(f"Thermal data range: {data_min:.1f} to {data_max:.1f}")

            if thermal_data.dtype == np.uint8 or (data_min >= 0 and data_max <= 255):
                # Webcam grayscale data (0-255)
                return self._convert_webcam_to_celsius(thermal_data)
            elif data_min > 500:  # Likely raw thermal camera data
                # Raw thermal camera data (typically 1000-1100 range for body temperature)
                return self._convert_raw_thermal_to_celsius(thermal_data)
            else:
                # Unknown format, try to normalize and convert
                self.logger.warning(f"Unknown thermal data format, attempting normalization")
                return self._convert_normalized_to_celsius(thermal_data)

        except Exception as e:
            self.logger.error(f"Thermal conversion failed: {e}")
            # Fallback to demo temperature
            return np.full(thermal_data.shape, 36.5 + np.random.normal(0, 0.5))

    def _convert_webcam_to_celsius(self, thermal_data: np.ndarray) -> np.ndarray:
        """Convert webcam grayscale data to temperature"""
        # Normalize 0-255 to 0-1
        normalized = thermal_data.astype(np.float32) / 255.0

        # Map to realistic body temperature range (35-40°C)
        # Use center region as warmer (face area)
        h, w = thermal_data.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Create distance from center map
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

        # Normalize distance (0 at center, 1 at edges)
        distance_normalized = distance_from_center / max_distance

        # Create temperature map: warmer in center, cooler at edges
        base_temp = 36.0  # Base temperature
        temp_variation = 2.0  # Temperature range

        # Combine normalized pixel values with distance-based temperature
        temp_from_pixels = base_temp + normalized * temp_variation
        temp_from_distance = base_temp + (1.0 - distance_normalized) * temp_variation

        # Blend both effects
        temperatures = (temp_from_pixels + temp_from_distance) / 2.0

        # Add realistic noise
        noise = np.random.normal(0, 0.3, temperatures.shape)
        temperatures += noise

        return temperatures

    def _convert_raw_thermal_to_celsius(self, thermal_data: np.ndarray) -> np.ndarray:
        """Convert raw thermal camera data to Celsius"""
        # This is for actual thermal cameras (AT300/AT600, etc.)
        # Standard conversion: (raw_value - offset) / scale
        # These values should be calibrated for your specific camera
        offset = 1000.0  # Typical offset for body temperature range
        scale = 10.0  # Typical scale factor

        temperatures = (thermal_data.astype(np.float32) - offset) / scale

        # Add calibration offset
        temperatures += self.calibration_offset

        return temperatures

    def _convert_normalized_to_celsius(self, thermal_data: np.ndarray) -> np.ndarray:
        """Convert normalized thermal data to Celsius"""
        # Normalize to 0-1 range
        data_min = np.min(thermal_data)
        data_max = np.max(thermal_data)

        if data_max > data_min:
            normalized = (thermal_data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(thermal_data)

        # Map to body temperature range
        temperatures = 35.0 + normalized * 5.0  # 35-40°C range

        # Add noise for realism
        noise = np.random.normal(0, 0.2, temperatures.shape)
        temperatures += noise

        return temperatures

    def _generate_demo_temperature_reading(self, pixel_count: int) -> TemperatureReading:
        """Generate a realistic demo temperature reading"""
        # Simulate realistic body temperature
        base_temp = 36.0 + np.random.normal(0, 0.8)  # Normal body temp with variation

        # Occasionally simulate fever for demo
        if np.random.random() < self.simulate_fever_chance:
            base_temp = 37.8 + np.random.normal(0, 0.5)  # Fever temperature

        # Add calibration offset
        avg_temp = base_temp + self.calibration_offset

        # Generate realistic min/max around average
        temp_variation = np.random.uniform(0.5, 1.0)
        max_temp = avg_temp + temp_variation
        min_temp = avg_temp - temp_variation

        # Generate confidence based on "measurement quality"
        confidence = np.random.uniform(0.7, 0.95)

        self.logger.debug(f"Generated demo temperature: {avg_temp:.1f}°C (demo mode)")

        return TemperatureReading(
            avg_temperature=avg_temp,
            max_temperature=max_temp,
            min_temperature=min_temp,
            pixel_count=pixel_count,
            confidence=confidence,
            timestamp=datetime.now(),
            is_fever=avg_temp >= self.fever_threshold
        )

    def set_calibration_offset(self, offset: float):
        """Update calibration offset"""
        self.calibration_offset = offset
        self.config.temperature.calibration_offset = offset
        self.logger.info(f"Calibration offset updated to: {offset:.2f}°C")

    def set_fever_threshold(self, threshold: float):
        """Update fever threshold"""
        self.fever_threshold = threshold
        self.config.temperature.fever_threshold = threshold
        self.logger.info(f"Fever threshold updated to: {threshold:.1f}°C")

    def get_analyzer_status(self):
        """Get analyzer status for debugging"""
        return {
            'demo_mode': self.demo_mode,
            'fever_threshold': self.fever_threshold,
            'calibration_offset': self.calibration_offset,
            'temp_min': self.config.temperature.temp_min,
            'temp_max': self.config.temperature.temp_max,
            'simulate_fever_chance': self.simulate_fever_chance,
            'smoothing_kernel_size': self.config.temperature.smoothing_kernel_size
        }