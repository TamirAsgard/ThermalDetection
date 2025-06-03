import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import logging


class ThermalCalibration:
    """Thermal camera calibration utilities"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.calibration_data = {}
        self.load_calibration()

    def load_calibration(self):
        """Load calibration data from file"""
        calib_file = Path("config/thermal_calibration.json")
        if calib_file.exists():
            try:
                with open(calib_file, 'r') as f:
                    self.calibration_data = json.load(f)
                self.logger.info("Calibration data loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load calibration: {e}")
        else:
            self.logger.warning("No calibration file found, using defaults")

    def save_calibration(self):
        """Save calibration data to file"""
        calib_file = Path("config/thermal_calibration.json")
        calib_file.parent.mkdir(exist_ok=True)

        try:
            with open(calib_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            self.logger.info("Calibration data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")

    def calibrate_blackbody(self, thermal_frame: np.ndarray,
                            reference_temp: float, roi: Tuple[int, int, int, int]):
        """Calibrate using blackbody reference"""
        x, y, w, h = roi
        roi_data = thermal_frame[y:y + h, x:x + w]

        # Calculate average thermal value in ROI
        avg_thermal_value = np.mean(roi_data)

        # Store calibration point
        if 'blackbody_points' not in self.calibration_data:
            self.calibration_data['blackbody_points'] = []

        self.calibration_data['blackbody_points'].append({
            'thermal_value': float(avg_thermal_value),
            'reference_temp': reference_temp
        })

        # Calculate linear calibration if we have multiple points
        if len(self.calibration_data['blackbody_points']) >= 2:
            self._calculate_linear_calibration()

        self.save_calibration()

    def _calculate_linear_calibration(self):
        """Calculate linear calibration coefficients"""
        points = self.calibration_data['blackbody_points']

        thermal_values = [p['thermal_value'] for p in points]
        reference_temps = [p['reference_temp'] for p in points]

        # Linear regression: temp = slope * thermal_value + intercept
        coeffs = np.polyfit(thermal_values, reference_temps, 1)

        self.calibration_data['calibration_coeffs'] = {
            'slope': float(coeffs[0]),
            'intercept': float(coeffs[1])
        }

        self.logger.info(f"Calibration updated: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}")

    def thermal_to_temperature(self, thermal_value: float) -> float:
        """Convert thermal value to temperature using calibration"""
        if 'calibration_coeffs' in self.calibration_data:
            coeffs = self.calibration_data['calibration_coeffs']
            temp = coeffs['slope'] * thermal_value + coeffs['intercept']
        else:
            # Default conversion (placeholder)
            temp = (thermal_value - 1000) / 10.0

        return temp + self.config.temperature.calibration_offset

    def get_calibration_status(self) -> Dict:
        """Get calibration status information"""
        return {
            'is_calibrated': 'calibration_coeffs' in self.calibration_data,
            'calibration_points': len(self.calibration_data.get('blackbody_points', [])),
            'calibration_offset': self.config.temperature.calibration_offset,
            'coefficients': self.calibration_data.get('calibration_coeffs', {})
        }