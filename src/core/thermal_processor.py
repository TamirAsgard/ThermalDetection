import asyncio
from typing import Optional
import logging
from datetime import datetime

from .face_detector import FaceDetector
from .temperature_analyzer import TemperatureAnalyzer, TemperatureReading


class ThermalProcessor:
    """Main processing pipeline for thermal detection"""

    def __init__(self, config, camera_manager):
        self.config = config
        self.camera_manager = camera_manager
        self.face_detector = FaceDetector(config)
        self.temperature_analyzer = TemperatureAnalyzer(config)
        self.logger = logging.getLogger(__name__)

        # Processing state
        self.is_running = False
        self.last_reading = None
        self.reading_history = []
        self.latest_detection_data = {}

        # Fever threshold
        self.fever_threshold = config.temperature.fever_threshold

    async def start_processing(self):
        """Start the processing pipeline"""
        self.is_running = True
        while self.is_running:
            try:
                reading = await self.process_frame()
                if reading:
                    self.last_reading = reading
                    self._update_history(reading)

                await asyncio.sleep(1 / 30)  # 30 FPS

            except Exception as e:
                # self.logger.error(f"Processing error: {e}")
                await asyncio.sleep(0.1)

    async def process_frame(self) -> Optional[TemperatureReading]:
        """Process frame and store detection data"""
        # Get RGB frame for face detection
        rgb_frame = await self.camera_manager.get_rgb_frame()
        if rgb_frame is None:
            self.latest_detection_data = {}
            return None

        # Get thermal frame
        thermal_frame = await self.camera_manager.get_thermal_frame()
        if thermal_frame is None:
            self.latest_detection_data = {}
            return None

        # Store frame dimensions
        frame_height, frame_width = rgb_frame.shape[:2]

        # Detect faces and foreheads using the updated method
        faces, foreheads = self.face_detector.detect_faces_and_foreheads(rgb_frame, draw_detections=False)

        if not faces:
            self.latest_detection_data = {
                'frame_width': frame_width,
                'frame_height': frame_height,
                'face_detection': None,
                'forehead_detection': None
            }
            return None

        # Get best face and corresponding forehead
        best_face, best_forehead = self.face_detector.get_best_face_and_forehead(faces, foreheads)

        if not best_face:
            self.latest_detection_data = {
                'frame_width': frame_width,
                'frame_height': frame_height,
                'face_detection': None,
                'forehead_detection': None
            }
            return None

        # Store detection data with proper structure
        self.latest_detection_data = {
            'frame_width': frame_width,
            'frame_height': frame_height,
            'face_detection': best_face,  # This is a FaceDetection object
            'forehead_detection': {
                'x': best_forehead.x if best_forehead else best_face.x,
                'y': best_forehead.y if best_forehead else best_face.y + int(best_face.height * 0.1),
                'width': best_forehead.width if best_forehead else int(best_face.width * 0.8),
                'height': best_forehead.height if best_forehead else int(best_face.height * 0.3),
                'confidence': best_forehead.confidence if best_forehead else best_face.confidence * 0.9
            }
        }

        # Extract forehead region for temperature calculation
        if best_forehead:
            # Use the forehead detection directly
            forehead_region = self.temperature_analyzer.extract_forehead_region_from_coordinates(
                best_forehead.x, best_forehead.y, best_forehead.width, best_forehead.height
            )
        else:
            # Fallback to extracting from face
            forehead_region = self.temperature_analyzer.extract_forehead_region(best_face)

        # Calculate temperature
        temperature_reading = self.temperature_analyzer.calculate_temperature(
            thermal_frame, forehead_region
        )

        return temperature_reading

    def _update_history(self, reading: TemperatureReading):
        """Update reading history for averaging"""
        self.reading_history.append(reading)
        max_history = self.config.temperature.averaging_samples
        if len(self.reading_history) > max_history:
            self.reading_history = self.reading_history[-max_history:]

    def get_averaged_reading(self) -> Optional[TemperatureReading]:
        """Get averaged temperature reading"""
        if not self.reading_history:
            return self.last_reading

        if len(self.reading_history) == 1:
            return self.reading_history[0]

        # Average the recent readings
        temps = [r.avg_temperature for r in self.reading_history]
        avg_temp = sum(temps) / len(temps)

        # Use the most recent reading as base and update temperature
        latest = self.reading_history[-1]
        return TemperatureReading(
            avg_temperature=avg_temp,
            max_temperature=latest.max_temperature,
            min_temperature=latest.min_temperature,
            pixel_count=latest.pixel_count,
            confidence=latest.confidence,
            timestamp=datetime.now(),
            is_fever=avg_temp >= self.fever_threshold
        )

    def get_latest_detection_data(self):
        """Get the latest detection data"""
        return self.latest_detection_data

    def stop_processing(self):
        """Stop the processing pipeline"""
        self.is_running = False