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

        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = datetime.now()

        # Fever threshold
        self.fever_threshold = config.temperature.fever_threshold

    async def start_processing(self):
        """Start the processing pipeline"""
        self.is_running = True
        self.start_time = datetime.now()
        self.logger.info("Starting thermal processing pipeline")

        while self.is_running:
            try:
                reading = await self.process_frame()
                if reading:
                    self.last_reading = reading
                    self._update_history(reading)
                    self.detection_count += 1
                    self.logger.debug(f"Temperature reading: {reading.avg_temperature:.1f}°C")

                self.frame_count += 1
                await asyncio.sleep(1 / 30)  # 30 FPS

            except Exception as e:
                self.logger.error(f"Processing error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def process_frame(self) -> Optional[TemperatureReading]:
        """Process frame and store detection data"""
        try:
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
            self.logger.debug(f"Processing frame: {frame_width}x{frame_height}")

            # Always initialize detection data with basic info
            self.latest_detection_data = {
                'frame_width': frame_width,
                'frame_height': frame_height,
                'face_detection': None,
                'forehead_detection': None,
                'detection_method': getattr(self.face_detector, 'detection_method', 'yolo11')
            }

            # Detect faces and foreheads using the updated method
            faces, foreheads = self.face_detector.detect_faces_and_foreheads(rgb_frame, draw_detections=False)

            self.logger.debug(f"Face detection results: {len(faces)} faces, {len(foreheads)} foreheads")

            if not faces:
                self.logger.debug("No faces detected")
                return None

            # Get best face and corresponding forehead
            best_face, best_forehead = self.face_detector.get_best_face_and_forehead(faces, foreheads)

            if not best_face:
                self.logger.debug("No best face found")
                return None

            self.logger.debug(
                f"Best face: conf={best_face.confidence:.2f}, pos=({best_face.x},{best_face.y}), size={best_face.width}x{best_face.height}")

            # Calculate forehead coordinates if not provided
            if best_forehead:
                forehead_dict = {
                    'x': best_forehead.x,
                    'y': best_forehead.y,
                    'width': best_forehead.width,
                    'height': best_forehead.height,
                    'confidence': best_forehead.confidence
                }
                self.logger.debug(f"Using detected forehead: {forehead_dict}")
            else:
                # Calculate forehead from face
                forehead_x = best_face.x + int(best_face.width * 0.1)
                forehead_y = best_face.y + int(best_face.height * 0.1)
                forehead_width = int(best_face.width * 0.8)
                forehead_height = int(best_face.height * 0.3)

                forehead_dict = {
                    'x': forehead_x,
                    'y': forehead_y,
                    'width': forehead_width,
                    'height': forehead_height,
                    'confidence': best_face.confidence * 0.9
                }
                self.logger.debug(f"Calculated forehead from face: {forehead_dict}")

            # Update detection data with proper structure
            self.latest_detection_data = {
                'frame_width': frame_width,
                'frame_height': frame_height,
                'face_detection': best_face,  # This is a FaceDetection object
                'forehead_detection': forehead_dict,  # This is a dict
                'detection_method': getattr(self.face_detector, 'detection_method', 'yolo11')
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

            self.logger.debug(
                f"Forehead region for temperature: {forehead_region.x},{forehead_region.y} {forehead_region.width}x{forehead_region.height}")

            # Calculate temperature
            temperature_reading = self.temperature_analyzer.calculate_temperature(
                thermal_frame, forehead_region
            )

            if temperature_reading:
                self.logger.debug(
                    f"Temperature calculated successfully: {temperature_reading.avg_temperature:.1f}°C, confidence={temperature_reading.confidence:.2f}")
            else:
                self.logger.warning("Temperature calculation failed")

            return temperature_reading

        except Exception as e:
            self.logger.error(f"Error in process_frame: {e}", exc_info=True)
            return None

    def _update_history(self, reading: TemperatureReading):
        """Update reading history for averaging"""
        self.reading_history.append(reading)
        max_history = self.config.temperature.averaging_samples
        if len(self.reading_history) > max_history:
            self.reading_history = self.reading_history[-max_history:]

        self.logger.debug(f"Updated history: {len(self.reading_history)} readings")

    def get_averaged_reading(self) -> Optional[TemperatureReading]:
        """Get averaged temperature reading"""
        if not self.reading_history:
            return self.last_reading

        if len(self.reading_history) == 1:
            return self.reading_history[0]

        try:
            # Average the recent readings
            temps = [r.avg_temperature for r in self.reading_history]
            avg_temp = sum(temps) / len(temps)

            # Use the most recent reading as base and update temperature
            latest = self.reading_history[-1]
            averaged_reading = TemperatureReading(
                avg_temperature=avg_temp,
                max_temperature=latest.max_temperature,
                min_temperature=latest.min_temperature,
                pixel_count=latest.pixel_count,
                confidence=latest.confidence,
                timestamp=datetime.now(),
                is_fever=avg_temp >= self.fever_threshold
            )

            self.logger.debug(f"Averaged reading: {avg_temp:.1f}°C from {len(temps)} samples")
            return averaged_reading

        except Exception as e:
            self.logger.error(f"Error calculating averaged reading: {e}")
            return self.last_reading

    def get_latest_detection_data(self):
        """Get the latest detection data"""
        # Ensure we always return a valid dict
        if not self.latest_detection_data:
            return {
                'frame_width': 640,
                'frame_height': 480,
                'face_detection': None,
                'forehead_detection': None,
                'detection_method': 'yolo11'
            }

        return self.latest_detection_data

    def get_processing_statistics(self):
        """Get processing performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'total_frames_processed': self.frame_count,
            'total_detections': self.detection_count,
            'detection_rate': self.detection_count / max(self.frame_count, 1),
            'processing_fps': self.frame_count / max(uptime, 1),
            'detection_method': getattr(self.face_detector, 'detection_method', 'unknown'),
            'avg_detection_time': getattr(self.face_detector, 'avg_detection_time', 0),
            'fever_threshold': self.fever_threshold,
            'readings_in_history': len(self.reading_history)
        }

    def stop_processing(self):
        """Stop the processing pipeline"""
        self.is_running = False
        self.logger.info("Thermal processing stopped")

    def reset_statistics(self):
        """Reset processing statistics"""
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = datetime.now()
        self.reading_history = []
        self.last_reading = None
        self.logger.info("Processing statistics reset")

    def update_configuration(self, updates: dict):
        """Update processing configuration"""
        try:
            if 'fever_threshold' in updates:
                self.fever_threshold = updates['fever_threshold']
                self.config.temperature.fever_threshold = updates['fever_threshold']

            if 'averaging_samples' in updates:
                self.config.temperature.averaging_samples = updates['averaging_samples']

            if 'confidence_threshold' in updates:
                self.config.detection.face_confidence_threshold = updates['confidence_threshold']

            self.logger.info(f"Configuration updated: {updates}")
            return True

        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False

    def get_current_status(self):
        """Get current processing status for debugging"""
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'has_recent_reading': self.last_reading is not None,
            'last_reading_time': self.last_reading.timestamp.isoformat() if self.last_reading else None,
            'detection_data_available': bool(self.latest_detection_data),
            'face_detected': self.latest_detection_data.get(
                'face_detection') is not None if self.latest_detection_data else False,
            'readings_in_history': len(self.reading_history)
        }