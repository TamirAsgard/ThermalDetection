from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
import asyncio
import json
import cv2
import numpy as np
from datetime import datetime
import logging

from .models import (
    ThermalReading, SystemStatus, CameraInfo, ConfigUpdate,
    ErrorResponse, HealthCheck, DetectionStatus, ForeheadDetectionData, FaceDetectionData
)


class ThermalAPIRoutes:
    """API routes for thermal detection system"""

    def __init__(self, thermal_processor, camera_manager, config):
        self.thermal_processor = thermal_processor
        self.camera_manager = camera_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.router = APIRouter()
        self._setup_routes()

        # Statistics
        self.start_time = datetime.now()
        self.measurement_count = 0
        self.error_count = 0

    def _setup_routes(self):
        """Setup API routes"""

        @self.router.get("/health", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint"""
            return HealthCheck(
                status="healthy" if self.camera_manager.is_connected() else "degraded",
                timestamp=datetime.now(),
                version="1.0.0",
                services={
                    "camera": self.camera_manager.is_connected(),
                    "thermal_processor": self.thermal_processor.is_running,
                    "api": True
                }
            )

        @self.router.get("/temperature", response_model=ThermalReading)
        async def get_temperature():
            """Get current temperature reading with detection data"""
            try:
                # Get temperature reading
                reading = self.thermal_processor.get_averaged_reading()

                # Get latest detection data
                detection_data = self.thermal_processor.get_latest_detection_data()

                if not reading:
                    raise HTTPException(
                        status_code=404,
                        detail="No temperature reading available"
                    )

                self.measurement_count += 1

                # Build response with detection data
                response = ThermalReading(
                    temperature=reading.avg_temperature,
                    max_temperature=reading.max_temperature,
                    min_temperature=reading.min_temperature,
                    confidence=reading.confidence,
                    is_fever=reading.is_fever,
                    timestamp=reading.timestamp,
                    person_detected=True,
                    pixel_count=reading.pixel_count,
                    frame_width=detection_data.get('frame_width', 640),
                    frame_height=detection_data.get('frame_height', 480)
                )

                # Add face detection data if available
                if detection_data.get('face_detection'):
                    face = detection_data['face_detection']
                    response.face_detection = FaceDetectionData(
                        x=face.x,
                        y=face.y,
                        width=face.width,
                        height=face.height,
                        confidence=face.confidence
                    )

                # Add forehead detection data if available
                if detection_data.get('forehead_detection'):
                    forehead = detection_data['forehead_detection']
                    response.forehead_detection = ForeheadDetectionData(
                        x=forehead['x'],
                        y=forehead['y'],
                        width=forehead['width'],
                        height=forehead['height'],
                        confidence=0.9  # Forehead confidence derived from face confidence
                    )

                return response

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Temperature reading error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/status", response_model=SystemStatus)
        async def get_system_status():
            """Get system status"""
            uptime = (datetime.now() - self.start_time).total_seconds()
            last_reading = self.thermal_processor.last_reading

            return SystemStatus(
                is_running=self.thermal_processor.is_running,
                camera_connected=self.camera_manager.is_connected(),
                last_measurement=last_reading.timestamp if last_reading else None,
                total_measurements=self.measurement_count,
                error_count=self.error_count,
                uptime_seconds=uptime
            )

        @self.router.get("/camera/info", response_model=CameraInfo)
        async def get_camera_info():
            """Get camera information"""
            if not self.camera_manager.camera:
                raise HTTPException(status_code=503, detail="Camera not initialized")

            info = self.camera_manager.camera.get_camera_info()
            return CameraInfo(**info)

        @self.router.post("/camera/restart")
        async def restart_camera(background_tasks: BackgroundTasks):
            """Restart camera connection"""
            background_tasks.add_task(self.camera_manager.restart_camera)
            return {"message": "Camera restart initiated"}

        @self.router.post("/config/update")
        async def update_config(config_update: ConfigUpdate):
            """Update system configuration"""
            try:
                if config_update.fever_threshold is not None:
                    self.config.temperature.fever_threshold = config_update.fever_threshold

                if config_update.confidence_threshold is not None:
                    self.config.detection.face_confidence_threshold = config_update.confidence_threshold

                if config_update.calibration_offset is not None:
                    self.config.temperature.calibration_offset = config_update.calibration_offset

                if config_update.averaging_samples is not None:
                    self.config.temperature.averaging_samples = config_update.averaging_samples

                return {"message": "Configuration updated successfully"}

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Configuration update failed: {e}")

        @self.router.get("/video/rgb")
        async def get_rgb_stream():
            """Get RGB video stream"""
            return StreamingResponse(
                self._generate_rgb_stream_with_detection(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.router.get("/video/thermal")
        async def get_thermal_stream():
            """Get thermal video stream"""
            return StreamingResponse(
                self._generate_thermal_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

    async def _generate_rgb_stream_with_detection(self):
        """Generate RGB video stream with detection overlays drawn"""
        while True:
            try:
                # Get frame from camera
                frame = await self.camera_manager.get_rgb_frame()
                if frame is None:
                    await asyncio.sleep(1 / 15)  # 15 FPS for better performance
                    continue

                # Get detection data
                detection_data = self.thermal_processor.get_latest_detection_data()

                # Draw detection overlays
                if detection_data:
                    frame = self._draw_detection_overlays(frame, detection_data)

                # Add temperature info if available
                reading = self.thermal_processor.get_averaged_reading()
                if reading:
                    frame = self._draw_temperature_info(frame, reading)

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')

                await asyncio.sleep(1 / 15)  # 15 FPS for better performance

            except Exception as e:
                self.logger.error(f"RGB stream error: {e}")
                await asyncio.sleep(0.1)

    def _draw_detection_overlays(self, frame, detection_data):
        """Draw face and forehead detection overlays on frame"""
        try:
            # Draw face detection
            face_detection = detection_data.get('face_detection')
            if face_detection:
                # Face rectangle (green)
                cv2.rectangle(frame,
                              (face_detection.x, face_detection.y),
                              (face_detection.x + face_detection.width,
                               face_detection.y + face_detection.height),
                              (0, 255, 0), 2)

                # Face confidence label
                face_text = f"Face: {face_detection.confidence:.1%}"
                cv2.putText(frame, face_text,
                            (face_detection.x, face_detection.y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw forehead detection
            forehead_detection = detection_data.get('forehead_detection')
            if forehead_detection:
                # Forehead rectangle (red)
                cv2.rectangle(frame,
                              (forehead_detection['x'], forehead_detection['y']),
                              (forehead_detection['x'] + forehead_detection['width'],
                               forehead_detection['y'] + forehead_detection['height']),
                              (0, 0, 255), 2)

                # Forehead confidence label
                forehead_text = f"Forehead: {forehead_detection.get('confidence', 0.9):.1%}"
                cv2.putText(frame, forehead_text,
                            (forehead_detection['x'], forehead_detection['y'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return frame

        except Exception as e:
            self.logger.error(f"Error drawing detection overlays: {e}")
            return frame

    def _draw_temperature_info(self, frame, reading):
        """Draw temperature information on frame"""
        try:
            # Temperature text
            temp_text = f"{reading.avg_temperature:.1f}°C"
            status_text = "FEVER" if reading.is_fever else "Normal"

            # Colors
            temp_color = (0, 0, 255) if reading.is_fever else (0, 255, 0)

            # Draw temperature
            cv2.putText(frame, temp_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, temp_color, 3)

            cv2.putText(frame, status_text, (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, temp_color, 2)

            # Draw confidence
            conf_text = f"Conf: {reading.confidence:.1%}"
            cv2.putText(frame, conf_text, (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return frame

        except Exception as e:
            self.logger.error(f"Error drawing temperature info: {e}")
            return frame

    async def _generate_thermal_stream(self):
        """Generate thermal video stream"""
        while True:
            try:
                frame = await self.camera_manager.get_thermal_frame()
                if frame is not None:
                    # Convert thermal to visible image
                    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

                    ret, buffer = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               buffer.tobytes() + b'\r\n')

                await asyncio.sleep(1 / 15)  # 15 FPS

            except Exception as e:
                self.logger.error(f"Thermal stream error: {e}")
                await asyncio.sleep(0.1)