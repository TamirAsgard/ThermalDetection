import time
from datetime import datetime

import cv2
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from .base_camera import BaseCameraInterface
from ..core.face_detector import FaceDetector, ForeheadDetection, FaceDetection


class WebcamCamera(BaseCameraInterface):
    """Dedicated webcam camera implementation for demo purposes"""

    def __init__(self, config):
        self.config = config
        self.cap = None
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Set macOS compatibility
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

        # Initialize face detector
        self._init_face_detector()

        # Camera settings
        self.camera_index = getattr(config.camera, 'rgb_camera_index', 0)
        self.frame_width = getattr(config.camera, 'frame_width', 640)
        self.frame_height = getattr(config.camera, 'frame_height', 480)
        self.fps = getattr(config.camera, 'fps', 30)

        # Detection data
        self.latest_faces = []
        self.latest_foreheads = []
        self.latest_frame = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.avg_detection_time = 0.0
        self.detection_times = []

    def _init_face_detector(self):
        """Initialize face detector"""
        try:
            # Import the updated face detector
            from ..core.face_detector import FaceDetector
            self.face_detector = FaceDetector(self.config)
            self.logger.info("Face detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            self.face_detector = None

    async def initialize(self) -> bool:
        """Initialize webcam"""
        try:
            self.logger.info(f"Initializing webcam (camera: {self.camera_index})")

            loop = asyncio.get_event_loop()
            self.cap = await loop.run_in_executor(
                self.executor,
                self._create_webcam_capture
            )

            if self.cap and self.cap.isOpened():
                # Test frame capture
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.is_initialized = True
                    self.start_time = datetime.now()
                    self.logger.info("Webcam initialized successfully")

                    # Log detection method if available
                    if self.face_detector:
                        detection_info = self.face_detector.get_detection_info()
                        self.logger.info(f"Using detection method: {detection_info['method']}")

                    return True
                else:
                    self.logger.error("Webcam opened but cannot capture frames")
                    return False
            else:
                self.logger.error("Failed to open webcam")
                return False

        except Exception as e:
            self.logger.error(f"Webcam initialization error: {e}")
            return False

    def _create_webcam_capture(self):
        """Create webcam capture object with macOS compatibility"""
        # Try different backends for better compatibility
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.camera_index, backend)

                if cap.isOpened():
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

                    self.logger.info(f"Webcam opened with backend: {backend}")
                    return cap
                else:
                    cap.release()

            except Exception as e:
                self.logger.warning(f"Backend {backend} failed: {e}")
                continue

        self.logger.error("All camera backends failed")
        return None

    async def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame from webcam and run detection"""
        if not self.is_initialized or not self.cap:
            return None

        try:
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(
                self.executor,
                self.cap.read
            )

            if ret and frame is not None:
                self.frame_count += 1

                # Run face detection in background to update detection data
                if self.face_detector:
                    try:
                        start_time = time.time()

                        # Perform detection
                        faces, foreheads = self.face_detector.detect_faces_and_foreheads(
                            frame, draw_detections=False  # Don't draw on the frame
                        )

                        detection_time = time.time() - start_time
                        self._update_detection_stats(detection_time)

                        # Store detection data
                        self.latest_faces = faces
                        self.latest_foreheads = foreheads

                    except Exception as e:
                        self.logger.error(f"Detection failed: {e}")
                        self.latest_faces = []
                        self.latest_foreheads = []

                self.latest_frame = frame.copy()
                return frame
            else:
                self.logger.warning("Failed to capture RGB frame")
                return None

        except Exception as e:
            self.logger.error(f"RGB frame capture failed: {e}")
            return None

    async def get_rgb_frame_with_detection(self) -> Optional[np.ndarray]:
        """Get RGB frame with face detection drawn"""
        if not self.is_initialized or not self.cap:
            return None

        try:
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(
                self.executor,
                self.cap.read
            )

            if ret and frame is not None:
                self.frame_count += 1

                # Run face detection if available
                if self.face_detector:
                    start_time = time.time()

                    # Perform detection and draw on frame
                    faces, foreheads = self.face_detector.detect_faces_and_foreheads(
                        frame, draw_detections=True
                    )

                    detection_time = time.time() - start_time
                    self._update_detection_stats(detection_time)

                    # Store detection data
                    self.latest_faces = faces
                    self.latest_foreheads = foreheads

                self.latest_frame = frame.copy()

                # Add frame info overlay
                self._draw_frame_info(frame)

                return frame
            else:
                self.logger.warning("Failed to capture frame with detection")
                return None

        except Exception as e:
            self.logger.error(f"Frame capture with detection failed: {e}")
            return None

    async def get_thermal_frame(self) -> Optional[np.ndarray]:
        """Get simulated thermal frame from webcam (for demo)"""
        if not self.is_initialized or not self.cap:
            return None

        try:
            # Get RGB frame first
            rgb_frame = await self.get_rgb_frame()

            if rgb_frame is not None:
                # Convert to simulated thermal data
                thermal_frame = self._simulate_thermal_data(rgb_frame)
                return thermal_frame
            else:
                return None

        except Exception as e:
            self.logger.error(f"Thermal frame simulation failed: {e}")
            return None

    def _simulate_thermal_data(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame to simulated thermal data"""
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        # Simulate thermal data (map grayscale to temperature-like values)
        # Map 0-255 to 1000-1100 (representing 30-40°C range in raw thermal units)
        thermal_sim = gray.astype(np.float32)
        thermal_sim = (thermal_sim / 255.0) * 100 + 1000

        # Add some realistic noise
        noise = np.random.normal(0, 1.5, thermal_sim.shape)
        thermal_sim += noise

        # Add some spatial variation to make it look more realistic
        # Slightly warmer in center (face area)
        h, w = thermal_sim.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

        # Add slight temperature gradient (warmer in center)
        temperature_boost = 10 * (1 - distance_from_center / max_distance)
        thermal_sim += temperature_boost

        return thermal_sim

    def _update_detection_stats(self, detection_time: float):
        """Update detection performance statistics"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 20:  # Keep last 20 measurements
            self.detection_times.pop(0)

        self.avg_detection_time = sum(self.detection_times) / len(self.detection_times)

    def _draw_frame_info(self, frame: np.ndarray):
        """Draw frame information overlay"""
        frame_height, frame_width = frame.shape[:2]

        # Calculate FPS
        current_fps = 0
        if self.start_time:
            elapsed = time.time() - self.start_time.timestamp()
            if elapsed > 0:
                current_fps = self.frame_count / elapsed

        # Prepare info text
        info_text = [
            f"Webcam Camera",
            f"Resolution: {frame_width}x{frame_height}",
            f"FPS: {current_fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Time: {time.strftime('%H:%M:%S')}"
        ]

        # Add detection info if available
        if self.face_detector:
            detection_info = self.face_detector.get_detection_info()
            info_text.extend([
                f"Detection: {detection_info['method'].upper()}",
                f"Faces: {len(self.latest_faces)}",
                f"Det FPS: {detection_info.get('fps', 0):.1f}"
            ])

        # Draw semi-transparent background
        info_height = len(info_text) * 25 + 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (10, 10), (280, info_height), (0, 255, 0), 1)

        # Draw info text
        for i, text in enumerate(info_text):
            color = (255, 255, 255)
            if "Detection:" in text:
                if "YOLO11" in text:
                    color = (0, 255, 255)  # Yellow for YOLO11
                elif "YuNet" in text:
                    color = (255, 0, 255)  # Magenta for YuNet
                elif "DNN" in text:
                    color = (255, 255, 0)  # Cyan for DNN
            elif "FPS:" in text:
                fps_val = float(text.split(": ")[1])
                color = (0, 255, 0) if fps_val > 15 else (0, 165, 255)

            cv2.putText(frame, text, (15, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw detection status at bottom
        if self.face_detector and self.latest_faces:
            best_face, best_forehead = self.face_detector.get_best_face_and_forehead(
                self.latest_faces, self.latest_foreheads
            )

            if best_face:
                status_y = frame_height - 60

                # Semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, status_y - 10), (400, frame_height - 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, f"Best Face: {best_face.confidence:.1%} confidence",
                            (15, status_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame, f"Size: {best_face.width}x{best_face.height} pixels",
                            (15, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if best_forehead:
                    cv2.putText(frame, f"Forehead: {best_forehead.confidence:.1%} confidence",
                                (15, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def get_latest_detection_data(self) -> Dict[str, Any]:
        """Get latest detection data for API"""
        best_face, best_forehead = None, None

        if self.face_detector and self.latest_faces and self.latest_foreheads:
            best_face, best_forehead = self.face_detector.get_best_face_and_forehead(
                self.latest_faces, self.latest_foreheads
            )

        frame_height, frame_width = (480, 640)
        if self.latest_frame is not None:
            frame_height, frame_width = self.latest_frame.shape[:2]

        detection_method = "none"
        if self.face_detector:
            detection_info = self.face_detector.get_detection_info()
            detection_method = detection_info['method']

        return {
            'frame_width': frame_width,
            'frame_height': frame_height,
            'face_detection': best_face,
            'forehead_detection': best_forehead,
            'all_faces': self.latest_faces,
            'all_foreheads': self.latest_foreheads,
            'detection_method': detection_method,
            'avg_detection_time': self.avg_detection_time,
            'frame_count': self.frame_count
        }

    def is_connected(self) -> bool:
        """Check if webcam is connected"""
        return self.cap is not None and self.cap.isOpened()

    def get_camera_info(self) -> Dict[str, Any]:
        """Get webcam information"""
        info = {
            "camera_type": "webcam",
            "camera_index": self.camera_index,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "connected": self.is_connected(),
            "frame_count": self.frame_count,
            "has_face_detector": self.face_detector is not None
        }

        if self.cap and self.cap.isOpened():
            try:
                info["backend"] = self.cap.getBackendName()
                info["actual_width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["actual_height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info["actual_fps"] = self.cap.get(cv2.CAP_PROP_FPS)
                info["brightness"] = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                info["contrast"] = self.cap.get(cv2.CAP_PROP_CONTRAST)
            except Exception as e:
                info["error"] = str(e)

        # Add detection info if available
        if self.face_detector:
            detection_info = self.face_detector.get_detection_info()
            info["detection_info"] = detection_info

        return info

    def release(self):
        """Release webcam resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

        self.executor.shutdown(wait=True)
        self.is_initialized = False
        self.logger.info("Webcam resources released")

    # Additional utility methods

    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Set camera property"""
        if self.cap and self.cap.isOpened():
            try:
                return self.cap.set(property_id, value)
            except Exception as e:
                self.logger.error(f"Failed to set camera property {property_id}: {e}")
                return False
        return False

    def get_camera_property(self, property_id: int) -> float:
        """Get camera property"""
        if self.cap and self.cap.isOpened():
            try:
                return self.cap.get(property_id)
            except Exception as e:
                self.logger.error(f"Failed to get camera property {property_id}: {e}")
                return 0.0
        return 0.0

    def save_frame(self, filename: str = None) -> bool:
        """Save current frame to file"""
        if self.latest_frame is not None:
            if filename is None:
                filename = f"webcam_frame_{int(time.time())}.jpg"

            try:
                cv2.imwrite(filename, self.latest_frame)
                self.logger.info(f"Frame saved to {filename}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save frame: {e}")
                return False
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get camera performance statistics"""
        current_fps = 0
        if self.start_time:
            elapsed = time.time() - self.start_time.timestamp()
            if elapsed > 0:
                current_fps = self.frame_count / elapsed

        stats = {
            "current_fps": current_fps,
            "target_fps": self.fps,
            "frame_count": self.frame_count,
            "uptime_seconds": time.time() - self.start_time.timestamp() if self.start_time else 0,
            "avg_detection_time": self.avg_detection_time,
            "detection_fps": 1.0 / max(self.avg_detection_time, 0.001) if self.avg_detection_time > 0 else 0,
            "has_detection_data": len(self.latest_faces) > 0
        }

        return stats