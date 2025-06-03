from pathlib import Path
from typing import Optional
import logging

from .base_camera import BaseCameraInterface
from .thermal_camera import ThermalCamera
from .webcam_camera import WebcamCamera


class CameraFactory:
    """Factory for creating camera instances"""

    @staticmethod
    def create_camera(config, camera_type: str = "thermal"):
        """Create camera instance with YOLO11 priority"""
        logger = logging.getLogger(__name__)

        try:
            camera_type_lower = camera_type.lower()

            if camera_type_lower in ["thermal", "webcam"]:
                # Check if YOLO11 model exists
                yolo11_path = Path("../../model/yolov11n-face.pt")
                if yolo11_path.exists():
                    logger.info("Creating YOLO11 webcam detector")
                    return WebcamCamera(config)
                else:
                    logger.info("YOLO11 model not found, creating standard webcam detector")
                    # You can import your previous webcam detector here as fallback
                    # return WebcamCamera(config)  # Will use fallback methods

            else:
                logger.error(f"Unknown camera type: {camera_type}")
                return None

        except Exception as e:
            logger.error(f"Camera creation failed: {e}")
            return None
