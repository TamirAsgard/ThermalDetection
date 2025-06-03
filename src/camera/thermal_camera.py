import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_camera import BaseCameraInterface


class ThermalCamera(BaseCameraInterface):
    """Thermal camera implementation for AT300/AT600 series"""

    def __init__(self, config):
        self.config = config
        self.rgb_cap = None
        self.thermal_cap = None
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Camera-specific settings
        self.thermal_device_type = config.camera.thermal_device_type
        self.frame_width = config.camera.frame_width
        self.frame_height = config.camera.frame_height

    async def initialize(self) -> bool:
        """Initialize both RGB and thermal cameras"""
        try:
            # Initialize RGB camera for face detection
            success_rgb = await self._initialize_rgb_camera()
            success_thermal = await self._initialize_thermal_camera()

            self.is_initialized = success_rgb and success_thermal

            if self.is_initialized:
                self.logger.info("Cameras initialized successfully")
            else:
                self.logger.error("Failed to initialize cameras")

            return self.is_initialized

        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    async def _initialize_rgb_camera(self) -> bool:
        """Initialize RGB camera"""
        try:
            loop = asyncio.get_event_loop()
            self.rgb_cap = await loop.run_in_executor(
                self.executor,
                self._create_rgb_capture
            )

            if self.rgb_cap and self.rgb_cap.isOpened():
                # Test frame capture
                ret, frame = self.rgb_cap.read()
                return ret and frame is not None

            return False

        except Exception as e:
            self.logger.error(f"RGB camera initialization failed: {e}")
            return False

    def _create_rgb_capture(self):
        """Create RGB camera capture object"""
        cap = cv2.VideoCapture(self.config.camera.rgb_camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        return cap

    async def _initialize_thermal_camera(self) -> bool:
        """Initialize thermal camera"""
        try:
            if self.thermal_device_type in ["AT300", "AT600"]:
                return await self._initialize_at_series()
            else:
                return await self._initialize_generic_thermal()

        except Exception as e:
            self.logger.error(f"Thermal camera initialization failed: {e}")
            return False

    async def _initialize_at_series(self) -> bool:
        """Initialize AT300/AT600 thermal camera

        Note: This requires the specific AT series SDK.
        You'll need to replace this with actual AT SDK calls.
        """
        try:
            # For AT300/AT600, you would typically use:
            # - Import the AT SDK library
            # - Initialize the thermal device
            # - Set up temperature calibration

            # Placeholder implementation using generic capture
            loop = asyncio.get_event_loop()
            self.thermal_cap = await loop.run_in_executor(
                self.executor,
                self._create_thermal_capture
            )

            return self.thermal_cap is not None and self.thermal_cap.isOpened()

        except Exception as e:
            self.logger.error(f"AT series initialization failed: {e}")
            return False

    def _create_thermal_capture(self):
        """Create thermal camera capture object"""
        # For AT300/AT600, replace this with actual SDK initialization
        cap = cv2.VideoCapture(self.config.camera.thermal_camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        return cap

    async def _initialize_generic_thermal(self) -> bool:
        """Initialize generic thermal camera"""
        loop = asyncio.get_event_loop()
        self.thermal_cap = await loop.run_in_executor(
            self.executor,
            self._create_thermal_capture
        )

        return self.thermal_cap is not None and self.thermal_cap.isOpened()

    async def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame asynchronously"""
        if not self.is_initialized or not self.rgb_cap:
            return None

        try:
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(
                self.executor,
                self.rgb_cap.read
            )

            return frame if ret else None

        except Exception as e:
            self.logger.error(f"RGB frame capture failed: {e}")
            return None

    async def get_thermal_frame(self) -> Optional[np.ndarray]:
        """Get thermal frame asynchronously"""
        if not self.is_initialized or not self.thermal_cap:
            return None

        try:
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(
                self.executor,
                self.thermal_cap.read
            )

            if ret and frame is not None:
                # Convert to thermal data format
                return self._process_thermal_frame(frame)

            return None

        except Exception as e:
            self.logger.error(f"Thermal frame capture failed: {e}")
            return None

    def _process_thermal_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Process raw thermal frame"""
        # Convert to grayscale if needed
        if len(raw_frame.shape) == 3:
            thermal_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_frame = raw_frame

        # Apply any camera-specific thermal processing
        if self.thermal_device_type in ["AT300", "AT600"]:
            return self._process_at_series_frame(thermal_frame)

        return thermal_frame

    def _process_at_series_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process AT300/AT600 specific thermal data"""
        # Apply AT series specific calibration and processing
        # This is where you'd apply the manufacturer's calibration
        return frame.astype(np.float32)

    def is_connected(self) -> bool:
        """Check if cameras are connected"""
        rgb_connected = self.rgb_cap is not None and self.rgb_cap.isOpened()
        thermal_connected = self.thermal_cap is not None and self.thermal_cap.isOpened()
        return rgb_connected and thermal_connected

    def get_camera_info(self) -> dict:
        """Get camera information"""
        info = {
            "thermal_device_type": self.thermal_device_type,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.config.camera.fps,
            "rgb_connected": False,
            "thermal_connected": False
        }

        if self.rgb_cap:
            info["rgb_connected"] = self.rgb_cap.isOpened()
            info["rgb_backend"] = self.rgb_cap.getBackendName()

        if self.thermal_cap:
            info["thermal_connected"] = self.thermal_cap.isOpened()
            info["thermal_backend"] = self.thermal_cap.getBackendName()

        return info

    def release(self):
        """Release camera resources"""
        if self.rgb_cap:
            self.rgb_cap.release()
            self.rgb_cap = None

        if self.thermal_cap:
            self.thermal_cap.release()
            self.thermal_cap = None

        self.executor.shutdown(wait=True)
        self.is_initialized = False
        self.logger.info("Camera resources released")