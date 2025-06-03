import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import time
import cv2
from datetime import datetime
from contextlib import asynccontextmanager

from .camera_factory import CameraFactory
from .base_camera import BaseCameraInterface


class CameraManager:
    """Manage camera operations and lifecycle"""

    def __init__(self, config):
        self.config = config
        self.camera: Optional[BaseCameraInterface] = None
        self.logger = logging.getLogger(__name__)
        self.is_running = False

        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None
        self.frame_rate = 0.0

        # Error handling
        self.consecutive_failures = 0
        self.max_failures = 5
        self.reconnect_delay = 2.0

        # Frame buffers for stability
        self.rgb_frame_buffer = None
        self.thermal_frame_buffer = None
        self.buffer_lock = asyncio.Lock()

        # Recording state
        self.recording_enabled = False
        self.recording_dir = None

    async def initialize(self) -> bool:
        """Initialize camera manager"""
        try:
            self.logger.info("Initializing camera manager...")

            # Create camera instance - use "thermal" type but it will handle webcam
            camera_type = self.config.camera.thermal_device_type.lower()
            self.camera = CameraFactory.create_camera(self.config, camera_type)
            if not self.camera:
                self.logger.error("Failed to create camera instance")
                return False

            # Initialize camera
            success = await self.camera.initialize()
            if success:
                self.is_running = True
                self.start_time = time.time()
                self.consecutive_failures = 0
                self.logger.info("Camera manager initialized successfully")

                # Start monitoring task
                asyncio.create_task(self._monitor_performance())
            else:
                self.logger.error("Camera initialization failed")

            return success

        except Exception as e:
            self.logger.error(f"Camera manager initialization error: {e}")
            return False

    async def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame with error handling and buffering"""
        if not self.is_running or not self.camera:
            return None

        try:
            frame = await self.camera.get_rgb_frame()

            if frame is not None:
                async with self.buffer_lock:
                    self.rgb_frame_buffer = frame.copy()
                self.consecutive_failures = 0
                self._update_frame_stats()
                return frame
            else:
                self.consecutive_failures += 1
                await self._handle_frame_failure("RGB")
                # Return buffered frame if available
                return self.rgb_frame_buffer

        except Exception as e:
            self.logger.error(f"RGB frame capture error: {e}")
            self.consecutive_failures += 1
            await self._handle_frame_failure("RGB")
            return self.rgb_frame_buffer

    async def get_thermal_frame(self) -> Optional[np.ndarray]:
        """Get thermal frame with error handling and buffering"""
        if not self.is_running or not self.camera:
            return None

        try:
            frame = await self.camera.get_thermal_frame()

            if frame is not None:
                async with self.buffer_lock:
                    self.thermal_frame_buffer = frame.copy()
                self.consecutive_failures = 0
                self._update_frame_stats()
                return frame
            else:
                self.consecutive_failures += 1
                await self._handle_frame_failure("Thermal")
                # Return buffered frame if available
                return self.thermal_frame_buffer

        except Exception as e:
            self.logger.error(f"Thermal frame capture error: {e}")
            self.consecutive_failures += 1
            await self._handle_frame_failure("Thermal")
            return self.thermal_frame_buffer

    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return (self.camera is not None and
                self.camera.is_connected() and
                self.consecutive_failures < self.max_failures)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive camera status"""
        status = {
            "is_running": self.is_running,
            "is_connected": self.is_connected(),
            "consecutive_failures": self.consecutive_failures,
            "frame_rate": self.frame_rate,
            "frame_count": self.frame_count,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "has_rgb_buffer": self.rgb_frame_buffer is not None,
            "has_thermal_buffer": self.thermal_frame_buffer is not None,
            "camera_info": {}
        }

        if self.camera:
            try:
                status["camera_info"] = self.camera.get_camera_info()
            except Exception as e:
                self.logger.error(f"Failed to get camera info: {e}")
                status["camera_info"] = {"error": str(e)}

        return status

    async def restart_camera(self) -> bool:
        """Restart camera connection"""
        self.logger.info("Restarting camera...")

        # Stop current camera
        await self.stop()

        # Wait before reconnecting
        await asyncio.sleep(self.reconnect_delay)

        # Reinitialize
        success = await self.initialize()

        if success:
            self.logger.info("Camera restart successful")
        else:
            self.logger.error("Camera restart failed")

        return success

    async def stop(self):
        """Stop camera manager"""
        self.logger.info("Stopping camera manager...")
        self.is_running = False

        if self.camera:
            try:
                self.camera.release()
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None

        # Clear buffers
        async with self.buffer_lock:
            self.rgb_frame_buffer = None
            self.thermal_frame_buffer = None

        self.logger.info("Camera manager stopped")

    async def _handle_frame_failure(self, frame_type: str):
        """Handle frame capture failures"""
        if self.consecutive_failures >= self.max_failures:
            self.logger.error(
                f"Too many consecutive {frame_type} frame failures "
                f"({self.consecutive_failures}). Attempting restart..."
            )

            # Try to restart camera automatically
            asyncio.create_task(self._auto_restart())

    async def _auto_restart(self):
        """Automatically restart camera after failures"""
        try:
            success = await self.restart_camera()
            if success:
                self.logger.info("Auto-restart successful")
            else:
                self.logger.error("Auto-restart failed")
                # Wait before next attempt
                await asyncio.sleep(self.reconnect_delay * 2)
        except Exception as e:
            self.logger.error(f"Auto-restart error: {e}")

    def _update_frame_stats(self):
        """Update frame rate statistics"""
        current_time = time.time()
        self.frame_count += 1

        if self.last_frame_time is not None:
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                instant_fps = 1.0 / time_diff
                # Exponential moving average for smooth frame rate
                alpha = 0.1
                self.frame_rate = alpha * instant_fps + (1 - alpha) * self.frame_rate

        self.last_frame_time = current_time

    async def _monitor_performance(self):
        """Monitor camera performance in background"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                if self.is_running:
                    self.logger.debug(
                        f"Camera performance: FPS={self.frame_rate:.1f}, "
                        f"Frames={self.frame_count}, "
                        f"Failures={self.consecutive_failures}"
                    )

                    # Log warning if frame rate is too low
                    expected_fps = self.config.camera.fps
                    if self.frame_rate < expected_fps * 0.5:
                        self.logger.warning(
                            f"Low frame rate: {self.frame_rate:.1f} < {expected_fps * 0.5:.1f}"
                        )

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")

    async def get_synchronized_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get temporally synchronized RGB and thermal frames"""
        try:
            # For webcam demo, both are the same camera, so capture once and use for both
            rgb_frame = await self.get_rgb_frame()

            # For demo purposes, thermal frame is the same as RGB (converted to grayscale)
            if rgb_frame is not None:
                thermal_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                return rgb_frame, thermal_frame
            else:
                return None, None

        except Exception as e:
            self.logger.error(f"Synchronized frame capture failed: {e}")
            return None, None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics information"""
        diagnostics = {
            "camera_manager": {
                "is_running": self.is_running,
                "consecutive_failures": self.consecutive_failures,
                "max_failures": self.max_failures,
                "frame_rate": self.frame_rate,
                "frame_count": self.frame_count,
                "uptime": time.time() - self.start_time if self.start_time else 0,
                "reconnect_delay": self.reconnect_delay
            },
            "buffers": {
                "rgb_buffer_available": self.rgb_frame_buffer is not None,
                "thermal_buffer_available": self.thermal_frame_buffer is not None,
                "rgb_buffer_shape": self.rgb_frame_buffer.shape if self.rgb_frame_buffer is not None else None,
                "thermal_buffer_shape": self.thermal_frame_buffer.shape if self.thermal_frame_buffer is not None else None
            },
            "camera_info": {}
        }

        if self.camera:
            try:
                diagnostics["camera_info"] = self.camera.get_camera_info()
            except Exception as e:
                diagnostics["camera_info"] = {"error": str(e)}

        return diagnostics

    async def test_camera_connectivity(self) -> Dict[str, Any]:
        """Test camera connectivity and performance"""
        test_results = {
            "rgb_camera": {
                "connected": False,
                "frame_capture": False,
                "frame_rate": 0.0,
                "resolution": None,
                "error": None
            },
            "thermal_camera": {
                "connected": False,
                "frame_capture": False,
                "frame_rate": 0.0,
                "resolution": None,
                "error": None
            },
            "overall_status": "failed"
        }

        try:
            if not self.camera:
                test_results["error"] = "Camera not initialized"
                return test_results

            # Test RGB camera
            try:
                start_time = time.time()
                rgb_frames = 0

                for _ in range(10):  # Test 10 frames
                    frame = await self.get_rgb_frame()
                    if frame is not None:
                        rgb_frames += 1
                        if test_results["rgb_camera"]["resolution"] is None:
                            test_results["rgb_camera"]["resolution"] = frame.shape
                    await asyncio.sleep(0.1)

                elapsed = time.time() - start_time
                test_results["rgb_camera"]["connected"] = True
                test_results["rgb_camera"]["frame_capture"] = rgb_frames > 0
                test_results["rgb_camera"]["frame_rate"] = rgb_frames / elapsed if elapsed > 0 else 0

            except Exception as e:
                test_results["rgb_camera"]["error"] = str(e)

            # Test thermal camera (for webcam demo, this will be similar to RGB)
            try:
                start_time = time.time()
                thermal_frames = 0

                for _ in range(10):  # Test 10 frames
                    frame = await self.get_thermal_frame()
                    if frame is not None:
                        thermal_frames += 1
                        if test_results["thermal_camera"]["resolution"] is None:
                            test_results["thermal_camera"]["resolution"] = frame.shape
                    await asyncio.sleep(0.1)

                elapsed = time.time() - start_time
                test_results["thermal_camera"]["connected"] = True
                test_results["thermal_camera"]["frame_capture"] = thermal_frames > 0
                test_results["thermal_camera"]["frame_rate"] = thermal_frames / elapsed if elapsed > 0 else 0

            except Exception as e:
                test_results["thermal_camera"]["error"] = str(e)

            # Overall status
            rgb_ok = test_results["rgb_camera"]["frame_capture"]
            thermal_ok = test_results["thermal_camera"]["frame_capture"]

            if rgb_ok and thermal_ok:
                test_results["overall_status"] = "passed"
            elif rgb_ok or thermal_ok:
                test_results["overall_status"] = "partial"
            else:
                test_results["overall_status"] = "failed"

            return test_results

        except Exception as e:
            test_results["error"] = str(e)
            return test_results

    def set_camera_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Set camera parameters dynamically"""
        try:
            if not self.camera:
                return False

            # Update configuration
            for key, value in parameters.items():
                if hasattr(self.config.camera, key):
                    setattr(self.config.camera, key, value)
                    self.logger.info(f"Updated camera parameter {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to set camera parameters: {e}")
            return False

    def get_frame_statistics(self) -> Dict[str, Any]:
        """Get detailed frame capture statistics"""
        return {
            "total_frames": self.frame_count,
            "current_fps": self.frame_rate,
            "target_fps": self.config.camera.fps,
            "fps_efficiency": (self.frame_rate / self.config.camera.fps) * 100 if self.config.camera.fps > 0 else 0,
            "consecutive_failures": self.consecutive_failures,
            "failure_rate": (self.consecutive_failures / max(self.frame_count, 1)) * 100,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "buffer_status": {
                "rgb_available": self.rgb_frame_buffer is not None,
                "thermal_available": self.thermal_frame_buffer is not None
            }
        }

    async def warmup_cameras(self, frames: int = 30) -> bool:
        """Warmup cameras by capturing and discarding initial frames"""
        try:
            self.logger.info(f"Warming up cameras with {frames} frames...")

            successful_captures = 0

            for i in range(frames):
                # Capture and discard frames
                rgb_frame = await self.get_rgb_frame()
                thermal_frame = await self.get_thermal_frame()

                if rgb_frame is not None and thermal_frame is not None:
                    successful_captures += 1

                # Small delay between captures
                await asyncio.sleep(0.033)  # ~30 FPS

                if i % 10 == 0:
                    self.logger.debug(f"Warmup progress: {i}/{frames}")

            success_rate = (successful_captures / frames) * 100
            self.logger.info(f"Camera warmup completed. Success rate: {success_rate:.1f}%")

            return success_rate > 80  # Consider successful if >80% frames captured

        except Exception as e:
            self.logger.error(f"Camera warmup failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.is_running:
            # Note: This is synchronous, so we can't await stop()
            # In practice, you'd use an async context manager
            self.is_running = False
            if self.camera:
                self.camera.release()
                self.camera = None


# ============================================================================
# src/camera/thermal_camera.py - Fixed Implementation
# ============================================================================

import cv2
import numpy as np
from typing import Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_camera import BaseCameraInterface


class ThermalCamera(BaseCameraInterface):
    """Thermal camera implementation for AT300/AT600 series and webcam demo"""

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

            # For webcam demo, thermal is same as RGB
            if self.thermal_device_type == "webcam":
                success_thermal = success_rgb  # Use same camera
                self.thermal_cap = self.rgb_cap
            else:
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
        """Initialize AT300/AT600 thermal camera"""
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
        if not self.is_initialized:
            return None

        try:
            if self.thermal_device_type == "webcam":
                # For webcam demo, use RGB frame and convert to "thermal"
                rgb_frame = await self.get_rgb_frame()
                if rgb_frame is not None:
                    return self._process_thermal_frame(rgb_frame)
                return None
            else:
                # Real thermal camera
                if not self.thermal_cap:
                    return None

                loop = asyncio.get_event_loop()
                ret, frame = await loop.run_in_executor(
                    self.executor,
                    self.thermal_cap.read
                )

                if ret and frame is not None:
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
        elif self.thermal_device_type == "webcam":
            # For demo, simulate thermal data
            return self._simulate_thermal_data(thermal_frame)

        return thermal_frame.astype(np.float32)

    def _simulate_thermal_data(self, frame: np.ndarray) -> np.ndarray:
        """Simulate thermal data from webcam frame for demo"""
        # Convert grayscale to simulated temperature values
        # Map 0-255 to temperature-like values (e.g., 1000-1100 for 30-40°C range)
        thermal_sim = frame.astype(np.float32)
        thermal_sim = (thermal_sim / 255.0) * 100 + 1000  # Map to 1000-1100 range

        # Add some noise for realism
        noise = np.random.normal(0, 2, thermal_sim.shape)
        thermal_sim += noise

        return thermal_sim

    def _process_at_series_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process AT300/AT600 specific thermal data"""
        # Apply AT series specific calibration and processing
        # This is where you'd apply the manufacturer's calibration
        return frame.astype(np.float32)

    def is_connected(self) -> bool:
        """Check if cameras are connected"""
        rgb_connected = self.rgb_cap is not None and self.rgb_cap.isOpened()

        if self.thermal_device_type == "webcam":
            return rgb_connected
        else:
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

        if self.thermal_device_type == "webcam":
            info["thermal_connected"] = info["rgb_connected"]
            info["thermal_backend"] = "webcam_simulation"
        elif self.thermal_cap:
            info["thermal_connected"] = self.thermal_cap.isOpened()
            info["thermal_backend"] = self.thermal_cap.getBackendName()

        return info

    def release(self):
        """Release camera resources"""
        if self.rgb_cap:
            self.rgb_cap.release()
            self.rgb_cap = None

        if self.thermal_cap and self.thermal_cap != self.rgb_cap:
            self.thermal_cap.release()
            self.thermal_cap = None

        self.executor.shutdown(wait=True)
        self.is_initialized = False
        self.logger.info("Camera resources released")