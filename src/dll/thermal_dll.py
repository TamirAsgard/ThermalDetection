import ctypes
from ctypes import Structure, c_float, c_int, c_bool, c_char_p, POINTER
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor


# Define C structures for DLL interface
class ThermalReading(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("max_temperature", c_float),
        ("min_temperature", c_float),
        ("confidence", c_float),
        ("is_fever", c_bool),
        ("pixel_count", c_int),
        ("timestamp", c_int)  # Unix timestamp
    ]


class CameraStatus(Structure):
    _fields_ = [
        ("is_connected", c_bool),
        ("frame_width", c_int),
        ("frame_height", c_int),
        ("fps", c_int),
        ("device_type", c_char_p)
    ]


class ThermalDLL:
    """DLL interface for thermal detection system"""

    def __init__(self, config_path: str = "config/thermal_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Initialize the thermal system
        from ..main import ThermalDetectionSystem
        self.thermal_system = ThermalDetectionSystem(config_path)

        # DLL state
        self.is_initialized = False
        self.last_reading = None

    async def initialize_async(self) -> bool:
        """Initialize the thermal system asynchronously"""
        try:
            success = await self.thermal_system.initialize()
            self.is_initialized = success
            return success
        except Exception as e:
            self.logger.error(f"DLL initialization failed: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize the thermal system (blocking)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.initialize_async())
        finally:
            loop.close()

    async def get_temperature_async(self) -> ThermalReading:
        """Get temperature reading asynchronously"""
        try:
            reading = self.thermal_system.thermal_processor.get_averaged_reading()

            if reading:
                self.last_reading = ThermalReading(
                    temperature=reading.avg_temperature,
                    max_temperature=reading.max_temperature,
                    min_temperature=reading.min_temperature,
                    confidence=reading.confidence,
                    is_fever=reading.is_fever,
                    pixel_count=reading.pixel_count,
                    timestamp=int(reading.timestamp.timestamp())
                )
            else:
                # Return empty reading
                self.last_reading = ThermalReading(
                    temperature=0.0,
                    max_temperature=0.0,
                    min_temperature=0.0,
                    confidence=0.0,
                    is_fever=False,
                    pixel_count=0,
                    timestamp=0
                )

            return self.last_reading

        except Exception as e:
            self.logger.error(f"Temperature reading failed: {e}")
            return ThermalReading()

    def get_temperature(self) -> ThermalReading:
        """Get temperature reading (blocking)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_temperature_async())
        finally:
            loop.close()

    def get_camera_status(self) -> CameraStatus:
        """Get camera status"""
        try:
            status = self.thermal_system.camera_manager.get_status()
            camera_info = status.get("camera_info", {})

            return CameraStatus(
                is_connected=status["is_connected"],
                frame_width=camera_info.get("frame_width", 0),
                frame_height=camera_info.get("frame_height", 0),
                fps=camera_info.get("fps", 0),
                device_type=camera_info.get("thermal_device_type", "").encode()
            )

        except Exception as e:
            self.logger.error(f"Camera status failed: {e}")
            return CameraStatus()

    def set_fever_threshold(self, threshold: float) -> bool:
        """Set fever threshold"""
        try:
            self.thermal_system.config.temperature.fever_threshold = threshold
            return True
        except Exception as e:
            self.logger.error(f"Set fever threshold failed: {e}")
            return False

    def set_calibration_offset(self, offset: float) -> bool:
        """Set calibration offset"""
        try:
            self.thermal_system.config.temperature.calibration_offset = offset
            return True
        except Exception as e:
            self.logger.error(f"Set calibration offset failed: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.thermal_system:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.thermal_system.cleanup())
                finally:
                    loop.close()

            self.executor.shutdown(wait=True)
            self.is_initialized = False

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# Global DLL instance
_dll_instance = None


def get_dll_instance():
    """Get global DLL instance"""
    global _dll_instance
    if _dll_instance is None:
        _dll_instance = ThermalDLL()
    return _dll_instance