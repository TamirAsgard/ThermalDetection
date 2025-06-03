"""
Main entry point for thermal detection system
"""

import asyncio
import logging
from pathlib import Path
import signal
import sys

from config.settings import ThermalConfig
from src.camera.camera_manager import CameraManager
from src.core.thermal_processor import ThermalProcessor
from src.utils.logger import ThermalLogger
from src.api.app import create_app


class ThermalDetectionSystem:
    """Main thermal detection system"""

    def __init__(self, config_path: str = "config/thermal_config.yaml"):
        self.config = ThermalConfig.from_yaml(config_path)
        self.camera_manager = None
        self.thermal_processor = None
        self.logger = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging system"""
        ThermalLogger.setup_logging("INFO", "logs")
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """Initialize the thermal detection system"""
        try:
            self.logger.info("Initializing thermal detection system...")

            # Initialize camera manager
            self.camera_manager = CameraManager(self.config)
            if not await self.camera_manager.initialize():
                self.logger.error("Failed to initialize camera manager")
                return False

            # Initialize thermal processor
            self.thermal_processor = ThermalProcessor(self.config, self.camera_manager)

            self.logger.info("Thermal detection system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False

    async def start_processing(self):
        """Start the thermal processing"""
        if self.thermal_processor:
            await self.thermal_processor.start_processing()

    async def cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Cleaning up thermal detection system...")

        if self.thermal_processor:
            self.thermal_processor.stop_processing()

        if self.camera_manager:
            await self.camera_manager.stop()

        self.logger.info("Cleanup completed")


async def main():
    """Main application entry point"""
    # Setup signal handlers for graceful shutdown
    system = ThermalDetectionSystem()

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        asyncio.create_task(system.cleanup())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize system
        if not await system.initialize():
            print("Failed to initialize system")
            return

        # Start processing
        await system.start_processing()

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        logging.error(f"Application error: {e}")
    finally:
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())