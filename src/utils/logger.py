import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json


class ThermalLogger:
    """Custom logger for thermal detection system"""

    @staticmethod
    def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
        """Setup logging configuration"""

        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Configure logging
        log_level_obj = getattr(logging, log_level.upper(), logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_obj)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(log_level_obj)
        root_logger.addHandler(console_handler)

        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "thermal_system.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "thermal_errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)

        # Measurement log handler (for audit trail)
        measurement_handler = logging.handlers.RotatingFileHandler(
            log_path / "measurements.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=10
        )
        measurement_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )

        # Create measurement logger
        measurement_logger = logging.getLogger('measurements')
        measurement_logger.addHandler(measurement_handler)
        measurement_logger.setLevel(logging.INFO)
        measurement_logger.propagate = False

        logging.info("Logging system initialized")


class MeasurementLogger:
    """Logger for temperature measurements (audit trail)"""

    def __init__(self):
        self.logger = logging.getLogger('measurements')

    def log_measurement(self, reading, person_id=None):
        """Log temperature measurement"""
        measurement_data = {
            "timestamp": reading.timestamp.isoformat(),
            "temperature": reading.avg_temperature,
            "max_temperature": reading.max_temperature,
            "min_temperature": reading.min_temperature,
            "confidence": reading.confidence,
            "is_fever": reading.is_fever,
            "pixel_count": reading.pixel_count,
            "person_id": person_id
        }

        self.logger.info(json.dumps(measurement_data))