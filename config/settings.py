from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path


@dataclass
class CameraSettings:
    """Camera configuration settings"""
    thermal_camera_index: int = 0
    rgb_camera_index: int = 1
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    thermal_device_type: str = "webcam"
    auto_exposure: bool = True
    auto_white_balance: bool = True
    buffer_size: int = 1


@dataclass
class DetectionSettings:
    """Detection algorithm settings"""
    face_confidence_threshold: float = 0.7
    forehead_region_ratio: float = 0.3
    min_face_size: int = 80
    max_face_size: int = 500
    face_cascade_path: str = "haarcascade_frontalface_default.xml"
    dnn_model_path: str = "opencv_face_detector_uint8.pb"
    dnn_config_path: str = "opencv_face_detector.pbtxt"
    detection_method: str = "cascade"
    enable_tracking: bool = True
    tracking_timeout: float = 2.0
    max_faces: int = 3
    closest_face_only: bool = True


@dataclass
class TemperatureSettings:
    """Temperature measurement settings"""
    temp_min: float = 30.0
    temp_max: float = 45.0
    fever_threshold: float = 37.5
    high_fever_threshold: float = 39.0
    calibration_offset: float = 0.0
    calibration_enabled: bool = True
    smoothing_kernel_size: int = 5
    averaging_samples: int = 10
    measurement_accuracy: float = 0.1
    ambient_temp_compensation: bool = True
    default_ambient_temp: float = 22.0
    temperature_unit: str = "celsius"


@dataclass
class APISettings:
    """API server settings"""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ])
    websocket_enabled: bool = True
    websocket_update_rate: int = 10
    max_connections: int = 100
    enable_video_stream: bool = True
    enable_screenshots: bool = True
    enable_recording: bool = False
    api_key_required: bool = False
    api_key: str = "your-secret-key"
    rate_limiting: bool = True
    max_requests_per_minute: int = 60


@dataclass
class LoggingSettings:
    """Logging configuration settings"""
    level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/thermal_system.log"
    max_log_size_mb: int = 10
    log_backup_count: int = 5
    log_measurements: bool = True
    measurement_log_path: str = "logs/measurements.log"
    log_to_console: bool = True
    console_log_level: str = "INFO"


@dataclass
class AlertSettings:
    """Alert system configuration"""
    enable_alerts: bool = True
    enable_sound_alerts: bool = True
    fever_sound_file: str = "sounds/fever_alert.wav"
    normal_sound_file: str = "sounds/normal_beep.wav"
    enable_visual_alerts: bool = True
    fever_alert_color: List[int] = field(default_factory=lambda: [255, 0, 0])
    normal_alert_color: List[int] = field(default_factory=lambda: [0, 255, 0])
    enable_email_alerts: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_user: str = "your_email@gmail.com"
    email_password: str = "your_app_password"
    alert_recipients: List[str] = field(default_factory=lambda: [
        "admin@yourcompany.com",
        "security@yourcompany.com"
    ])
    enable_webhook_alerts: bool = False
    webhook_url: str = "https://your-webhook-url.com/alerts"


@dataclass
class DatabaseSettings:
    """Database configuration"""
    type: str = "sqlite"
    sqlite_path: str = "data/thermal_readings.db"
    host: str = "localhost"
    port: int = 3306
    database: str = "thermal_detection"
    username: str = "thermal_user"
    password: str = "secure_password"
    keep_readings_days: int = 90
    auto_cleanup: bool = True


@dataclass
class PerformanceSettings:
    """Performance configuration"""
    use_gpu: bool = False
    num_worker_threads: int = 2
    frame_buffer_size: int = 3
    max_memory_usage_mb: int = 500
    process_every_nth_frame: int = 1
    optimize_for_accuracy: bool = True
    enable_face_tracking: bool = True


@dataclass
class CalibrationSettings:
    """Calibration configuration"""
    enable_auto_calibration: bool = False
    calibration_interval_hours: int = 24
    blackbody_temperature: float = 37.0
    blackbody_coordinates: List[int] = field(default_factory=lambda: [320, 240])
    use_ambient_compensation: bool = True
    ambient_sensor_enabled: bool = False
    store_calibration_history: bool = True
    calibration_log_path: str = "data/calibration_history.json"


@dataclass
class DevelopmentSettings:
    """Development and testing configuration"""
    demo_mode: bool = True
    simulate_fever_chance: float = 0.1
    save_test_images: bool = False
    test_image_path: str = "test_data/images/"
    show_debug_info: bool = True
    draw_face_rectangles: bool = True
    draw_forehead_region: bool = True
    show_temperature_overlay: bool = True
    show_fps_counter: bool = True
    log_performance_stats: bool = True


@dataclass
class HardwareSettings:
    """Hardware integration configuration"""
    enable_ambient_sensor: bool = False
    ambient_sensor_type: str = "DHT22"
    ambient_sensor_pin: int = 4
    enable_external_display: bool = False
    display_type: str = "LCD1602"
    display_i2c_address: int = 0x27
    enable_gpio_controls: bool = False
    status_led_pin: int = 18
    buzzer_pin: int = 12
    enable_relay_control: bool = False
    relay_pin: int = 22
    relay_trigger_on_fever: bool = True


@dataclass
class BackupSettings:
    """Backup and recovery configuration"""
    enable_automatic_backup: bool = True
    backup_interval_hours: int = 6
    backup_location: str = "backups/"
    max_backup_files: int = 10
    backup_config_changes: bool = True
    config_backup_path: str = "backups/config/"


@dataclass
class SystemSettings:
    """System information and compliance"""
    system_name: str = "Thermal Detection Station 1"
    location: str = "Main Entrance"
    operator: str = "Security Team"
    installation_date: str = "2024-01-01"
    data_privacy_mode: bool = True
    anonymize_data: bool = False
    consent_required: bool = False
    maintenance_mode: bool = False
    maintenance_message: str = "System under maintenance"


@dataclass
class ThermalConfig:
    """Main configuration class that encompasses all settings"""
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    temperature: TemperatureSettings = field(default_factory=TemperatureSettings)
    api: APISettings = field(default_factory=APISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    development: DevelopmentSettings = field(default_factory=DevelopmentSettings)
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    backup: BackupSettings = field(default_factory=BackupSettings)
    system: SystemSettings = field(default_factory=SystemSettings)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ThermalConfig':
        """Load configuration from YAML file"""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            # Create configuration object with all sections
            return cls(
                camera=CameraSettings(**config_data.get('camera', {})),
                detection=DetectionSettings(**config_data.get('detection', {})),
                temperature=TemperatureSettings(**config_data.get('temperature', {})),
                api=APISettings(**config_data.get('api', {})),
                logging=LoggingSettings(**config_data.get('logging', {})),
                alerts=AlertSettings(**config_data.get('alerts', {})),
                database=DatabaseSettings(**config_data.get('database', {})),
                performance=PerformanceSettings(**config_data.get('performance', {})),
                calibration=CalibrationSettings(**config_data.get('calibration', {})),
                development=DevelopmentSettings(**config_data.get('development', {})),
                hardware=HardwareSettings(**config_data.get('hardware', {})),
                backup=BackupSettings(**config_data.get('backup', {})),
                system=SystemSettings(**config_data.get('system', {}))
            )

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def to_yaml(self, config_path: str) -> None:
        """Save current configuration to YAML file"""
        config_dict = {
            'camera': self._dataclass_to_dict(self.camera),
            'detection': self._dataclass_to_dict(self.detection),
            'temperature': self._dataclass_to_dict(self.temperature),
            'api': self._dataclass_to_dict(self.api),
            'logging': self._dataclass_to_dict(self.logging),
            'alerts': self._dataclass_to_dict(self.alerts),
            'database': self._dataclass_to_dict(self.database),
            'performance': self._dataclass_to_dict(self.performance),
            'calibration': self._dataclass_to_dict(self.calibration),
            'development': self._dataclass_to_dict(self.development),
            'hardware': self._dataclass_to_dict(self.hardware),
            'backup': self._dataclass_to_dict(self.backup),
            'system': self._dataclass_to_dict(self.system)
        }

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        else:
            return obj

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate camera settings
        if self.camera.frame_width <= 0 or self.camera.frame_height <= 0:
            errors.append("Camera frame dimensions must be positive")

        if self.camera.fps <= 0:
            errors.append("Camera FPS must be positive")

        # Validate temperature settings
        if self.temperature.temp_min >= self.temperature.temp_max:
            errors.append("Temperature min must be less than max")

        if not (30.0 <= self.temperature.fever_threshold <= 45.0):
            errors.append("Fever threshold should be between 30-45°C")

        # Validate API settings
        if not (1024 <= self.api.port <= 65535):
            errors.append("API port must be between 1024-65535")

        # Validate detection settings
        if not (0.0 <= self.detection.face_confidence_threshold <= 1.0):
            errors.append("Face confidence threshold must be between 0.0-1.0")

        return errors

    def get_section(self, section_name: str):
        """Get a specific configuration section"""
        return getattr(self, section_name, None)

    def update_section(self, section_name: str, updates: Dict[str, Any]) -> bool:
        """Update a specific configuration section"""
        try:
            section = getattr(self, section_name, None)
            if section is None:
                return False

            for key, value in updates.items():
                if hasattr(section, key):
                    setattr(section, key, value)
                else:
                    print(f"Warning: Unknown setting '{key}' in section '{section_name}'")

            return True
        except Exception:
            return False
