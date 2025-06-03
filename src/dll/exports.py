"""
DLL export functions for C/C++ integration
"""

import ctypes
from ctypes import c_bool, c_float, c_int, c_char_p, POINTER
import logging
from pathlib import Path

from .thermal_dll import ThermalDLL, ThermalReading, CameraStatus, get_dll_instance

# Setup logging for DLL
logging.basicConfig(level=logging.INFO)


# ============================================================================
# C-style export functions
# ============================================================================

def init_thermal_system(config_path: c_char_p = None) -> c_bool:
    """Initialize thermal detection system

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        dll = get_dll_instance()
        if config_path:
            dll.config_path = config_path.decode('utf-8')

        return dll.initialize()

    except Exception as e:
        logging.error(f"init_thermal_system failed: {e}")
        return False


def get_temperature_reading(reading: POINTER(ThermalReading)) -> c_bool:
    """Get current temperature reading

    Args:
        reading: Pointer to ThermalReading structure to fill

    Returns:
        True if reading obtained, False otherwise
    """
    try:
        dll = get_dll_instance()
        if not dll.is_initialized:
            return False

        current_reading = dll.get_temperature()
        reading.contents = current_reading
        return True

    except Exception as e:
        logging.error(f"get_temperature_reading failed: {e}")
        return False


def get_camera_status_info(status: POINTER(CameraStatus)) -> c_bool:
    """Get camera status information

    Args:
        status: Pointer to CameraStatus structure to fill

    Returns:
        True if status obtained, False otherwise
    """
    try:
        dll = get_dll_instance()
        current_status = dll.get_camera_status()
        status.contents = current_status
        return True

    except Exception as e:
        logging.error(f"get_camera_status_info failed: {e}")
        return False


def set_fever_threshold_value(threshold: c_float) -> c_bool:
    """Set fever threshold temperature

    Args:
        threshold: Fever threshold in Celsius

    Returns:
        True if set successfully, False otherwise
    """
    try:
        dll = get_dll_instance()
        return dll.set_fever_threshold(threshold)

    except Exception as e:
        logging.error(f"set_fever_threshold_value failed: {e}")
        return False


def set_calibration_offset_value(offset: c_float) -> c_bool:
    """Set temperature calibration offset

    Args:
        offset: Calibration offset in Celsius

    Returns:
        True if set successfully, False otherwise
    """
    try:
        dll = get_dll_instance()
        return dll.set_calibration_offset(offset)

    except Exception as e:
        logging.error(f"set_calibration_offset_value failed: {e}")
        return False


def cleanup_thermal_system() -> c_bool:
    """Cleanup thermal detection system

    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        global _dll_instance
        if _dll_instance:
            _dll_instance.cleanup()
            _dll_instance = None
        return True

    except Exception as e:
        logging.error(f"cleanup_thermal_system failed: {e}")
        return False


# ============================================================================
# Export function definitions for building shared library
# ============================================================================

# Define function prototypes for shared library export
EXPORTED_FUNCTIONS = {
    'init_thermal_system': init_thermal_system,
    'get_temperature_reading': get_temperature_reading,
    'get_camera_status_info': get_camera_status_info,
    'set_fever_threshold_value': set_fever_threshold_value,
    'set_calibration_offset_value': set_calibration_offset_value,
    'cleanup_thermal_system': cleanup_thermal_system
}

# Function signatures for C header generation
FUNCTION_SIGNATURES = """
// Thermal Detection System DLL Interface
// Generated header file

#ifndef THERMAL_DETECTION_H
#define THERMAL_DETECTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// Structures
typedef struct {
    float temperature;
    float max_temperature;
    float min_temperature;
    float confidence;
    bool is_fever;
    int pixel_count;
    int timestamp;
} ThermalReading;

typedef struct {
    bool is_connected;
    int frame_width;
    int frame_height;
    int fps;
    char* device_type;
} CameraStatus;

// Functions
bool init_thermal_system(const char* config_path);
bool get_temperature_reading(ThermalReading* reading);
bool get_camera_status_info(CameraStatus* status);
bool set_fever_threshold_value(float threshold);
bool set_calibration_offset_value(float offset);
bool cleanup_thermal_system(void);

#ifdef __cplusplus
}
#endif

#endif // THERMAL_DETECTION_H
"""


def generate_header_file():
    """Generate C header file for DLL interface"""
    header_path = Path("include/thermal_detection.h")
    header_path.parent.mkdir(exist_ok=True)

    with open(header_path, 'w') as f:
        f.write(FUNCTION_SIGNATURES)

    print(f"Header file generated: {header_path}")