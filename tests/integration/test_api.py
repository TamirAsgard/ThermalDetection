import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import sys
import os
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.routes import ThermalAPIRoutes
from src.core.temperature_analyzer import TemperatureReading
from src.core.face_detector import FaceDetection


class TestThermalAPIRoutes(unittest.TestCase):
    """Test cases for ThermalAPIRoutes class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock components
        self.mock_thermal_processor = Mock()
        self.mock_camera_manager = Mock()
        self.mock_config = Mock()

        # Create API routes instance
        self.api_routes = ThermalAPIRoutes(
            self.mock_thermal_processor,
            self.mock_camera_manager,
            self.mock_config
        )

        # Create test client
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(self.api_routes.router)
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check endpoint"""
        self.mock_camera_manager.is_connected.return_value = True
        self.mock_thermal_processor.is_running = True

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("services", data)
        self.assertTrue(data["services"]["camera"])
        self.assertTrue(data["services"]["thermal_processor"])

    def test_temperature_endpoint_success(self):
        """Test temperature endpoint with successful reading"""
        # Mock temperature reading
        test_reading = TemperatureReading(
            avg_temperature=36.5,
            max_temperature=37.0,
            min_temperature=36.0,
            pixel_count=100,
            confidence=0.9,
            timestamp=datetime.now(),
            is_fever=False
        )
        self.mock_thermal_processor.get_averaged_reading.return_value = test_reading

        # Mock detection data
        test_face = FaceDetection(x=200, y=150, width=200, height=200, confidence=0.85)
        detection_data = {
            'frame_width': 640,
            'frame_height': 480,
            'face_detection': test_face,
            'forehead_detection': {
                'x': 220, 'y': 170, 'width': 160, 'height': 60, 'confidence': 0.8
            }
        }
        self.mock_thermal_processor.get_latest_detection_data.return_value = detection_data

        response = self.client.get("/temperature")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["temperature"], 36.5)
        self.assertFalse(data["is_fever"])
        self.assertEqual(data["confidence"], 0.9)
        self.assertIn("face_detection", data)
        self.assertIn("forehead_detection", data)

    def test_temperature_endpoint_no_reading(self):
        """Test temperature endpoint with no reading available"""
        self.mock_thermal_processor.get_averaged_reading.return_value = None

        response = self.client.get("/temperature")

        self.assertEqual(response.status_code, 404)

    def test_status_endpoint(self):
        """Test system status endpoint"""
        self.mock_thermal_processor.is_running = True
        self.mock_camera_manager.is_connected.return_value = True
        self.mock_thermal_processor.last_reading = Mock()
        self.mock_thermal_processor.last_reading.timestamp = datetime.now()

        response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_running"])
        self.assertTrue(data["camera_connected"])
        self.assertIn("total_measurements", data)
        self.assertIn("uptime_seconds", data)

    def test_camera_info_endpoint(self):
        """Test camera info endpoint"""
        self.mock_camera_manager.camera = Mock()
        mock_info = {
            "thermal_device_type": "webcam",
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30,
            "rgb_connected": True,
            "thermal_connected": True
        }
        self.mock_camera_manager.camera.get_camera_info.return_value = mock_info

        response = self.client.get("/camera/info")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["thermal_device_type"], "webcam")
        self.assertEqual(data["frame_width"], 640)
        self.assertTrue(data["rgb_connected"])

    def test_camera_info_no_camera(self):
        """Test camera info endpoint when no camera initialized"""
        self.mock_camera_manager.camera = None

        response = self.client.get("/camera/info")

        self.assertEqual(response.status_code, 503)

    def test_config_update_endpoint(self):
        """Test configuration update endpoint"""
        update_data = {
            "fever_threshold": 38.0,
            "confidence_threshold": 0.8,
            "calibration_offset": 0.5
        }

        response = self.client.post("/config/update", json=update_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)

    def test_config_update_invalid_data(self):
        """Test configuration update with invalid data"""
        invalid_data = {
            "fever_threshold": 100.0,  # Invalid temperature
            "confidence_threshold": 2.0  # Invalid confidence (>1.0)
        }

        response = self.client.post("/config/update", json=invalid_data)

        # Should return 422 for validation error or 400 for bad request
        self.assertIn(response.status_code, [400, 422])

    def test_camera_restart_endpoint(self):
        """Test camera restart endpoint"""
        response = self.client.post("/camera/restart")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints"""

    def setUp(self):
        """Set up integration test fixtures"""
        # Create more realistic mocks
        self.mock_thermal_processor = Mock()
        self.mock_camera_manager = Mock()
        self.mock_config = Mock()

        # Setup default return values
        self.mock_camera_manager.is_connected.return_value = True
        self.mock_thermal_processor.is_running = True

    def test_api_workflow(self):
        """Test complete API workflow"""
        api_routes = ThermalAPIRoutes(
            self.mock_thermal_processor,
            self.mock_camera_manager,
            self.mock_config
        )

        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_routes.router)
        client = TestClient(app)

        # Test health check
        health_response = client.get("/health")
        self.assertEqual(health_response.status_code, 200)

        # Test status
        status_response = client.get("/status")
        self.assertEqual(status_response.status_code, 200)

        # Test camera info (should fail without camera)
        self.mock_camera_manager.camera = None
        camera_response = client.get("/camera/info")
        self.assertEqual(camera_response.status_code, 503)

    def test_error_handling(self):
        """Test API error handling"""
        # Setup failing mocks
        self.mock_thermal_processor.get_averaged_reading.side_effect = Exception("Test error")

        api_routes = ThermalAPIRoutes(
            self.mock_thermal_processor,
            self.mock_camera_manager,
            self.mock_config
        )

        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(api_routes.router)
        client = TestClient(app)

        response = client.get("/temperature")
        self.assertEqual(response.status_code, 500)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestThermalAPIRoutes))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)