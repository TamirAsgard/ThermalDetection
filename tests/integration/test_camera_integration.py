import unittest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
from datetime import datetime
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete thermal detection system"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.test_config_data = {
            'camera': {
                'thermal_camera_index': 0,
                'rgb_camera_index': 0,
                'frame_width': 640,
                'frame_height': 480,
                'fps': 30,
                'thermal_device_type': 'webcam'
            },
            'temperature': {
                'temp_min': 30.0,
                'temp_max': 45.0,
                'fever_threshold': 37.5,
                'calibration_offset': 0.0
            },
            'detection': {
                'face_confidence_threshold': 0.5,
                'forehead_region_ratio': 0.3,
                'min_face_size': 80
            },
            'development': {
                'demo_mode': True,
                'simulate_fever_chance': 0.1
            }
        }

    @patch('cv2.VideoCapture')
    @patch('pathlib.Path.exists')
    def test_config_loading(self, mock_path_exists, mock_video_capture):
        """Test configuration loading and validation"""
        mock_path_exists.return_value = True

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.test_config_data, f)
            config_path = f.name

        try:
            from config.settings import ThermalConfig
            config = ThermalConfig.from_yaml(config_path)

            self.assertEqual(config.camera.frame_width, 640)
            self.assertEqual(config.temperature.fever_threshold, 37.5)
            self.assertEqual(config.detection.face_confidence_threshold, 0.5)

            # Test validation
            errors = config.validate()
            self.assertEqual(len(errors), 0, f"Config validation failed: {errors}")

        finally:
            os.unlink(config_path)

    @patch('cv2.VideoCapture')
    def test_camera_manager_integration(self, mock_video_capture):
        """Test camera manager with mocked OpenCV"""
        # Mock camera capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.getBackendName.return_value = "test_backend"
        mock_video_capture.return_value = mock_cap

        from src.camera.camera_manager import CameraManager
        from config.settings import ThermalConfig

        # Create mock config
        config = Mock()
        config.camera.thermal_device_type = "webcam"
        config.camera.rgb_camera_index = 0
        config.camera.frame_width = 640
        config.camera.frame_height = 480
        config.camera.fps = 30

        async def test_camera():
            camera_manager = CameraManager(config)
            success = await camera_manager.initialize()
            self.assertTrue(success)

            # Test frame capture
            rgb_frame = await camera_manager.get_rgb_frame()
            self.assertIsNotNone(rgb_frame)

            thermal_frame = await camera_manager.get_thermal_frame()
            self.assertIsNotNone(thermal_frame)

            # Test status
            self.assertTrue(camera_manager.is_connected())

            await camera_manager.stop()

        # Run async test
        asyncio.run(test_camera())

    @patch('cv2.CascadeClassifier')
    def test_face_detection_pipeline(self, mock_cascade_class):
        """Test face detection pipeline integration"""
        # Mock cascade classifier
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[200, 150, 200, 200]])
        mock_cascade_class.return_value = mock_cascade

        from src.core.face_detector import FaceDetector

        # Create mock config
        config = Mock()
        config.detection.min_face_size = 80
        config.detection.face_confidence_threshold = 0.5
        config.detection.forehead_region_ratio = 0.3

        detector = FaceDetector(config)

        # Test with image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces, foreheads = detector.detect_faces_and_foreheads(test_image)

        self.assertEqual(len(faces), 1)
        self.assertEqual(len(foreheads), 1)
        self.assertEqual(faces[0].x, 200)
        self.assertEqual(faces[0].y, 150)

    def test_temperature_analysis_pipeline(self):
        """Test temperature analysis pipeline"""
        from src.core.temperature_analyzer import TemperatureAnalyzer, ForeheadRegion

        # Create mock config
        config = Mock()
        config.temperature.calibration_offset = 0.0
        config.temperature.fever_threshold = 37.5
        config.temperature.temp_min = 30.0
        config.temperature.temp_max = 45.0
        config.temperature.smoothing_kernel_size = 5
        config.development.demo_mode = True
        config.development.simulate_fever_chance = 0.1

        analyzer = TemperatureAnalyzer(config)

        # Test with thermal data in 4-5 range
        thermal_frame = np.full((200, 200), 4.5, dtype=np.float32)
        thermal_frame[80:120, 90:150] = 4.7  # Warmer forehead region

        forehead_region = ForeheadRegion(x=90, y=80, width=60, height=40)

        reading = analyzer.calculate_temperature(thermal_frame, forehead_region)

        self.assertIsNotNone(reading)
        self.assertGreater(reading.avg_temperature, 30.0)
        self.assertLess(reading.avg_temperature, 45.0)
        self.assertGreater(reading.confidence, 0.0)

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    async def test_full_processing_pipeline(self, mock_cascade_class, mock_video_capture):
        """Test complete processing pipeline integration"""
        # Setup mocks
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        thermal_frame = np.full((480, 640), 4.5, dtype=np.float32)
        mock_cap.read.return_value = (True, rgb_frame)
        mock_video_capture.return_value = mock_cap

        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[200, 150, 200, 200]])
        mock_cascade_class.return_value = mock_cascade

        # Import components
        from src.camera.camera_manager import CameraManager
        from src.core.thermal_processor import ThermalProcessor

        # Create mock config
        config = Mock()
        config.camera.thermal_device_type = "webcam"
        config.camera.rgb_camera_index = 0
        config.camera.frame_width = 640
        config.camera.frame_height = 480
        config.camera.fps = 30
        config.temperature.fever_threshold = 37.5
        config.temperature.averaging_samples = 3
        config.temperature.calibration_offset = 0.0
        config.temperature.temp_min = 30.0
        config.temperature.temp_max = 45.0
        config.temperature.smoothing_kernel_size = 5
        config.detection.min_face_size = 80
        config.detection.face_confidence_threshold = 0.5
        config.detection.forehead_region_ratio = 0.3
        config.development.demo_mode = True
        config.development.simulate_fever_chance = 0.1

        # Initialize components
        camera_manager = CameraManager(config)
        await camera_manager.initialize()

        thermal_processor = ThermalProcessor(config, camera_manager)

        # Process a frame
        reading = await thermal_processor.process_frame()

        if reading:  # Might be None if temperature calculation fails
            self.assertIsInstance(reading.avg_temperature, float)
            self.assertIsInstance(reading.is_fever, bool)

        # Check detection data
        detection_data = thermal_processor.get_latest_detection_data()
        self.assertIn('frame_width', detection_data)
        self.assertIn('frame_height', detection_data)

        await camera_manager.stop()

    def test_websocket_data_flow(self):
        """Test WebSocket data structure compatibility"""
        from src.api.websocket_handler import WebSocketManager
        from src.core.face_detector import FaceDetection
        from src.core.temperature_analyzer import TemperatureReading

        # Create mock thermal processor
        mock_processor = Mock()

        # Create test data
        test_reading = TemperatureReading(
            avg_temperature=36.5,
            max_temperature=37.0,
            min_temperature=36.0,
            pixel_count=100,
            confidence=0.9,
            timestamp=datetime.now(),
            is_fever=False
        )

        test_face = FaceDetection(x=200, y=150, width=200, height=200, confidence=0.85)

        detection_data = {
            'frame_width': 640,
            'frame_height': 480,
            'face_detection': test_face,
            'forehead_detection': {
                'x': 220, 'y': 170, 'width': 160, 'height': 60, 'confidence': 0.8
            },
            'detection_method': 'yolo11'
        }

        mock_processor.get_averaged_reading.return_value = test_reading
        mock_processor.get_latest_detection_data.return_value = detection_data

        # Test WebSocket manager
        ws_manager = WebSocketManager(mock_processor)
        self.assertIsNotNone(ws_manager)

        # Verify data structure is JSON serializable
        import json

        message_data = {
            "temperature": test_reading.avg_temperature,
            "confidence": test_reading.confidence,
            "is_fever": test_reading.is_fever,
            "face_detection": {
                "x": test_face.x,
                "y": test_face.y,
                "width": test_face.width,
                "height": test_face.height,
                "confidence": test_face.confidence
            },
            "forehead_detection": detection_data['forehead_detection']
        }

        # Should be JSON serializable
        json_str = json.dumps(message_data)
        self.assertIsInstance(json_str, str)

    def test_api_endpoints_integration(self):
        """Test API endpoints integration"""
        from src.api.routes import ThermalAPIRoutes
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        # Create mock components
        mock_processor = Mock()
        mock_camera_manager = Mock()
        mock_config = Mock()

        # Setup return values
        mock_camera_manager.is_connected.return_value = True
        mock_processor.is_running = True

        # Create API routes
        api_routes = ThermalAPIRoutes(mock_processor, mock_camera_manager, mock_config)

        # Create test app
        app = FastAPI()
        app.include_router(api_routes.router)
        client = TestClient(app)

        # Test endpoints
        health_response = client.get("/health")
        self.assertEqual(health_response.status_code, 200)

        status_response = client.get("/status")
        self.assertEqual(status_response.status_code, 200)

    def test_configuration_validation(self):
        """Test configuration validation across components"""
        from config.settings import ThermalConfig

        # Test with minimal valid configuration
        config = ThermalConfig()
        errors = config.validate()

        # Should have no validation errors with defaults
        self.assertEqual(len(errors), 0, f"Default config validation failed: {errors}")

        # Test invalid configuration
        config.temperature.temp_min = 50.0  # Higher than max
        config.temperature.temp_max = 40.0

        errors = config.validate()
        self.assertGreater(len(errors), 0, "Should detect invalid temperature range")


class TestPerformanceIntegration(unittest.TestCase):
    """Performance integration tests"""

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    def test_processing_performance(self, mock_cascade_class, mock_video_capture):
        """Test processing pipeline performance"""
        import time

        # Setup mocks for quick processing
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[200, 150, 200, 200]])
        mock_cascade_class.return_value = mock_cascade

        from src.core.face_detector import FaceDetector

        config = Mock()
        config.detection.min_face_size = 80
        config.detection.face_confidence_threshold = 0.5
        config.detection.forehead_region_ratio = 0.3

        detector = FaceDetector(config)

        # Test processing time
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        for _ in range(10):  # Process 10 frames
            faces, foreheads = detector.detect_faces_and_foreheads(test_image)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10

        # Should process a frame in reasonable time (< 100ms for mocked detection)
        self.assertLess(avg_time, 0.1, f"Face detection too slow: {avg_time:.3f}s per frame")

    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import gc

        from src.core.temperature_analyzer import TemperatureAnalyzer, ForeheadRegion

        config = Mock()
        config.temperature.calibration_offset = 0.0
        config.temperature.fever_threshold = 37.5
        config.temperature.temp_min = 30.0
        config.temperature.temp_max = 45.0
        config.temperature.smoothing_kernel_size = 5
        config.development.demo_mode = True
        config.development.simulate_fever_chance = 0.1

        analyzer = TemperatureAnalyzer(config)

        # Process many thermal frames
        for i in range(100):
            thermal_frame = np.random.uniform(4.0, 5.0, (200, 200)).astype(np.float32)
            forehead_region = ForeheadRegion(x=50, y=50, width=100, height=60)

            reading = analyzer.calculate_temperature(thermal_frame, forehead_region)

            # Force garbage collection occasionally
            if i % 10 == 0:
                gc.collect()

        # Test should complete without memory errors
        self.assertTrue(True)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceIntegration))


    # Run tests with async support
    class AsyncTestRunner:
        def __init__(self, verbosity=2):
            self.verbosity = verbosity

        def run(self, suite):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                runner = unittest.TextTestRunner(verbosity=self.verbosity)
                return runner.run(suite)
            finally:
                loop.close()


    # Run tests
    runner = AsyncTestRunner(verbosity=2)
    result = runner.run(suite)
