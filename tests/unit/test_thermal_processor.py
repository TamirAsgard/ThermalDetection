import unittest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.thermal_processor import ThermalProcessor
from src.core.face_detector import FaceDetection, ForeheadDetection
from src.core.temperature_analyzer import TemperatureReading


class TestThermalProcessor(unittest.TestCase):
    """Test cases for ThermalProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.temperature.fever_threshold = 37.5
        self.mock_config.temperature.averaging_samples = 5

        # Create mock camera manager
        self.mock_camera_manager = Mock()
        self.mock_camera_manager.get_rgb_frame = AsyncMock()
        self.mock_camera_manager.get_thermal_frame = AsyncMock()
        self.mock_camera_manager.get_frame_statistics = Mock()

        # Create mock face detector
        self.mock_face_detector = Mock()
        self.mock_face_detector.detection_method = "yolo11"
        self.mock_face_detector.detect_faces_and_foreheads = Mock()
        self.mock_face_detector.get_best_face_and_forehead = Mock()
        self.mock_face_detector.get_detection_info = Mock()

        # Create mock temperature analyzer
        self.mock_temperature_analyzer = Mock()
        self.mock_temperature_analyzer.extract_forehead_region = Mock()
        self.mock_temperature_analyzer.extract_forehead_region_from_coordinates = Mock()
        self.mock_temperature_analyzer.calculate_temperature = Mock()

        # Create test frames
        self.test_rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_thermal_frame = np.ones((480, 640), dtype=np.float32) * 4.5

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_thermal_processor_initialization(self, mock_temp_analyzer, mock_face_detector):
        """Test thermal processor initialization"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        self.assertFalse(processor.is_running)
        self.assertIsNone(processor.last_reading)
        self.assertEqual(len(processor.reading_history), 0)
        self.assertEqual(processor.fever_threshold, 37.5)
        self.assertEqual(processor.frame_count, 0)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    async def test_process_frame_successful(self, mock_temp_analyzer, mock_face_detector):
        """Test successful frame processing"""
        # Setup mocks
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Mock camera frames
        self.mock_camera_manager.get_rgb_frame.return_value = self.test_rgb_frame
        self.mock_camera_manager.get_thermal_frame.return_value = self.test_thermal_frame

        # Mock face detection
        test_face = FaceDetection(x=200, y=150, width=200, height=200, confidence=0.85)
        test_forehead = ForeheadDetection(x=220, y=170, width=160, height=60, confidence=0.8)

        self.mock_face_detector.detect_faces_and_foreheads.return_value = ([test_face], [test_forehead])
        self.mock_face_detector.get_best_face_and_forehead.return_value = (test_face, test_forehead)

        # Mock temperature calculation
        test_reading = TemperatureReading(
            avg_temperature=36.5,
            max_temperature=37.0,
            min_temperature=36.0,
            pixel_count=100,
            confidence=0.9,
            timestamp=datetime.now(),
            is_fever=False
        )
        self.mock_temperature_analyzer.calculate_temperature.return_value = test_reading

        # Process frame
        result = await processor.process_frame()

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.avg_temperature, 36.5)
        self.assertFalse(result.is_fever)

        # Verify detection data was stored
        detection_data = processor.get_latest_detection_data()
        self.assertIsNotNone(detection_data['face_detection'])
        self.assertIsNotNone(detection_data['forehead_detection'])

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    async def test_process_frame_no_faces(self, mock_temp_analyzer, mock_face_detector):
        """Test frame processing when no faces detected"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Mock camera frames
        self.mock_camera_manager.get_rgb_frame.return_value = self.test_rgb_frame
        self.mock_camera_manager.get_thermal_frame.return_value = self.test_thermal_frame

        # Mock no face detection
        self.mock_face_detector.detect_faces_and_foreheads.return_value = ([], [])

        # Process frame
        result = await processor.process_frame()

        # Should return None when no faces detected
        self.assertIsNone(result)

        # Detection data should still be populated with basic info
        detection_data = processor.get_latest_detection_data()
        self.assertIsNone(detection_data['face_detection'])
        self.assertIsNone(detection_data['forehead_detection'])

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    async def test_process_frame_no_camera_frames(self, mock_temp_analyzer, mock_face_detector):
        """Test frame processing when camera frames unavailable"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Mock camera returning None
        self.mock_camera_manager.get_rgb_frame.return_value = None
        self.mock_camera_manager.get_thermal_frame.return_value = None

        # Process frame
        result = await processor.process_frame()

        # Should return None when no frames available
        self.assertIsNone(result)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_reading_history_management(self, mock_temp_analyzer, mock_face_detector):
        """Test reading history management and averaging"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Create test readings
        readings = []
        for i in range(7):  # More than averaging_samples (5)
            reading = TemperatureReading(
                avg_temperature=36.0 + i * 0.1,
                max_temperature=37.0,
                min_temperature=36.0,
                pixel_count=100,
                confidence=0.9,
                timestamp=datetime.now(),
                is_fever=False
            )
            readings.append(reading)
            processor._update_history(reading)

        # Should only keep last 5 readings
        self.assertEqual(len(processor.reading_history), 5)

        # Get averaged reading
        averaged = processor.get_averaged_reading()
        self.assertIsNotNone(averaged)

        # Should average the last 5 temperatures
        expected_avg = sum(r.avg_temperature for r in readings[-5:]) / 5
        self.assertAlmostEqual(averaged.avg_temperature, expected_avg, places=2)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_fever_detection_in_averaging(self, mock_temp_analyzer, mock_face_detector):
        """Test fever detection in averaged readings"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Add readings that average to fever temperature
        fever_readings = [
            TemperatureReading(37.6, 38.0, 37.0, 100, 0.9, datetime.now(), True),
            TemperatureReading(37.8, 38.2, 37.4, 100, 0.9, datetime.now(), True),
            TemperatureReading(37.7, 38.1, 37.3, 100, 0.9, datetime.now(), True)
        ]

        for reading in fever_readings:
            processor._update_history(reading)

        averaged = processor.get_averaged_reading()
        self.assertIsNotNone(averaged)
        self.assertTrue(averaged.is_fever)
        self.assertGreater(averaged.avg_temperature, processor.fever_threshold)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_processing_statistics(self, mock_temp_analyzer, mock_face_detector):
        """Test processing statistics collection"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Simulate some processing
        processor.frame_count = 100
        processor.detection_count = 75

        stats = processor.get_processing_statistics()

        self.assertEqual(stats['total_frames_processed'], 100)
        self.assertEqual(stats['total_detections'], 75)
        self.assertEqual(stats['detection_rate'], 0.75)
        self.assertIn('processing_fps', stats)
        self.assertIn('uptime_seconds', stats)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_configuration_updates(self, mock_temp_analyzer, mock_face_detector):
        """Test configuration updates"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Test configuration update
        updates = {
            'fever_threshold': 38.0,
            'averaging_samples': 10,
            'confidence_threshold': 0.8
        }

        result = processor.update_configuration(updates)

        self.assertTrue(result)
        self.assertEqual(processor.fever_threshold, 38.0)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_current_status(self, mock_temp_analyzer, mock_face_detector):
        """Test current status retrieval"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Add a test reading
        test_reading = TemperatureReading(36.5, 37.0, 36.0, 100, 0.9, datetime.now(), False)
        processor.last_reading = test_reading
        processor.latest_detection_data = {'face_detection': Mock()}

        status = processor.get_current_status()

        self.assertIn('is_running', status)
        self.assertIn('frame_count', status)
        self.assertIn('has_recent_reading', status)
        self.assertIn('face_detected', status)
        self.assertTrue(status['has_recent_reading'])
        self.assertTrue(status['face_detected'])

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_reset_statistics(self, mock_temp_analyzer, mock_face_detector):
        """Test statistics reset"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Simulate some activity
        processor.frame_count = 50
        processor.detection_count = 30
        processor.reading_history = [Mock(), Mock(), Mock()]
        processor.last_reading = Mock()

        # Reset statistics
        processor.reset_statistics()

        self.assertEqual(processor.frame_count, 0)
        self.assertEqual(processor.detection_count, 0)
        self.assertEqual(len(processor.reading_history), 0)
        self.assertIsNone(processor.last_reading)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    async def test_start_stop_processing(self, mock_temp_analyzer, mock_face_detector):
        """Test start and stop processing"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Mock process_frame to return quickly
        original_process_frame = processor.process_frame

        async def mock_process_frame():
            processor.frame_count += 1
            await asyncio.sleep(0.001)  # Very short sleep
            return None

        processor.process_frame = mock_process_frame

        # Start processing
        self.assertFalse(processor.is_running)

        # Run for a short time
        start_task = asyncio.create_task(processor.start_processing())
        await asyncio.sleep(0.1)  # Let it run briefly

        self.assertTrue(processor.is_running)
        self.assertGreater(processor.frame_count, 0)

        # Stop processing
        processor.stop_processing()
        start_task.cancel()

        self.assertFalse(processor.is_running)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_empty_reading_history(self, mock_temp_analyzer, mock_face_detector):
        """Test behavior with empty reading history"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Test with empty history
        averaged = processor.get_averaged_reading()
        self.assertIsNone(averaged)  # Should return None when no readings

        # Test with single reading
        single_reading = TemperatureReading(36.5, 37.0, 36.0, 100, 0.9, datetime.now(), False)
        processor._update_history(single_reading)

        averaged = processor.get_averaged_reading()
        self.assertIsNotNone(averaged)
        self.assertEqual(averaged.avg_temperature, single_reading.avg_temperature)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    def test_latest_detection_data_fallback(self, mock_temp_analyzer, mock_face_detector):
        """Test latest detection data with fallback values"""
        mock_face_detector.return_value = self.mock_face_detector
        mock_temp_analyzer.return_value = self.mock_temperature_analyzer

        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Test with empty detection data
        detection_data = processor.get_latest_detection_data()

        self.assertIsInstance(detection_data, dict)
        self.assertIn('frame_width', detection_data)
        self.assertIn('frame_height', detection_data)
        self.assertIn('face_detection', detection_data)
        self.assertIn('forehead_detection', detection_data)


class TestThermalProcessorIntegration(unittest.TestCase):
    """Integration tests for thermal processor"""

    def setUp(self):
        """Set up integration test fixtures"""
        # Create more realistic mock config
        self.mock_config = Mock()
        self.mock_config.temperature.fever_threshold = 37.5
        self.mock_config.temperature.averaging_samples = 3

        # Create mock camera manager with realistic behavior
        self.mock_camera_manager = Mock()

        # Realistic test frames
        self.rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.thermal_frame = np.random.uniform(4.2, 4.8, (480, 640)).astype(np.float32)

    @patch('src.core.thermal_processor.FaceDetector')
    @patch('src.core.thermal_processor.TemperatureAnalyzer')
    async def test_end_to_end_processing(self, mock_temp_analyzer_class, mock_face_detector_class):
        """Test end-to-end processing workflow"""
        # Create mock instances
        mock_face_detector = Mock()
        mock_temp_analyzer = Mock()
        mock_face_detector_class.return_value = mock_face_detector
        mock_temp_analyzer_class.return_value = mock_temp_analyzer

        # Setup camera manager
        self.mock_camera_manager.get_rgb_frame = AsyncMock(return_value=self.rgb_frame)
        self.mock_camera_manager.get_thermal_frame = AsyncMock(return_value=self.thermal_frame)

        # Setup face detector
        test_face = FaceDetection(x=200, y=150, width=200, height=200, confidence=0.85)
        test_forehead = ForeheadDetection(x=220, y=170, width=160, height=60, confidence=0.8)
        mock_face_detector.detect_faces_and_foreheads.return_value = ([test_face], [test_forehead])
        mock_face_detector.get_best_face_and_forehead.return_value = (test_face, test_forehead)
        mock_face_detector.detection_method = "yolo11"

        # Setup temperature analyzer
        test_reading = TemperatureReading(
            avg_temperature=36.8,
            max_temperature=37.2,
            min_temperature=36.4,
            pixel_count=150,
            confidence=0.92,
            timestamp=datetime.now(),
            is_fever=False
        )
        mock_temp_analyzer.calculate_temperature.return_value = test_reading

        # Create processor and test
        processor = ThermalProcessor(self.mock_config, self.mock_camera_manager)

        # Process multiple frames
        for i in range(5):
            result = await processor.process_frame()
            if result:
                processor.last_reading = result
                processor._update_history(result)

        # Verify results
        self.assertIsNotNone(processor.last_reading)
        self.assertGreater(len(processor.reading_history), 0)

        # Test averaged reading
        averaged = processor.get_averaged_reading()
        self.assertIsNotNone(averaged)
        self.assertAlmostEqual(averaged.avg_temperature, 36.8, places=1)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestThermalProcessor))

    # Run tests with asyncio support
    import asyncio


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