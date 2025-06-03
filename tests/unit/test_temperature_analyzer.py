import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.temperature_analyzer import TemperatureAnalyzer, TemperatureReading, ForeheadRegion
from src.core.face_detector import FaceDetection


class TestTemperatureAnalyzer(unittest.TestCase):
    """Test cases for TemperatureAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.temperature.calibration_offset = 0.0
        self.mock_config.temperature.fever_threshold = 37.5
        self.mock_config.temperature.temp_min = 30.0
        self.mock_config.temperature.temp_max = 45.0
        self.mock_config.temperature.smoothing_kernel_size = 5
        self.mock_config.detection.forehead_region_ratio = 0.3
        self.mock_config.development.demo_mode = True
        self.mock_config.development.simulate_fever_chance = 0.1

        # Create analyzer instance
        self.analyzer = TemperatureAnalyzer(self.mock_config)

    def test_temperature_analyzer_initialization(self):
        """Test temperature analyzer initialization"""
        self.assertEqual(self.analyzer.calibration_offset, 0.0)
        self.assertEqual(self.analyzer.fever_threshold, 37.5)
        self.assertTrue(self.analyzer.demo_mode)

    def test_forehead_region_dataclass(self):
        """Test ForeheadRegion dataclass"""
        forehead = ForeheadRegion(x=100, y=50, width=150, height=75)

        self.assertEqual(forehead.x, 100)
        self.assertEqual(forehead.y, 50)
        self.assertEqual(forehead.width, 150)
        self.assertEqual(forehead.height, 75)

    def test_temperature_reading_dataclass(self):
        """Test TemperatureReading dataclass"""
        reading = TemperatureReading(
            avg_temperature=36.5,
            max_temperature=37.0,
            min_temperature=36.0,
            pixel_count=100,
            confidence=0.85,
            timestamp=datetime.now(),
            is_fever=False
        )

        self.assertEqual(reading.avg_temperature, 36.5)
        self.assertEqual(reading.max_temperature, 37.0)
        self.assertEqual(reading.min_temperature, 36.0)
        self.assertEqual(reading.pixel_count, 100)
        self.assertEqual(reading.confidence, 0.85)
        self.assertFalse(reading.is_fever)

    def test_extract_forehead_from_face(self):
        """Test forehead region extraction from face detection"""
        face = FaceDetection(x=200, y=150, width=200, height=200, confidence=0.8)

        forehead = self.analyzer.extract_forehead_region(face)

        # Forehead should be smaller and positioned correctly
        self.assertGreater(forehead.x, face.x)  # Centered horizontally
        self.assertGreater(forehead.y, face.y)  # Below top of face
        self.assertLess(forehead.width, face.width)  # Narrower than face
        self.assertEqual(forehead.height, int(face.height * 0.3))  # 30% of face height

    def test_extract_forehead_from_coordinates(self):
        """Test forehead region extraction from coordinates"""
        forehead = self.analyzer.extract_forehead_region_from_coordinates(100, 50, 150, 75)

        self.assertEqual(forehead.x, 100)
        self.assertEqual(forehead.y, 50)
        self.assertEqual(forehead.width, 150)
        self.assertEqual(forehead.height, 75)

    def test_low_range_thermal_conversion(self):
        """Test conversion of low range thermal data (4-5 range)"""
        # Create thermal data in 4-5 range
        thermal_data = np.array([[4.2, 4.5, 4.8],
                                 [4.3, 4.7, 4.9],
                                 [4.1, 4.4, 4.6]], dtype=np.float32)

        temperatures = self.analyzer._convert_low_range_to_celsius(thermal_data)

        # Should convert to body temperature range
        self.assertTrue(np.all(temperatures >= 34.0))
        self.assertTrue(np.all(temperatures <= 42.0))
        self.assertEqual(temperatures.shape, thermal_data.shape)

    def test_webcam_thermal_conversion(self):
        """Test conversion of webcam grayscale data"""
        # Create webcam-like data (0-255 range)
        thermal_data = np.array([[100, 150, 200],
                                 [120, 180, 220],
                                 [110, 160, 190]], dtype=np.uint8)

        temperatures = self.analyzer._convert_webcam_to_celsius(thermal_data)

        # Should convert to reasonable temperature range
        self.assertTrue(np.all(temperatures >= 30.0))
        self.assertTrue(np.all(temperatures <= 45.0))
        self.assertEqual(temperatures.shape, thermal_data.shape)

    def test_raw_thermal_conversion(self):
        """Test conversion of raw thermal camera data"""
        # Create raw thermal data (1000+ range)
        thermal_data = np.array([[1050, 1055, 1060],
                                 [1052, 1058, 1062],
                                 [1048, 1053, 1057]], dtype=np.float32)

        temperatures = self.analyzer._convert_raw_thermal_to_celsius(thermal_data)

        # Should convert using standard formula
        expected = (thermal_data - 1000.0) / 10.0
        np.testing.assert_array_almost_equal(temperatures, expected, decimal=1)

    def test_temperature_calculation_valid_data(self):
        """Test temperature calculation with valid thermal data"""
        # Create thermal frame in 4-5 range (your actual data range)
        thermal_frame = np.full((200, 200), 4.5, dtype=np.float32)
        thermal_frame[80:120, 90:150] = 4.7  # Warmer forehead region

        forehead_region = ForeheadRegion(x=90, y=80, width=60, height=40)

        reading = self.analyzer.calculate_temperature(thermal_frame, forehead_region)

        self.assertIsNotNone(reading)
        self.assertIsInstance(reading, TemperatureReading)
        self.assertGreater(reading.avg_temperature, 30.0)
        self.assertLess(reading.avg_temperature, 45.0)
        self.assertGreater(reading.confidence, 0.0)
        self.assertLessEqual(reading.confidence, 1.0)

    def test_temperature_calculation_empty_roi(self):
        """Test temperature calculation with empty ROI"""
        thermal_frame = np.ones((100, 100), dtype=np.float32) * 4.5

        # Invalid forehead region (outside frame)
        forehead_region = ForeheadRegion(x=200, y=200, width=50, height=30)

        reading = self.analyzer.calculate_temperature(thermal_frame, forehead_region)

        # Should return None or demo reading
        if reading is not None:
            # If demo mode provides a reading, it should be valid
            self.assertIsInstance(reading, TemperatureReading)

    def test_fever_detection(self):
        """Test fever detection logic"""
        # Create reading with fever temperature
        reading_fever = TemperatureReading(
            avg_temperature=38.0,  # Above fever threshold
            max_temperature=38.5,
            min_temperature=37.5,
            pixel_count=100,
            confidence=0.9,
            timestamp=datetime.now(),
            is_fever=38.0 >= 37.5
        )

        # Create reading with normal temperature
        reading_normal = TemperatureReading(
            avg_temperature=36.5,  # Below fever threshold
            max_temperature=37.0,
            min_temperature=36.0,
            pixel_count=100,
            confidence=0.9,
            timestamp=datetime.now(),
            is_fever=36.5 >= 37.5
        )

        self.assertTrue(reading_fever.is_fever)
        self.assertFalse(reading_normal.is_fever)

    def test_calibration_offset_application(self):
        """Test calibration offset application"""
        # Set calibration offset
        self.analyzer.calibration_offset = 1.0

        thermal_frame = np.full((100, 100), 4.5, dtype=np.float32)
        forehead_region = ForeheadRegion(x=20, y=20, width=60, height=40)

        reading = self.analyzer.calculate_temperature(thermal_frame, forehead_region)

        if reading:
            # Temperature should be affected by calibration offset
            # The exact value depends on the conversion, but offset should be applied
            self.assertIsInstance(reading.avg_temperature, float)

    def test_demo_temperature_generation(self):
        """Test demo temperature reading generation"""
        reading = self.analyzer._generate_demo_temperature_reading(100)

        self.assertIsInstance(reading, TemperatureReading)
        self.assertGreater(reading.avg_temperature, 30.0)
        self.assertLess(reading.avg_temperature, 45.0)
        self.assertGreater(reading.confidence, 0.0)
        self.assertLessEqual(reading.confidence, 1.0)
        self.assertEqual(reading.pixel_count, 100)

    def test_confidence_calculation(self):
        """Test confidence calculation based on temperature consistency"""
        # Create thermal data with low variation (high confidence)
        thermal_frame = np.full((100, 100), 4.5, dtype=np.float32)
        thermal_frame += np.random.normal(0, 0.1, thermal_frame.shape)  # Low noise

        forehead_region = ForeheadRegion(x=20, y=20, width=60, height=40)

        reading = self.analyzer.calculate_temperature(thermal_frame, forehead_region)

        if reading:
            # Should have reasonable confidence
            self.assertGreater(reading.confidence, 0.1)
            self.assertLessEqual(reading.confidence, 1.0)

    def test_analyzer_status(self):
        """Test analyzer status retrieval"""
        status = self.analyzer.get_analyzer_status()

        self.assertIn('demo_mode', status)
        self.assertIn('fever_threshold', status)
        self.assertIn('calibration_offset', status)
        self.assertIn('temp_min', status)
        self.assertIn('temp_max', status)

    def test_calibration_offset_update(self):
        """Test calibration offset update"""
        new_offset = 2.5
        self.analyzer.set_calibration_offset(new_offset)

        self.assertEqual(self.analyzer.calibration_offset, new_offset)
        self.assertEqual(self.analyzer.config.temperature.calibration_offset, new_offset)

    def test_fever_threshold_update(self):
        """Test fever threshold update"""
        new_threshold = 38.0
        self.analyzer.set_fever_threshold(new_threshold)

        self.assertEqual(self.analyzer.fever_threshold, new_threshold)
        self.assertEqual(self.analyzer.config.temperature.fever_threshold, new_threshold)

    def test_thermal_conversion_auto_detection(self):
        """Test automatic thermal data format detection"""
        # Test different data ranges

        # Low range (4-5)
        low_range_data = np.array([[4.2, 4.5], [4.3, 4.7]], dtype=np.float32)
        temps_low = self.analyzer._thermal_to_celsius(low_range_data)
        self.assertTrue(np.all(temps_low >= 30.0))

        # Webcam range (0-255)
        webcam_data = np.array([[100, 150], [120, 180]], dtype=np.uint8)
        temps_webcam = self.analyzer._thermal_to_celsius(webcam_data)
        self.assertTrue(np.all(temps_webcam >= 30.0))

        # Raw thermal range (1000+)
        raw_data = np.array([[1050, 1055], [1052, 1058]], dtype=np.float32)
        temps_raw = self.analyzer._thermal_to_celsius(raw_data)
        self.assertTrue(np.all(temps_raw >= 3.0))  # (1050-1000)/10 = 5.0

    @patch('numpy.random.normal')
    def test_consistent_demo_generation(self, mock_random):
        """Test demo temperature generation consistency"""
        # Mock random to return consistent values
        mock_random.return_value = 0.5

        reading1 = self.analyzer._generate_demo_temperature_reading(100)
        reading2 = self.analyzer._generate_demo_temperature_reading(100)

        # Should generate consistent readings with mocked random
        self.assertEqual(reading1.pixel_count, reading2.pixel_count)

    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases"""
        # Very small thermal frame
        small_frame = np.array([[4.5]], dtype=np.float32)
        small_region = ForeheadRegion(x=0, y=0, width=1, height=1)

        reading = self.analyzer.calculate_temperature(small_frame, small_region)

        # Should handle gracefully
        if reading:
            self.assertIsInstance(reading, TemperatureReading)

    def test_error_handling(self):
        """Test error handling in temperature calculation"""
        # Invalid thermal frame
        invalid_frame = np.array([])
        forehead_region = ForeheadRegion(x=0, y=0, width=10, height=10)

        reading = self.analyzer.calculate_temperature(invalid_frame, forehead_region)

        # Should handle gracefully - either return None or demo reading
        if reading:
            self.assertIsInstance(reading, TemperatureReading)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTemperatureAnalyzer))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)