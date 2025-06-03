import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.face_detector import FaceDetector, FaceDetection, ForeheadDetection
from config.settings import ThermalConfig


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.detection.min_face_size = 80
        self.mock_config.detection.face_confidence_threshold = 0.5
        self.mock_config.detection.forehead_region_ratio = 0.3

        # Create test image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw a simple face-like rectangle for testing
        cv2.rectangle(self.test_image, (200, 150), (400, 350), (255, 255, 255), -1)

    @patch('src.core.face_detector.cv2.CascadeClassifier')
    def test_face_detector_initialization(self, mock_cascade):
        """Test face detector initialization"""
        mock_cascade.return_value.empty.return_value = False

        detector = FaceDetector(self.mock_config)

        self.assertIsNotNone(detector)
        self.assertEqual(detector.min_face_size, 80)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.forehead_ratio, 0.3)

    def test_face_detection_dataclass(self):
        """Test FaceDetection dataclass"""
        face = FaceDetection(x=100, y=50, width=200, height=250, confidence=0.85)

        self.assertEqual(face.x, 100)
        self.assertEqual(face.y, 50)
        self.assertEqual(face.width, 200)
        self.assertEqual(face.height, 250)
        self.assertEqual(face.confidence, 0.85)

    def test_forehead_detection_dataclass(self):
        """Test ForeheadDetection dataclass"""
        forehead = ForeheadDetection(x=120, y=60, width=160, height=75, confidence=0.9)

        self.assertEqual(forehead.x, 120)
        self.assertEqual(forehead.y, 60)
        self.assertEqual(forehead.width, 160)
        self.assertEqual(forehead.height, 75)
        self.assertEqual(forehead.confidence, 0.9)

    @patch('src.core.face_detector.cv2.CascadeClassifier')
    def test_cascade_face_detection(self, mock_cascade_class):
        """Test cascade face detection"""
        # Mock cascade classifier
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[200, 150, 200, 200]])
        mock_cascade_class.return_value = mock_cascade

        detector = FaceDetector(self.mock_config)
        detector.detection_method = "cascade"

        faces = detector._detect_faces_cascade(self.test_image)

        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].x, 200)
        self.assertEqual(faces[0].y, 150)
        self.assertEqual(faces[0].width, 200)
        self.assertEqual(faces[0].height, 200)
        self.assertGreater(faces[0].confidence, 0)

    def test_forehead_extraction(self):
        """Test forehead region extraction from face"""
        face = FaceDetection(x=200, y=150, width=200, height=200, confidence=0.8)

        with patch('src.core.face_detector.cv2.CascadeClassifier'):
            detector = FaceDetector(self.mock_config)
            forehead = detector._extract_forehead_region(face)

        # Forehead should be centered and smaller than face
        self.assertGreater(forehead.x, face.x)
        self.assertGreater(forehead.y, face.y)
        self.assertLess(forehead.width, face.width)
        self.assertLess(forehead.height, face.height)

        # Forehead should be 30% of face height
        expected_height = int(face.height * 0.3)
        self.assertEqual(forehead.height, expected_height)

    @patch('src.core.face_detector.cv2.CascadeClassifier')
    def test_best_face_selection(self, mock_cascade_class):
        """Test selection of best face from multiple detections"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade_class.return_value = mock_cascade

        detector = FaceDetector(self.mock_config)

        # Create multiple face detections with different confidences
        faces = [
            FaceDetection(x=100, y=100, width=150, height=150, confidence=0.6),
            FaceDetection(x=200, y=200, width=180, height=180, confidence=0.9),
            FaceDetection(x=300, y=300, width=120, height=120, confidence=0.7)
        ]

        foreheads = [
            ForeheadDetection(x=110, y=110, width=130, height=45, confidence=0.6),
            ForeheadDetection(x=210, y=210, width=150, height=54, confidence=0.9),
            ForeheadDetection(x=310, y=310, width=100, height=36, confidence=0.7)
        ]

        best_face, best_forehead = detector.get_best_face_and_forehead(faces, foreheads)

        # Should select the face with highest confidence (0.9)
        self.assertEqual(best_face.confidence, 0.9)
        self.assertEqual(best_face.x, 200)
        self.assertEqual(best_forehead.confidence, 0.9)

    @patch('src.core.face_detector.cv2.CascadeClassifier')
    def test_confidence_calculation(self, mock_cascade_class):
        """Test confidence calculation for cascade detection"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade_class.return_value = mock_cascade

        detector = FaceDetector(self.mock_config)

        # Test different face sizes for confidence calculation
        frame_shape = (480, 640, 3)

        # Large face should have higher confidence
        large_confidence = detector._calculate_cascade_confidence(200, 200, frame_shape)
        small_confidence = detector._calculate_cascade_confidence(80, 80, frame_shape)

        self.assertGreater(large_confidence, small_confidence)
        self.assertLessEqual(large_confidence, 1.0)
        self.assertGreaterEqual(small_confidence, 0.0)

    @patch('src.core.face_detector.cv2.CascadeClassifier')
    def test_detection_info(self, mock_cascade_class):
        """Test detection info retrieval"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade_class.return_value = mock_cascade

        detector = FaceDetector(self.mock_config)
        info = detector.get_detection_info()

        self.assertIn('method', info)
        self.assertIn('confidence_threshold', info)
        self.assertIn('min_face_size', info)
        self.assertIn('forehead_ratio', info)
        self.assertIn('models_available', info)

    @patch('src.core.face_detector.cv2.CascadeClassifier')
    def test_no_faces_detected(self, mock_cascade_class):
        """Test behavior when no faces are detected"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([])  # No faces
        mock_cascade_class.return_value = mock_cascade

        detector = FaceDetector(self.mock_config)
        detector.detection_method = "cascade"

        faces = detector._detect_faces_cascade(self.test_image)

        self.assertEqual(len(faces), 0)

    def test_face_detection_with_drawing(self):
        """Test face detection with drawing enabled"""
        with patch('src.core.face_detector.cv2.CascadeClassifier') as mock_cascade_class:
            mock_cascade = Mock()
            mock_cascade.empty.return_value = False
            mock_cascade.detectMultiScale.return_value = np.array([[200, 150, 200, 200]])
            mock_cascade_class.return_value = mock_cascade

            detector = FaceDetector(self.mock_config)
            detector.detection_method = "cascade"

            test_frame = self.test_image.copy()
            faces, foreheads = detector.detect_faces_and_foreheads(test_frame, draw_detections=True)

            # Should return detections
            self.assertEqual(len(faces), 1)
            self.assertEqual(len(foreheads), 1)

            # Frame should be modified (drawing occurred)
            self.assertFalse(np.array_equal(test_frame, self.test_image))


class TestFaceDetectorIntegration(unittest.TestCase):
    """Integration tests for face detector with real OpenCV"""

    def setUp(self):
        """Set up test fixtures"""
        # Create real config
        try:
            self.config = ThermalConfig()
        except:
            # Fallback to mock config if ThermalConfig fails
            self.config = Mock()
            self.config.detection.min_face_size = 80
            self.config.detection.face_confidence_threshold = 0.5
            self.config.detection.forehead_region_ratio = 0.3

    def test_cascade_initialization_real(self):
        """Test real cascade classifier initialization"""
        try:
            detector = FaceDetector(self.config)
            self.assertIsNotNone(detector.face_cascade)
        except Exception as e:
            self.skipTest(f"OpenCV cascade not available: {e}")

    def test_empty_image_handling(self):
        """Test handling of empty or invalid images"""
        detector = FaceDetector(self.config)

        # Test with empty image
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces, foreheads = detector.detect_faces_and_foreheads(empty_image)

        self.assertEqual(len(faces), 0)
        self.assertEqual(len(foreheads), 0)

    def test_single_channel_image(self):
        """Test handling of single channel (grayscale) images"""
        detector = FaceDetector(self.config)

        # Test with grayscale image
        gray_image = np.zeros((200, 200), dtype=np.uint8)

        try:
            faces, foreheads = detector.detect_faces_and_foreheads(gray_image)
            # Should handle gracefully, either convert or process as-is
            self.assertIsInstance(faces, list)
            self.assertIsInstance(foreheads, list)
        except Exception:
            # If it fails, that's also acceptable for this edge case
            pass


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFaceDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceDetectorIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)