from pathlib import Path
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class FaceDetection:
    """Face detection result"""
    x: int
    y: int
    width: int
    height: int
    confidence: float

@dataclass
class ForeheadDetection:
    """Forehead detection result"""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetector:
    """Face detection using OpenCV"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize face detection models
        self.face_cascade = None
        self.net = None
        self.yunet_detector = None
        self.yolo_model = None
        self.detection_method = "yolo11"  # "cascade" or "yolo11"

        # Detection parameters
        self.min_face_size = getattr(config.detection, 'min_face_size', 80)
        self.confidence_threshold = getattr(config.detection, 'face_confidence_threshold', 0.5)
        self.forehead_ratio = getattr(config.detection, 'forehead_region_ratio', 0.3)
        self.nms_threshold = 0.4

        # Drawing colors (BGR format for OpenCV)
        self.face_color = (0, 255, 0)  # Green for face
        self.forehead_color = (0, 0, 255)  # Red for forehead
        self.text_color = (255, 255, 255)  # White for text
        self.bg_color = (0, 0, 0)  # Black for text background

        # Model paths
        self.model_dir = Path("model")
        self.model_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.detection_times = []
        self.avg_detection_time = 0.0

        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize the best available detector"""
        # Try YOLO11 first (your model)
        if self._try_initialize_yolo11():
            self.detection_method = "yolo11"
            self.logger.info("YOLO11 face detector initialized successfully")
            return

        # Fallback to Haar Cascade
        self._initialize_cascade_fallback()
        self.detection_method = "cascade"
        self.logger.info("Using Haar Cascade detector")

    def _try_initialize_yolo11(self) -> bool:
        """Try to initialize YOLO11 with your model"""
        try:
            yolo11_path = "../.." / self.model_dir / "yolov11n-face.pt"
            if not yolo11_path.exists():
                self.logger.info(f"YOLO11 model not found at {yolo11_path}")
                return False

            # Try to import and load YOLO
            from ultralytics import YOLO
            self.yolo_model = YOLO(str(yolo11_path))

            self.logger.info(f"YOLO11 model loaded from {yolo11_path}")
            return True

        except ImportError:
            self.logger.warning("ultralytics package not found. Install with: pip install ultralytics")
            return False
        except Exception as e:
            self.logger.warning(f"YOLO11 initialization failed: {e}")
            return False

    def _initialize_cascade_fallback(self):
        """Initialize Haar Cascade detector (guaranteed fallback)"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.logger.info(f"Loading Haar Cascade from: {cascade_path}")

            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                self.logger.error("Haar Cascade classifier is empty")
                return False

            self.logger.info("✅ Haar Cascade detector initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Haar Cascade initialization failed: {e}")
            return False

    def detect_faces_and_foreheads(self, frame: np.ndarray,
                                   draw_detections: bool = True) -> Tuple[List[FaceDetection], List[ForeheadDetection]]:
        """Detect faces and extract foreheads using the best available method"""

        # Detect faces using the available method
        if self.detection_method == "yolo11":
            faces = self._detect_faces_yolo11(frame)
        else:
            faces = self._detect_faces_cascade(frame)

        # Extract forehead regions
        foreheads = []
        for face in faces:
            forehead = self._extract_forehead_region(face)
            foreheads.append(forehead)

        # Draw detections if requested
        if draw_detections:
            self._draw_detections(frame, faces, foreheads)

        return faces, foreheads

    def _update_detection_stats(self, detection_time: float):
        """Update detection performance statistics"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 20:  # Keep last 20 measurements
            self.detection_times.pop(0)

        self.avg_detection_time = sum(self.detection_times) / len(self.detection_times)

    def _detect_faces_yolo11(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using YOLO11"""
        try:
            # Run YOLO11 inference
            results = self.yolo_model(frame, verbose=False, conf=self.confidence_threshold)

            detections = []

            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Get bounding box coordinates (xyxy format)
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())

                        # Convert to xywh format
                        x1, y1, x2, y2 = box
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)

                        # Filter by size and confidence
                        if (confidence >= self.confidence_threshold and
                                w >= self.min_face_size and h >= self.min_face_size):
                            detections.append(FaceDetection(
                                x=x, y=y, width=w, height=h, confidence=confidence
                            ))

            # Sort by confidence
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections[:10]  # Return top 10 detections

        except Exception as e:
            self.logger.error(f"YOLO11 detection failed: {e}")
            return []

    def _detect_faces_cascade(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using Haar Cascade fallback"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                maxSize=(400, 400)
            )

            detections = []
            for (x, y, w, h) in faces:
                confidence = self._calculate_cascade_confidence(w, h, frame.shape)
                detections.append(FaceDetection(
                    x=int(x), y=int(y), width=int(w), height=int(h), confidence=confidence
                ))

            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections[:3]

        except Exception as e:
            self.logger.error(f"Cascade detection failed: {e}")
            return []

    def _calculate_cascade_confidence(self, width: int, height: int, frame_shape: Tuple) -> float:
        """Calculate pseudo-confidence for cascade detection"""
        frame_height, frame_width = frame_shape[:2]
        face_area = width * height
        frame_area = frame_width * frame_height
        relative_size = face_area / frame_area

        if relative_size > 0.15:
            return 0.85
        elif relative_size > 0.08:
            return 0.75
        elif relative_size > 0.04:
            return 0.65
        else:
            return 0.55

    def _extract_forehead_region(self, face: FaceDetection) -> ForeheadDetection:
        """Extract forehead region from detected face"""
        # Forehead parameters (optimized for thermal measurement)
        forehead_height = int(face.height * self.forehead_ratio)
        forehead_y = face.y + int(face.height * 0.1)  # Start 10% down from top

        # Make forehead slightly narrower than face
        forehead_width = int(face.width * 0.8)  # 80% of face width
        forehead_x = face.x + int((face.width - forehead_width) / 2)

        # Forehead confidence derived from face confidence
        forehead_confidence = max(0.5, face.confidence - 0.1)

        return ForeheadDetection(
            x=forehead_x,
            y=forehead_y,
            width=forehead_width,
            height=forehead_height,
            confidence=forehead_confidence
        )

    def _draw_detections(self, frame: np.ndarray,
                         faces: List[FaceDetection],
                         foreheads: List[ForeheadDetection]):
        """Draw detection boxes and labels"""

        # Draw method indicator
        method_text = f"Method: {self.detection_method.upper()}"
        fps_text = f"FPS: {1.0 / max(self.avg_detection_time, 0.001):.1f}"

        cv2.putText(frame, method_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, fps_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw face detections
        for i, face in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(frame,
                          (face.x, face.y),
                          (face.x + face.width, face.y + face.height),
                          self.face_color, 2)

            # Draw confidence label
            face_text = f"Face: {face.confidence:.1%}"
            self._draw_label(frame, face_text, (face.x, face.y - 10), self.face_color)

            # Draw center point
            center_x = face.x + face.width // 2
            center_y = face.y + face.height // 2
            cv2.circle(frame, (center_x, center_y), 3, self.face_color, -1)

        # Draw forehead detections
        for i, forehead in enumerate(foreheads):
            # Draw forehead rectangle
            cv2.rectangle(frame,
                          (forehead.x, forehead.y),
                          (forehead.x + forehead.width, forehead.y + forehead.height),
                          self.forehead_color, 2)

            # Draw confidence label
            forehead_text = f"Forehead: {forehead.confidence:.1%}"
            self._draw_label(frame, forehead_text, (forehead.x, forehead.y - 10), self.forehead_color)

            # Draw center point
            center_x = forehead.x + forehead.width // 2
            center_y = forehead.y + forehead.height // 2
            cv2.circle(frame, (center_x, center_y), 2, self.forehead_color, -1)

    def _draw_label(self, frame: np.ndarray, text: str,
                    position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw text label with background"""
        x, y = position

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background
        cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width + 5, y + 5), self.bg_color, -1)

        # Draw text
        cv2.putText(frame, text, (x + 2, y - 2), font, font_scale, color, thickness)

    def get_best_face_and_forehead(self, faces: List[FaceDetection],
                                   foreheads: List[ForeheadDetection]) -> Tuple[
        Optional[FaceDetection], Optional[ForeheadDetection]]:
        """Get the highest confidence face and corresponding forehead"""
        if not faces or not foreheads:
            return None, None

        # Get highest confidence face
        best_face = max(faces, key=lambda f: f.confidence)
        best_face_index = faces.index(best_face)

        # Get corresponding forehead
        if best_face_index < len(foreheads):
            best_forehead = foreheads[best_face_index]
            return best_face, best_forehead

        return best_face, None

    def get_detection_info(self) -> dict:
        """Get information about the current detection method"""
        return {
            "method": self.detection_method,
            "confidence_threshold": self.confidence_threshold,
            "min_face_size": self.min_face_size,
            "forehead_ratio": self.forehead_ratio,
            "avg_detection_time": self.avg_detection_time,
            "fps": 1.0 / max(self.avg_detection_time, 0.001),
            "models_available": {
                "yolo11": self.yolo_model is not None,
                "yunet": self.yunet_detector is not None,
                "opencv_dnn": self.net is not None,
                "cascade": hasattr(self, 'face_cascade')
            }
        }

    # Legacy methods for compatibility
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Legacy method - detect faces only"""
        faces, _ = self.detect_faces_and_foreheads(frame, draw_detections=True)
        return faces

    def get_closest_face(self, faces: List[FaceDetection]) -> Optional[FaceDetection]:
        """Legacy method - get closest (largest) face"""
        if not faces:
            return None
        return max(faces, key=lambda f: f.width * f.height)



