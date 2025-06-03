import cv2
import numpy as np
from typing import Tuple, Optional
import logging


class ImageProcessor:
    """Image processing utilities for thermal detection"""

    @staticmethod
    def align_thermal_rgb(thermal_frame: np.ndarray,
                          rgb_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align thermal and RGB frames"""
        # Simple resize-based alignment
        # For production, use proper camera calibration

        h_rgb, w_rgb = rgb_frame.shape[:2]
        thermal_aligned = cv2.resize(thermal_frame, (w_rgb, h_rgb))

        return thermal_aligned, rgb_frame

    @staticmethod
    def enhance_thermal_image(thermal_frame: np.ndarray) -> np.ndarray:
        """Enhance thermal image for better visualization"""
        # Normalize to 0-255 range
        normalized = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(normalized.astype(np.uint8))

        # Apply Gaussian blur for noise reduction
        smoothed = cv2.GaussianBlur(equalized, (3, 3), 0)

        return smoothed

    @staticmethod
    def apply_thermal_colormap(thermal_frame: np.ndarray,
                               colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply colormap to thermal image"""
        normalized = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), colormap)
        return colored

    @staticmethod
    def overlay_temperature_info(frame: np.ndarray,
                                 temperature: float,
                                 position: Tuple[int, int],
                                 is_fever: bool = False) -> np.ndarray:
        """Overlay temperature information on frame"""
        x, y = position
        temp_text = f"{temperature:.1f}°C"

        # Choose color based on fever status
        color = (0, 0, 255) if is_fever else (0, 255, 0)  # Red for fever, green for normal

        # Add background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            temp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        cv2.rectangle(frame,
                      (x, y - text_height - 10),
                      (x + text_width + 10, y + 5),
                      (0, 0, 0), -1)

        # Add text
        cv2.putText(frame, temp_text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    @staticmethod
    def draw_face_rectangle(frame: np.ndarray,
                            face_detection,
                            color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw rectangle around detected face"""
        cv2.rectangle(frame,
                      (face_detection.x, face_detection.y),
                      (face_detection.x + face_detection.width,
                       face_detection.y + face_detection.height),
                      color, 2)

        # Add confidence text
        conf_text = f"Conf: {face_detection.confidence:.2f}"
        cv2.putText(frame, conf_text,
                    (face_detection.x, face_detection.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    @staticmethod
    def draw_forehead_region(frame: np.ndarray,
                             forehead_region,
                             color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Draw rectangle around forehead region"""
        cv2.rectangle(frame,
                      (forehead_region.x, forehead_region.y),
                      (forehead_region.x + forehead_region.width,
                       forehead_region.y + forehead_region.height),
                      color, 1)

        return frame