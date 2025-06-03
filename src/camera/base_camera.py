from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple
import asyncio


class BaseCameraInterface(ABC):
    """Abstract base class for camera interfaces"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize camera connection"""
        pass

    @abstractmethod
    async def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame for face detection"""
        pass

    @abstractmethod
    async def get_thermal_frame(self) -> Optional[np.ndarray]:
        """Get thermal frame for temperature measurement"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        pass

    @abstractmethod
    def get_camera_info(self) -> dict:
        """Get camera information and capabilities"""
        pass

    @abstractmethod
    def release(self):
        """Release camera resources"""
        pass