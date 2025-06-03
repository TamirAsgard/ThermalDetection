from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from typing import List, Dict, Any
from datetime import datetime


class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self, thermal_processor):
        self.thermal_processor = thermal_processor
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
        self.is_broadcasting = False

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

        # Start broadcasting if first connection
        if len(self.active_connections) == 1 and not self.is_broadcasting:
            asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_message(self, message: str):
        """Broadcast message to all connected WebSockets"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def _broadcast_loop(self):
        """Main broadcasting loop"""
        self.is_broadcasting = True

        while self.active_connections:
            try:
                # Get current reading
                reading = self.thermal_processor.get_averaged_reading()

                # Get latest detection data (this was missing!)
                detection_data = self.thermal_processor.get_latest_detection_data()

                if reading and detection_data.get('face_detection'):
                    # Build complete message with detection data
                    message_data = {
                        "temperature": reading.avg_temperature,
                        "max_temperature": reading.max_temperature,
                        "min_temperature": reading.min_temperature,
                        "confidence": reading.confidence,
                        "is_fever": reading.is_fever,
                        "timestamp": reading.timestamp.isoformat(),
                        "person_detected": True,
                        "pixel_count": reading.pixel_count,
                        # Add frame dimensions
                        "frame_width": detection_data.get('frame_width', 640),
                        "frame_height": detection_data.get('frame_height', 480)
                    }

                    # Add face detection data if available
                    face_detection = detection_data.get('face_detection')
                    if face_detection:
                        message_data["face_detection"] = {
                            "x": face_detection.x,
                            "y": face_detection.y,
                            "width": face_detection.width,
                            "height": face_detection.height,
                            "confidence": face_detection.confidence
                        }

                    # Add forehead detection data if available
                    forehead_detection = detection_data.get('forehead_detection')
                    if forehead_detection:
                        message_data["forehead_detection"] = {
                            "x": forehead_detection['x'],
                            "y": forehead_detection['y'],
                            "width": forehead_detection['width'],
                            "height": forehead_detection['height'],
                            "confidence": 0.9  # Derived confidence
                        }

                    message = {
                        "type": "temperature_reading",
                        "data": message_data
                    }
                else:
                    message = {
                        "type": "no_detection",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                            "person_detected": False
                        }
                    }

                await self.broadcast_message(json.dumps(message))
                await asyncio.sleep(1 / 10)  # 10 Hz updates

            except Exception as e:
                self.logger.error(f"Broadcasting error: {e}")
                await asyncio.sleep(1)

        self.is_broadcasting = False