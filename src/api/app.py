from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import logging
from contextlib import asynccontextmanager

from src.api.routes import ThermalAPIRoutes
from src.api.websocket_handler import WebSocketManager
from src.core.thermal_processor import ThermalProcessor
from src.camera.camera_manager import CameraManager
from config.settings import ThermalConfig
import os

os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

class ThermalAPI:
    """Main FastAPI application for thermal detection system"""

    def __init__(self, config: ThermalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.camera_manager = CameraManager(config)
        self.thermal_processor = ThermalProcessor(config, self.camera_manager)
        self.websocket_manager = WebSocketManager(self.thermal_processor)

        # Create FastAPI app
        self.app = FastAPI(
            title="Thermal Forehead Detection API",
            description="Real-time forehead temperature measurement system",
            version="1.0.0",
            lifespan=self.lifespan
        )

        self._setup_middleware()
        self._setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan manager"""
        # Startup
        self.logger.info("Starting thermal detection system...")

        # Initialize camera
        camera_success = await self.camera_manager.initialize()
        if not camera_success:
            self.logger.error("Failed to initialize camera")
            raise RuntimeError("Camera initialization failed")

        # Start thermal processor
        processor_task = asyncio.create_task(
            self.thermal_processor.start_processing()
        )

        self.logger.info("Thermal detection system started successfully")

        yield

        # Shutdown
        self.logger.info("Shutting down thermal detection system...")

        # Stop processor
        self.thermal_processor.stop_processing()
        processor_task.cancel()

        # Stop camera
        await self.camera_manager.stop()

        self.logger.info("Thermal detection system stopped")

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""
        # API routes
        api_routes = ThermalAPIRoutes(
            self.thermal_processor,
            self.camera_manager,
            self.config
        )
        self.app.include_router(api_routes.router, prefix="/api/v1")

        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)

        # Root endpoint
        @self.app.get("/")
        async def root():
            return {"message": "Thermal Forehead Detection System", "version": "1.0.0"}

        # Simple web interface
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            return """
<!DOCTYPE html>
<html>
<head>
    <title>Thermal Detection Dashboard - Enhanced</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .status { 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 10px; 
            text-align: center;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .healthy { 
            background: linear-gradient(45deg, #4CAF50, #45a049); 
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .error { 
            background: linear-gradient(45deg, #f44336, #da190b); 
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        
        .measurement { 
            font-size: 3em; 
            margin: 20px 0; 
            text-align: center;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .fever { color: #ff4757; }
        .normal { color: #2ed573; }
        
        .detection-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .video-stream {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .detection-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .face-box {
            position: absolute;
            border: 3px solid #00ff00;
            border-radius: 5px;
            background: rgba(0, 255, 0, 0.1);
            transition: all 0.3s ease;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
        }
        
        .forehead-box {
            position: absolute;
            border: 3px solid #ff6b6b;
            border-radius: 5px;
            background: rgba(255, 107, 107, 0.1);
            transition: all 0.3s ease;
            box-shadow: 0 0 15px rgba(255, 107, 107, 0.5);
        }
        
        .confidence-label {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            top: -25px;
            left: 0;
            white-space: nowrap;
        }
        
        .detection-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-title {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }
        
        #temperatureChart {
            max-height: 300px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌡️ Thermal Detection Dashboard</h1>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>System Status</h3>
                <div id="status" class="status">Connecting...</div>
                <div id="measurement" class="measurement">No reading</div>
                
                <div class="detection-stats">
                    <div class="stat-card">
                        <div class="stat-title">Face Confidence</div>
                        <div id="faceConfidence" class="stat-value">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Forehead Confidence</div>
                        <div id="foreheadConfidence" class="stat-value">--</div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn" onclick="toggleStream()">Toggle Stream</button>
                    <button class="btn" onclick="takeScreenshot()">Screenshot</button>
                    <button class="btn" onclick="resetChart()">Reset Chart</button>
                </div>
            </div>
            
            <div class="card">
                <h3>Live Detection</h3>
                <div class="detection-container">
                    <img id="videoStream" class="video-stream" 
                         src="/api/v1/video/rgb" alt="Live Video Stream">
                    <div class="detection-overlay">
                        <div id="faceBox" class="face-box" style="display: none;">
                            <div class="confidence-label" id="faceLabel">Face: 0%</div>
                        </div>
                        <div id="foreheadBox" class="forehead-box" style="display: none;">
                            <div class="confidence-label" id="foreheadLabel">Forehead: 0%</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Temperature History</h3>
            <canvas id="temperatureChart"></canvas>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const tempData = [];
        const labels = [];
        let streamEnabled = true;
        
        // Chart setup
        const ctx = document.getElementById('temperatureChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: tempData,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: 'white' }
                    }
                },
                scales: {
                    y: { 
                        min: 35, 
                        max: 40,
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });
        
        // WebSocket message handling
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const statusDiv = document.getElementById('status');
            const measurementDiv = document.getElementById('measurement');
            
            if (data.type === 'temperature_reading') {
                const temp = data.data.temperature.toFixed(1);
                const isFever = data.data.is_fever;
                
                // Update status
                statusDiv.className = 'status healthy';
                statusDiv.textContent = `System Online - Confidence: ${(data.data.confidence * 100).toFixed(1)}%`;
                
                // Update temperature display
                measurementDiv.className = `measurement ${isFever ? 'fever' : 'normal'}`;
                measurementDiv.innerHTML = `${temp}°C<br><small>${isFever ? '🔥 FEVER' : '✓ Normal'}</small>`;
                
                if (isFever) {
                    measurementDiv.classList.add('pulse');
                } else {
                    measurementDiv.classList.remove('pulse');
                }
                
                // Update detection visualization
                updateDetectionBoxes(data.data);
                
                // Update confidence displays
                updateConfidenceStats(data.data);
                
                // Update chart
                const now = new Date().toLocaleTimeString();
                tempData.push(data.data.temperature);
                labels.push(now);
                
                if (tempData.length > 20) {
                    tempData.shift();
                    labels.shift();
                }
                
                chart.update('none'); // Smooth update without animation
                
            } else if (data.type === 'no_detection') {
                measurementDiv.textContent = 'No person detected';
                measurementDiv.className = 'measurement';
                measurementDiv.classList.remove('pulse');
                hideDetectionBoxes();
                updateConfidenceStats(null);
            }
        };
        
        function updateDetectionBoxes(data) {
            const videoStream = document.getElementById('videoStream');
            const faceBox = document.getElementById('faceBox');
            const foreheadBox = document.getElementById('foreheadBox');
            const faceLabel = document.getElementById('faceLabel');
            const foreheadLabel = document.getElementById('foreheadLabel');
            
            // Get video dimensions
            const videoRect = videoStream.getBoundingClientRect();
            const scaleX = videoRect.width / (data.frame_width || 640);
            const scaleY = videoRect.height / (data.frame_height || 480);
            
            // Update face detection box
            if (data.face_detection) {
                const face = data.face_detection;
                faceBox.style.display = 'block';
                faceBox.style.left = (face.x * scaleX) + 'px';
                faceBox.style.top = (face.y * scaleY) + 'px';
                faceBox.style.width = (face.width * scaleX) + 'px';
                faceBox.style.height = (face.height * scaleY) + 'px';
                faceLabel.textContent = `Face: ${(face.confidence * 100).toFixed(1)}%`;
            } else {
                faceBox.style.display = 'none';
            }
            
            // Update forehead detection box
            if (data.forehead_detection) {
                const forehead = data.forehead_detection;
                foreheadBox.style.display = 'block';
                foreheadBox.style.left = (forehead.x * scaleX) + 'px';
                foreheadBox.style.top = (forehead.y * scaleY) + 'px';
                foreheadBox.style.width = (forehead.width * scaleX) + 'px';
                foreheadBox.style.height = (forehead.height * scaleY) + 'px';
                foreheadLabel.textContent = `Forehead: ${(forehead.confidence * 100).toFixed(1)}%`;
            } else {
                foreheadBox.style.display = 'none';
            }
        }
        
        function hideDetectionBoxes() {
            document.getElementById('faceBox').style.display = 'none';
            document.getElementById('foreheadBox').style.display = 'none';
        }
        
        function updateConfidenceStats(data) {
            const faceConfidence = document.getElementById('faceConfidence');
            const foreheadConfidence = document.getElementById('foreheadConfidence');
            
            if (data && data.face_detection) {
                faceConfidence.textContent = `${(data.face_detection.confidence * 100).toFixed(1)}%`;
                faceConfidence.style.color = data.face_detection.confidence > 0.8 ? '#2ed573' : '#ff4757';
            } else {
                faceConfidence.textContent = '--';
                faceConfidence.style.color = 'white';
            }
            
            if (data && data.forehead_detection) {
                foreheadConfidence.textContent = `${(data.forehead_detection.confidence * 100).toFixed(1)}%`;
                foreheadConfidence.style.color = data.forehead_detection.confidence > 0.8 ? '#2ed573' : '#ff4757';
            } else {
                foreheadConfidence.textContent = '--';
                foreheadConfidence.style.color = 'white';
            }
        }
        
        function toggleStream() {
            const videoStream = document.getElementById('videoStream');
            streamEnabled = !streamEnabled;
            videoStream.style.display = streamEnabled ? 'block' : 'none';
        }
        
        function takeScreenshot() {
            const link = document.createElement('a');
            link.href = '/api/v1/video/rgb';
            link.download = `thermal_screenshot_${new Date().getTime()}.jpg`;
            link.click();
        }
        
        function resetChart() {
            tempData.length = 0;
            labels.length = 0;
            chart.update();
        }
        
        // WebSocket connection handlers
        ws.onopen = function() {
            document.getElementById('status').textContent = 'Connected to system';
            document.getElementById('status').className = 'status healthy';
        };
        
        ws.onclose = function() {
            document.getElementById('status').textContent = 'Disconnected from system';
            document.getElementById('status').className = 'status error';
            hideDetectionBoxes();
        };
        
        // Handle video stream load
        document.getElementById('videoStream').onload = function() {
            console.log('Video stream loaded');
        };
        
        document.getElementById('videoStream').onerror = function() {
            console.log('Video stream error');
            document.getElementById('videoStream').style.display = 'none';
        };
    </script>
</body>
</html>
            """


def create_app(config_path: str = "../../config/thermal_config.yaml") -> FastAPI:
    """Create and configure the FastAPI application"""
    config = ThermalConfig.from_yaml(config_path)
    thermal_api = ThermalAPI(config)
    return thermal_api.app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )