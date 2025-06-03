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

        # Enhanced web interface with dual camera support
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

        .thermal-forehead {
            border-color: #ff6b6b !important;
            background: rgba(255, 107, 107, 0.2) !important;
            box-shadow: 0 0 20px rgba(255, 107, 107, 0.7) !important;
        }

        .thermal-temp-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 10px;
            padding: 10px;
            color: white;
            text-align: center;
            min-width: 80px;
        }

        .temp-value {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .temp-status {
            font-size: 0.9em;
            opacity: 0.9;
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

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 10px;
        }

        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .stat-number {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
            color: #2ed573;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .measurement-timer {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            text-align: center;
        }

        .timer-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }

        .timer-value {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .timer-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
            overflow: hidden;
        }

        .timer-progress {
            height: 100%;
            background: linear-gradient(90deg, #2ed573, #00d2ff);
            border-radius: 3px;
            transition: width 0.1s ease;
            width: 0%;
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
                    <div class="stat-card">
                        <div class="stat-title">Detection Method</div>
                        <div id="detectionMethod" class="stat-value">--</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Frame Rate</div>
                        <div id="frameRate" class="stat-value">--</div>
                    </div>
                </div>

                <div class="measurement-timer">
                    <div class="timer-label">Next Measurement In:</div>
                    <div id="measurementTimer" class="timer-value">Ready</div>
                    <div class="timer-bar">
                        <div id="timerProgress" class="timer-progress"></div>
                    </div>
                </div>

                <div class="controls">
                    <button class="btn" onclick="takeScreenshot()">Screenshot</button>
                    <button class="btn" onclick="resetChart()">Reset Chart</button>
                    <button class="btn" onclick="calibrateSystem()">Calibrate</button>
                </div>
            </div>

            <div class="card">
                <h3>Live RGB Detection</h3>
                <div class="detection-container">
                    <img id="videoStream" class="video-stream" 
                         src="/api/v1/video/rgb" alt="Live RGB Video Stream">
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

        <div class="grid">
            <div class="card">
                <h3>Thermal Camera View</h3>
                <div class="detection-container">
                    <img id="thermalStream" class="video-stream" 
                         src="/api/v1/video/thermal" alt="Thermal Video Stream">
                    <div class="detection-overlay">
                        <div id="thermalFaceBox" class="face-box" style="display: none;">
                            <div class="confidence-label" id="thermalFaceLabel">Face: 0%</div>
                        </div>
                        <div id="thermalForeheadBox" class="forehead-box thermal-forehead" style="display: none;">
                            <div class="confidence-label" id="thermalForeheadLabel">Forehead: 0%</div>
                        </div>
                        <div id="thermalTemp" class="thermal-temp-overlay" style="display: none;">
                            <div class="temp-value">36.5°C</div>
                            <div class="temp-status">Normal</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Detection Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number" id="totalMeasurements">0</div>
                        <div class="stat-label">Total Measurements</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="feverDetections">0</div>
                        <div class="stat-label">Fever Detections</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="avgTemperature">--</div>
                        <div class="stat-label">Average Temp</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="systemUptime">--</div>
                        <div class="stat-label">System Uptime</div>
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
        let totalMeasurements = 0;
        let feverCount = 0;
        let temperatureSum = 0;
        let startTime = new Date();
        let lastMeasurementTime = 0;
        const MEASUREMENT_INTERVAL = 3000; // 3 seconds between measurements

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
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#ff6b6b',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
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
                        ticks: { 
                            color: 'white',
                            callback: function(value) {
                                return value.toFixed(1) + '°C';
                            }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        title: {
                            display: true,
                            text: 'Temperature (°C)',
                            color: 'white'
                        }
                    },
                    x: {
                        ticks: { 
                            color: 'white',
                            maxTicksLimit: 10
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        title: {
                            display: true,
                            text: 'Time',
                            color: 'white'
                        }
                    }
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });

        // WebSocket message handling
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const statusDiv = document.getElementById('status');
            const measurementDiv = document.getElementById('measurement');

            console.log('WebSocket message:', data.type, data.data);

            if (data.type === 'temperature_reading') {
                const currentTime = Date.now();
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

                // Only record measurement if enough time has passed
                if (currentTime - lastMeasurementTime >= MEASUREMENT_INTERVAL) {
                    // Update statistics
                    totalMeasurements++;
                    temperatureSum += data.data.temperature;
                    if (isFever) {
                        feverCount++;
                    }
                    updateStatistics();

                    // Update chart with actual temperature reading
                    const now = new Date();
                    const timeLabel = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
                    tempData.push(data.data.temperature);
                    labels.push(timeLabel);

                    if (tempData.length > 20) {  // Keep last 20 readings
                        tempData.shift();
                        labels.shift();
                    }

                    chart.update('none'); // Smooth update without animation
                    lastMeasurementTime = currentTime;

                    console.log(`New measurement recorded: ${temp}°C at ${timeLabel}`);
                }

                // Always update detection visualization
                updateDetectionBoxes(data.data, 'rgb');
                updateDetectionBoxes(data.data, 'thermal');
                updateThermalOverlay(data.data);

                // Update confidence displays
                updateConfidenceStats(data.data);

            } else if (data.type === 'detection_only') {
                // Person detected but no valid temperature reading
                statusDiv.className = 'status healthy';
                statusDiv.textContent = 'Person Detected - Measuring Temperature...';

                measurementDiv.textContent = 'Measuring...';
                measurementDiv.className = 'measurement';
                measurementDiv.classList.remove('pulse');

                // Update detection visualization for both RGB and thermal
                updateDetectionBoxes(data.data, 'rgb');
                updateDetectionBoxes(data.data, 'thermal');

                // Update confidence displays
                updateConfidenceStats(data.data);

                // Don't update thermal overlay or chart for detection-only

            } else if (data.type === 'no_detection') {
                statusDiv.className = 'status healthy';
                statusDiv.textContent = 'System Online - No Person Detected';

                measurementDiv.textContent = 'No person detected';
                measurementDiv.className = 'measurement';
                measurementDiv.classList.remove('pulse');
                hideDetectionBoxes();
                hideThermalOverlay();
                updateConfidenceStats(null);

                // Update method info even when no detection
                updateSystemInfo(data.data);
            }
        };

        function updateDetectionBoxes(data, streamType = 'rgb') {
            const prefix = streamType === 'thermal' ? 'thermal' : '';
            const videoStream = document.getElementById(streamType === 'thermal' ? 'thermalStream' : 'videoStream');
            const faceBox = document.getElementById(prefix + 'faceBox');
            const foreheadBox = document.getElementById(prefix + 'foreheadBox');
            const faceLabel = document.getElementById(prefix + 'faceLabel');
            const foreheadLabel = document.getElementById(prefix + 'foreheadLabel');

            if (!videoStream || !faceBox || !foreheadBox) {
                console.warn('Missing DOM elements for', streamType);
                return;
            }

            // Get video dimensions
            const videoRect = videoStream.getBoundingClientRect();
            const scaleX = videoRect.width / (data.frame_width || 640);
            const scaleY = videoRect.height / (data.frame_height || 480);

            console.log(`Updating ${streamType} detection boxes:`, data);

            // Update face detection box
            if (data.face_detection && data.face_detection.x !== undefined) {
                const face = data.face_detection;
                faceBox.style.display = 'block';
                faceBox.style.left = (face.x * scaleX) + 'px';
                faceBox.style.top = (face.y * scaleY) + 'px';
                faceBox.style.width = (face.width * scaleX) + 'px';
                faceBox.style.height = (face.height * scaleY) + 'px';
                faceLabel.textContent = `Face: ${(face.confidence * 100).toFixed(1)}%`;
                console.log(`${streamType} face box updated:`, face);
            } else {
                faceBox.style.display = 'none';
                console.log(`${streamType} face box hidden`);
            }

            // Update forehead detection box
            if (data.forehead_detection && data.forehead_detection.x !== undefined) {
                const forehead = data.forehead_detection;
                foreheadBox.style.display = 'block';
                foreheadBox.style.left = (forehead.x * scaleX) + 'px';
                foreheadBox.style.top = (forehead.y * scaleY) + 'px';
                foreheadBox.style.width = (forehead.width * scaleX) + 'px';
                foreheadBox.style.height = (forehead.height * scaleY) + 'px';
                foreheadLabel.textContent = `Forehead: ${(forehead.confidence * 100).toFixed(1)}%`;
                console.log(`${streamType} forehead box updated:`, forehead);
            } else {
                foreheadBox.style.display = 'none';
                console.log(`${streamType} forehead box hidden`);
            }
        }

        function updateThermalOverlay(data) {
            const thermalTemp = document.getElementById('thermalTemp');

            if (data.temperature && data.person_detected) {
                thermalTemp.style.display = 'block';

                const tempValue = document.querySelector('#thermalTemp .temp-value');
                const tempStatus = document.querySelector('#thermalTemp .temp-status');

                tempValue.textContent = `${data.temperature.toFixed(1)}°C`;
                tempStatus.textContent = data.is_fever ? 'FEVER' : 'Normal';

                // Color coding
                if (data.is_fever) {
                    tempValue.style.color = '#ff4757';
                    tempStatus.style.color = '#ff4757';
                    thermalTemp.style.borderLeft = '4px solid #ff4757';
                } else {
                    tempValue.style.color = '#2ed573';
                    tempStatus.style.color = '#2ed573';
                    thermalTemp.style.borderLeft = '4px solid #2ed573';
                }
            } else {
                thermalTemp.style.display = 'none';
            }
        }

        function hideDetectionBoxes() {
            ['faceBox', 'foreheadBox', 'thermalFaceBox', 'thermalForeheadBox'].forEach(id => {
                document.getElementById(id).style.display = 'none';
            });
        }

        function hideThermalOverlay() {
            document.getElementById('thermalTemp').style.display = 'none';
        }

        function updateConfidenceStats(data) {
            const faceConfidence = document.getElementById('faceConfidence');
            const foreheadConfidence = document.getElementById('foreheadConfidence');
            const detectionMethod = document.getElementById('detectionMethod');
            const frameRate = document.getElementById('frameRate');

            console.log('Updating confidence stats:', data);

            if (data && data.face_detection) {
                const faceConf = data.face_detection.confidence;
                faceConfidence.textContent = `${(faceConf * 100).toFixed(1)}%`;
                faceConfidence.style.color = faceConf > 0.8 ? '#2ed573' : faceConf > 0.6 ? '#ffaa00' : '#ff4757';
                console.log('Face confidence updated:', faceConf);
            } else {
                faceConfidence.textContent = '--';
                faceConfidence.style.color = 'white';
            }

            if (data && data.forehead_detection) {
                const foreheadConf = data.forehead_detection.confidence;
                foreheadConfidence.textContent = `${(foreheadConf * 100).toFixed(1)}%`;
                foreheadConfidence.style.color = foreheadConf > 0.8 ? '#2ed573' : foreheadConf > 0.6 ? '#ffaa00' : '#ff4757';
                console.log('Forehead confidence updated:', foreheadConf);
            } else {
                foreheadConfidence.textContent = '--';
                foreheadConfidence.style.color = 'white';
            }

            // Update detection method and frame rate
            if (data && data.detection_method) {
                detectionMethod.textContent = data.detection_method.toUpperCase();
                detectionMethod.style.color = data.detection_method === 'yolo11' ? '#00ff00' : '#ffaa00';
            } else {
                detectionMethod.textContent = 'YOLO11';
                detectionMethod.style.color = '#00ff00';
            }

            // Update frame rate
            if (data && data.fps) {
                frameRate.textContent = `${data.fps.toFixed(1)} FPS`;
                frameRate.style.color = data.fps > 20 ? '#2ed573' : data.fps > 10 ? '#ffaa00' : '#ff4757';
            } else {
                frameRate.textContent = '30.0 FPS';
                frameRate.style.color = '#2ed573';
            }
        }

        function updateSystemInfo(data) {
            // Update system info for no-detection cases
            const detectionMethod = document.getElementById('detectionMethod');
            const frameRate = document.getElementById('frameRate');

            if (data && data.detection_method) {
                detectionMethod.textContent = data.detection_method.toUpperCase();
                detectionMethod.style.color = data.detection_method === 'yolo11' ? '#00ff00' : '#ffaa00';
            }

            if (data && data.fps) {
                frameRate.textContent = `${data.fps.toFixed(1)} FPS`;
                frameRate.style.color = data.fps > 20 ? '#2ed573' : data.fps > 10 ? '#ffaa00' : '#ff4757';
            }
        }

        function updateStatistics() {
            // Update total measurements
            document.getElementById('totalMeasurements').textContent = totalMeasurements;

            // Update fever detections
            document.getElementById('feverDetections').textContent = feverCount;
            document.getElementById('feverDetections').style.color = feverCount > 0 ? '#ff4757' : '#2ed573';

            // Update average temperature
            const avgTemp = totalMeasurements > 0 ? (temperatureSum / totalMeasurements) : 0;
            document.getElementById('avgTemperature').textContent = totalMeasurements > 0 ? `${avgTemp.toFixed(1)}°C` : '--';

            // Update system uptime
            const uptime = new Date() - startTime;
            const hours = Math.floor(uptime / (1000 * 60 * 60));
            const minutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
            document.getElementById('systemUptime').textContent = `${hours}h ${minutes}m`;

            // Update measurement rate
            const uptimeSeconds = uptime / 1000;
            const measurementRate = totalMeasurements > 0 ? (totalMeasurements / (uptimeSeconds / 60)).toFixed(1) : '0.0';

            console.log(`Statistics: ${totalMeasurements} measurements, ${measurementRate} per minute, avg temp: ${avgTemp.toFixed(1)}°C`);
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
            totalMeasurements = 0;
            feverCount = 0;
            temperatureSum = 0;
            startTime = new Date();
            lastMeasurementTime = 0;
            updateStatistics();
            chart.update();
        }

        function calibrateSystem() {
            fetch('/api/v1/config/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    calibration_offset: 0.0
                })
            })
            .then(response => response.json())
            .then(data => {
                alert('System calibration reset successfully');
            })
            .catch(error => {
                console.error('Calibration failed:', error);
                alert('Calibration failed');
            });
        }

        // WebSocket connection handlers
        ws.onopen = function() {
            document.getElementById('status').textContent = 'Connected to system';
            document.getElementById('status').className = 'status healthy';
            console.log('WebSocket connected');
        };

        ws.onclose = function() {
            document.getElementById('status').textContent = 'Disconnected from system';
            document.getElementById('status').className = 'status error';
            hideDetectionBoxes();
            hideThermalOverlay();
            console.log('WebSocket disconnected');
        };

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            document.getElementById('status').textContent = 'Connection error';
            document.getElementById('status').className = 'status error';
        };

        // Handle video stream load events
        document.getElementById('videoStream').onload = function() {
            console.log('RGB video stream loaded');
        };

        document.getElementById('videoStream').onerror = function() {
            console.log('RGB video stream error');
        };

        document.getElementById('thermalStream').onload = function() {
            console.log('Thermal video stream loaded');
        };

        document.getElementById('thermalStream').onerror = function() {
            console.log('Thermal video stream error');
        };

        // Initialize statistics and timer
        updateStatistics();

        // Update uptime every minute
        setInterval(updateStatistics, 60000);

        // Update measurement timer every 100ms
        setInterval(updateMeasurementTimer, 100);

        function updateMeasurementTimer() {
            const currentTime = Date.now();
            const timeSinceLastMeasurement = currentTime - lastMeasurementTime;
            const timeUntilNext = Math.max(0, MEASUREMENT_INTERVAL - timeSinceLastMeasurement);

            const timerElement = document.getElementById('measurementTimer');
            const progressElement = document.getElementById('timerProgress');

            if (timeUntilNext === 0) {
                timerElement.textContent = 'Ready';
                timerElement.style.color = '#2ed573';
                progressElement.style.width = '100%';
            } else {
                const seconds = Math.ceil(timeUntilNext / 1000);
                timerElement.textContent = `${seconds}s`;
                timerElement.style.color = '#ffaa00';

                const progress = ((MEASUREMENT_INTERVAL - timeUntilNext) / MEASUREMENT_INTERVAL) * 100;
                progressElement.style.width = `${progress}%`;
            }
        }

        // Add pulse animation styles
        const pulseStyle = document.createElement('style');
        pulseStyle.textContent = `
            .pulse {
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
        `;
        document.head.appendChild(pulseStyle);
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