# Thermal Detection System - API Documentation

## Overview

The Thermal Detection System provides a comprehensive REST API and WebSocket interface for real-time thermal forehead temperature measurement. The API is built with FastAPI and provides endpoints for temperature readings, system status, configuration management, and live video streaming.

## Base URL

```
Production: https://your-domain.com/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

Currently, the API supports optional API key authentication. To enable authentication, set `api_key_required: true` in the configuration and provide a valid API key.

### API Key Authentication
```http
GET /api/v1/temperature
Authorization: Bearer your-api-key-here
```

## Rate Limiting

The API implements rate limiting to prevent abuse:
- **Default limit**: 60 requests per minute per IP
- **WebSocket connections**: 5 connections per minute per IP
- **Burst limit**: 20 requests allowed in burst

## Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **Video Streams**: `multipart/x-mixed-replace; boundary=frame`

---

## Core Endpoints

### 1. Health Check

**GET** `/health`

Check the system health and service status.

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "camera": true,
    "thermal_processor": true,
    "api": true
  }
}
```

#### Response Codes
- `200`: System is healthy
- `503`: System degraded (some services unavailable)

---

### 2. Temperature Reading

**GET** `/temperature`

Get the current temperature reading with face detection data.

#### Response
```json
{
  "temperature": 36.5,
  "max_temperature": 37.0,
  "min_temperature": 36.0,
  "confidence": 0.92,
  "is_fever": false,
  "timestamp": "2024-01-15T10:30:15Z",
  "person_detected": true,
  "pixel_count": 150,
  "face_detection": {
    "x": 200,
    "y": 150,
    "width": 200,
    "height": 200,
    "confidence": 0.85
  },
  "forehead_detection": {
    "x": 220,
    "y": 170,
    "width": 160,
    "height": 60,
    "confidence": 0.80
  },
  "frame_width": 640,
  "frame_height": 480
}
```

#### Response Codes
- `200`: Temperature reading available
- `404`: No temperature reading available (no person detected)
- `500`: System error

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | float | Average forehead temperature in Celsius |
| `max_temperature` | float | Maximum temperature in measurement region |
| `min_temperature` | float | Minimum temperature in measurement region |
| `confidence` | float | Measurement confidence (0.0-1.0) |
| `is_fever` | boolean | Whether temperature indicates fever |
| `timestamp` | string | ISO 8601 timestamp of measurement |
| `person_detected` | boolean | Whether a person was detected |
| `pixel_count` | integer | Number of pixels used in measurement |
| `face_detection` | object | Face detection coordinates and confidence |
| `forehead_detection` | object | Forehead region coordinates and confidence |
| `frame_width` | integer | Video frame width for coordinate reference |
| `frame_height` | integer | Video frame height for coordinate reference |

---

### 3. System Status

**GET** `/status`

Get detailed system status and performance metrics.

#### Response
```json
{
  "is_running": true,
  "camera_connected": true,
  "last_measurement": "2024-01-15T10:30:15Z",
  "total_measurements": 1547,
  "error_count": 3,
  "uptime_seconds": 86400.0
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_running` | boolean | Whether thermal processor is running |
| `camera_connected` | boolean | Camera connection status |
| `last_measurement` | string | Timestamp of last successful measurement |
| `total_measurements` | integer | Total number of measurements taken |
| `error_count` | integer | Number of errors since startup |
| `uptime_seconds` | float | System uptime in seconds |

---

### 4. Camera Information

**GET** `/camera/info`

Get camera configuration and status information.

#### Response
```json
{
  "thermal_device_type": "webcam",
  "frame_width": 640,
  "frame_height": 480,
  "fps": 30,
  "rgb_connected": true,
  "thermal_connected": true
}
```

#### Response Codes
- `200`: Camera information available
- `503`: Camera not initialized

---

### 5. Camera Restart

**POST** `/camera/restart`

Restart the camera connection (useful for recovering from errors).

#### Response
```json
{
  "message": "Camera restart initiated"
}
```

#### Response Codes
- `200`: Restart initiated successfully
- `500`: Failed to restart camera

---

### 6. Configuration Update

**POST** `/config/update`

Update system configuration parameters.

#### Request Body
```json
{
  "fever_threshold": 37.8,
  "confidence_threshold": 0.7,
  "calibration_offset": 0.5,
  "averaging_samples": 10
}
```

#### Request Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `fever_threshold` | float | 30.0-45.0 | Temperature threshold for fever detection (°C) |
| `confidence_threshold` | float | 0.0-1.0 | Minimum confidence for face detection |
| `calibration_offset` | float | -5.0-5.0 | Temperature calibration offset (°C) |
| `averaging_samples` | integer | 1-50 | Number of readings to average |

#### Response
```json
{
  "message": "Configuration updated successfully"
}
```

#### Response Codes
- `200`: Configuration updated successfully
- `400`: Invalid configuration parameters
- `422`: Validation error

---

## Video Streaming Endpoints

### 7. RGB Video Stream

**GET** `/video/rgb`

Get live RGB video stream with face detection overlays.

#### Response
- **Content-Type**: `multipart/x-mixed-replace; boundary=frame`
- **Format**: MJPEG stream
- **Overlays**: Face detection rectangles, temperature information

#### Usage Example
```html
<img src="/api/v1/video/rgb" alt="Live RGB Stream" />
```

---

### 8. Thermal Video Stream

**GET** `/video/thermal`

Get live thermal video stream with temperature visualization.

#### Response
- **Content-Type**: `multipart/x-mixed-replace; boundary=frame`
- **Format**: MJPEG stream with thermal colormap
- **Overlays**: Temperature regions, detection boxes

#### Usage Example
```html
<img src="/api/v1/video/thermal" alt="Live Thermal Stream" />
```

---

## WebSocket API

### Connection

Connect to the WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Message Types

#### 1. Temperature Reading Message
```json
{
  "type": "temperature_reading",
  "data": {
    "temperature": 36.5,
    "confidence": 0.92,
    "is_fever": false,
    "timestamp": "2024-01-15T10:30:15Z",
    "person_detected": true,
    "face_detection": {
      "x": 200,
      "y": 150,
      "width": 200,
      "height": 200,
      "confidence": 0.85
    },
    "forehead_detection": {
      "x": 220,
      "y": 170,
      "width": 160,
      "height": 60,
      "confidence": 0.80
    },
    "frame_width": 640,
    "frame_height": 480,
    "detection_method": "yolo11",
    "fps": 28.5
  }
}
```

#### 2. Detection Only Message
```json
{
  "type": "detection_only",
  "data": {
    "timestamp": "2024-01-15T10:30:15Z",
    "person_detected": true,
    "temperature_available": false,
    "face_detection": {
      "x": 200,
      "y": 150,
      "width": 200,
      "height": 200,
      "confidence": 0.85
    },
    "detection_method": "yolo11",
    "fps": 30.0
  }
}
```

#### 3. No Detection Message
```json
{
  "type": "no_detection",
  "data": {
    "timestamp": "2024-01-15T10:30:15Z",
    "person_detected": false,
    "detection_method": "yolo11",
    "fps": 30.0
  }
}
```

### WebSocket Usage Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    console.log('Connected to thermal detection system');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'temperature_reading':
            updateTemperatureDisplay(data.data);
            break;
        case 'detection_only':
            updateDetectionDisplay(data.data);
            break;
        case 'no_detection':
            clearDisplay();
            break;
    }
};

ws.onclose = function(event) {
    console.log('Disconnected from thermal detection system');
};

function updateTemperatureDisplay(data) {
    document.getElementById('temperature').textContent = 
        `${data.temperature.toFixed(1)}°C`;
    document.getElementById('status').textContent = 
        data.is_fever ? 'FEVER' : 'Normal';
}
```

---

## Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "error": "ErrorType",
  "detail": "Detailed error message",
  "timestamp": "2024-01-15T10:30:15Z"
}
```

### Common Error Codes

| Code | Error | Description |
|------|-------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Invalid or missing API key |
| 404 | Not Found | Resource not found |
| 422 | Validation Error | Request validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | System error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Example Error Response

```json
{
  "error": "ValidationError",
  "detail": "fever_threshold must be between 30.0 and 45.0",
  "timestamp": "2024-01-15T10:30:15Z"
}
```

---

## SDK Examples

### Python SDK Example

```python
import requests
import json

class ThermalDetectionClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def get_temperature(self):
        """Get current temperature reading"""
        response = requests.get(f'{self.base_url}/temperature', 
                              headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_status(self):
        """Get system status"""
        response = requests.get(f'{self.base_url}/status', 
                              headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def update_config(self, **kwargs):
        """Update configuration"""
        response = requests.post(f'{self.base_url}/config/update',
                               json=kwargs, headers=self.headers)
        response.raise_for_status()
        return response.json()

# Usage
client = ThermalDetectionClient('http://localhost:8000/api/v1')

# Get temperature
try:
    temp_data = client.get_temperature()
    print(f"Temperature: {temp_data['temperature']}°C")
    print(f"Fever: {temp_data['is_fever']}")
except requests.exceptions.HTTPError as e:
    print(f"Error: {e}")
```

### JavaScript SDK Example

```javascript
class ThermalDetectionClient {
    constructor(baseUrl, apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json',
            ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
        };
    }

    async getTemperature() {
        const response = await fetch(`${this.baseUrl}/temperature`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    async updateConfig(config) {
        const response = await fetch(`${this.baseUrl}/config/update`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage
const client = new ThermalDetectionClient('http://localhost:8000/api/v1');

client.getTemperature()
    .then(data => {
        console.log(`Temperature: ${data.temperature}°C`);
        console.log(`Fever: ${data.is_fever}`);
    })
    .catch(error => console.error('Error:', error));
```

---

## Performance Considerations

### Response Times
- **Temperature readings**: < 50ms
- **Status endpoints**: < 10ms
- **Video streams**: 15-30 FPS depending on hardware
- **WebSocket updates**: 10 Hz (10 updates per second)

### Optimization Tips
1. **Use WebSocket** for real-time updates instead of polling
2. **Cache status information** - it doesn't change frequently
3. **Limit video stream** resolution if bandwidth is constrained
4. **Batch configuration updates** instead of individual calls

### Scalability
- **Concurrent connections**: Up to 100 WebSocket connections
- **API requests**: 60 requests/minute per IP (configurable)
- **Memory usage**: ~500MB per instance
- **CPU usage**: 1-2 cores recommended

---

## Integration Examples

### Home Assistant Integration

```yaml
# configuration.yaml
sensor:
  - platform: rest
    name: "Thermal Detection Temperature"
    resource: "http://192.168.1.100:8000/api/v1/temperature"
    value_template: "{{ value_json.temperature }}"
    unit_of_measurement: "°C"
    scan_interval: 5

binary_sensor:
  - platform: rest
    name: "Thermal Detection Fever"
    resource: "http://192.168.1.100:8000/api/v1/temperature"
    value_template: "{{ value_json.is_fever }}"
    device_class: "heat"
```

### Node-RED Integration

```json
[
    {
        "id": "thermal_temp",
        "type": "http request",
        "method": "GET",
        "url": "http://localhost:8000/api/v1/temperature",
        "name": "Get Temperature"
    },
    {
        "id": "fever_alert",
        "type": "switch",
        "property": "payload.is_fever",
        "rules": [
            {
                "t": "true"
            }
        ],
        "name": "Fever Check"
    }
]
```

---

## Troubleshooting

### Common Issues

#### 1. No Temperature Readings (404)
**Cause**: No person detected by camera
**Solutions**:
- Check camera connection
- Ensure proper lighting
- Verify face is visible and properly positioned
- Check detection confidence threshold

#### 2. Low Confidence Readings
**Cause**: Poor face detection or thermal data quality
**Solutions**:
- Improve lighting conditions
- Adjust camera position
- Lower confidence threshold in configuration
- Calibrate thermal camera

#### 3. WebSocket Connection Issues
**Cause**: Network connectivity or server overload
**Solutions**:
- Check network connectivity
- Verify WebSocket URL
- Check server logs for errors
- Restart the service

#### 4. High Memory Usage
**Cause**: Memory leaks or large video buffers
**Solutions**:
- Restart the service
- Reduce video resolution
- Check for memory leaks in logs
- Update to latest version

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Environment variable
THERMAL_LOG_LEVEL=DEBUG

# Or in configuration
logging:
  level: DEBUG
  log_to_console: true
```

---

## API Changelog

### Version 1.0.0
- Initial API release
- Core temperature measurement endpoints
- WebSocket real-time updates
- Video streaming support
- Basic configuration management

### Upcoming Features
- Historical data endpoints
- Advanced analytics API
- Multi-camera support
- Export/import functionality
- Enhanced authentication options