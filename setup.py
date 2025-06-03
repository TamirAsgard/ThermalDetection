#!/usr/bin/env python3
"""
Setup script for Thermal Forehead Detection System
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path


# Read version from __init__.py
def get_version():
    """Get version from package __init__.py"""
    init_file = Path("src/__init__.py")
    if init_file.exists():
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"


# Read long description from README
def get_long_description():
    """Get long description from README file"""
    readme_file = Path("README.md")
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "Thermal Forehead Detection System for real-time temperature screening"


# Read requirements from requirements.txt
def get_requirements():
    """Get requirements from requirements.txt"""
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


# Read development requirements
def get_dev_requirements():
    """Get development requirements"""
    dev_requirements_file = Path("tests/test_requirements.txt")
    if dev_requirements_file.exists():
        with open(dev_requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


# Platform-specific requirements
def get_platform_requirements():
    """Get platform-specific requirements"""
    platform_reqs = []

    # Raspberry Pi specific requirements
    if 'arm' in os.uname().machine.lower():
        platform_reqs.extend([
            'RPi.GPIO>=0.7.1',
            'picamera>=1.13',
            'adafruit-circuitpython-dht>=3.7.0'
        ])

    return platform_reqs


# Console scripts
console_scripts = [
    'thermal-detection=src.main:main',
    'thermal-api=src.api.app:main',
    'thermal-test=tests.test_runner:main',
    'thermal-calibrate=scripts.calibrate_cameras:main',
]

setup(
    name="thermal-detection-system",
    version=get_version(),
    author="Thermal Detection Team",
    author_email="info@thermal-detection.com",
    description="Real-time thermal forehead temperature detection system",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/thermal-detection-system",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/thermal-detection-system/issues",
        "Documentation": "https://thermal-detection-system.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/thermal-detection-system",
    },

    # Package discovery
    packages=find_packages(where="."),
    package_dir={"": "."},

    # Include non-Python files
    include_package_data=True,
    package_data={
        "": [
            "*.yaml", "*.yml", "*.json", "*.xml", "*.txt", "*.md",
            "*.wav", "*.mp3", "*.png", "*.jpg", "*.jpeg",
            "*.pb", "*.pbtxt", "*.pt", "*.onnx",
            "config/*", "model/*", "sounds/*", "static/*", "templates/*"
        ],
    },

    # Requirements
    python_requires=">=3.8",
    install_requires=get_requirements() + get_platform_requirements(),

    extras_require={
        "dev": get_dev_requirements(),
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "coverage>=7.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
            "tensorrt>=8.6.0",
        ],
        "raspberry-pi": [
            "RPi.GPIO>=0.7.1",
            "picamera>=1.13",
            "adafruit-circuitpython-dht>=3.7.0",
            "gpiozero>=1.6.2",
        ],
        "industrial": [
            "modbus-tk>=1.1.3",
            "pyserial>=3.5",
            "opcua>=0.98.13",
        ]
    },

    # Console scripts
    entry_points={
        "console_scripts": console_scripts,
    },

    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Environment :: Web Environment",
        "Environment :: Console",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
    ],

    # Keywords
    keywords=[
        "thermal", "temperature", "detection", "computer-vision",
        "face-detection", "health-screening", "opencv", "fastapi",
        "yolo", "real-time", "iot", "raspberry-pi", "medical"
    ],

    # Zip safe
    zip_safe=False,

    # Data files
    data_files=[
        ("config", ["config/thermal_config.yaml"]),
        ("docs", ["README.md", "LICENSE", "CHANGELOG.md"]),
        ("scripts", [
            "scripts/setup_environment.py",
            "scripts/calibrate_cameras.py",
            "scripts/run_tests.py"
        ]),
    ],
)


# Post-installation setup
class PostInstallCommand:
    """Post-installation setup commands"""

    @staticmethod
    def create_directories():
        """Create necessary directories"""
        directories = [
            "logs", "data", "backups", "test_data/images",
            "model", "sounds", "config/backups"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")

    @staticmethod
    def download_models():
        """Download required model files"""
        model_urls = {
            "haarcascade_frontalface_default.xml":
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            "opencv_face_detector_uint8.pb":
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
            "opencv_face_detector.pbtxt":
                "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/opencv_face_detector.pbtxt"
        }

        import urllib.request
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)

        for filename, url in model_urls.items():
            model_path = model_dir / filename
            if not model_path.exists():
                try:
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, model_path)
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")

    @staticmethod
    def setup_systemd_service():
        """Setup systemd service (Linux only)"""
        if sys.platform.startswith('linux'):
            service_content = """[Unit]
Description=Thermal Detection System
After=network.target

[Service]
Type=simple
User=thermal
WorkingDirectory=/opt/thermal-detection
ExecStart=/usr/local/bin/thermal-detection
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

            service_path = Path("/etc/systemd/system/thermal-detection.service")
            try:
                with open(service_path, 'w') as f:
                    f.write(service_content)
                print(f"Created systemd service: {service_path}")
                print("Run 'sudo systemctl enable thermal-detection' to enable autostart")
            except PermissionError:
                print("Note: Run as root to install systemd service")

    @classmethod
    def run_all(cls):
        """Run all post-installation commands"""
        print("\n" + "=" * 50)
        print("Running post-installation setup...")
        print("=" * 50)

        cls.create_directories()
        cls.download_models()
        cls.setup_systemd_service()

        print("\n" + "=" * 50)
        print("Installation completed successfully!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Configure your system: edit config/thermal_config.yaml")
        print("2. Test the installation: thermal-test")
        print("3. Start the system: thermal-detection")
        print("4. Access the dashboard: http://localhost:8000/dashboard")


# Custom install command
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Custom installation command"""

    def run(self):
        # Run standard installation
        install.run(self)

        # Run post-installation setup
        PostInstallCommand.run_all()


# Custom develop command
from setuptools.command.develop import develop


class CustomDevelopCommand(develop):
    """Custom development installation command"""

    def run(self):
        # Run standard development installation
        develop.run(self)

        # Run post-installation setup
        PostInstallCommand.run_all()


# Add custom commands
setup.cmdclass = {
    'install': CustomInstallCommand,
    'develop': CustomDevelopCommand,
}

if __name__ == "__main__":
    # Print installation information
    print("\n" + "=" * 60)
    print("THERMAL DETECTION SYSTEM INSTALLATION")
    print("=" * 60)
    print("Installing comprehensive thermal screening solution...")
    print("This package includes:")
    print("  • Real-time face detection (YOLO11, OpenCV)")
    print("  • Thermal temperature analysis")
    print("  • FastAPI web interface")
    print("  • WebSocket real-time updates")
    print("  • Comprehensive test suite")
    print("  • Configuration management")
    print("  • Performance monitoring")
    print("=" * 60)