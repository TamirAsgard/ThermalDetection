#!/usr/bin/env python3
"""
Comprehensive test runner for the Thermal Detection System
"""

import sys
import os
import unittest
import asyncio
import argparse
import time
from pathlib import Path
import importlib.util
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


class ThermalTestRunner:
    """Test runner for thermal detection system"""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0

    def discover_tests(self, pattern="test_*.py"):
        """Discover all test files"""
        test_files = list(self.test_dir.glob(pattern))
        return [f.stem for f in test_files if f.is_file()]

    def run_test_module(self, module_name, verbosity=2):
        """Run tests from a specific module"""
        print(f"\n{'=' * 60}")
        print(f"Running tests from: {module_name}")
        print(f"{'=' * 60}")

        try:
            # Import the test module
            spec = importlib.util.spec_from_file_location(
                module_name,
                self.test_dir / f"{module_name}.py"
            )
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)

            # Run tests
            if hasattr(test_module, 'AsyncTestRunner'):
                # Use custom async test runner if available
                runner = test_module.AsyncTestRunner(verbosity=verbosity)
            else:
                runner = unittest.TextTestRunner(verbosity=verbosity)

            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()

            # Store results
            self.results[module_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': result.wasSuccessful(),
                'duration': end_time - start_time
            }

            self.total_tests += result.testsRun
            self.total_failures += len(result.failures)
            self.total_errors += len(result.errors)

            return result.wasSuccessful()

        except Exception as e:
            print(f"Error running {module_name}: {e}")
            self.results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False,
                'duration': 0,
                'error': str(e)
            }
            self.total_errors += 1
            return False

    def run_all_tests(self, verbosity=2, pattern="test_*.py"):
        """Run all discovered tests"""
        test_modules = self.discover_tests(pattern)

        if not test_modules:
            print("No test files found!")
            return False

        print(f"Discovered {len(test_modules)} test modules:")
        for module in test_modules:
            print(f"  - {module}")

        print(f"\n{'=' * 60}")
        print("STARTING THERMAL DETECTION SYSTEM TESTS")
        print(f"{'=' * 60}")

        all_success = True

        for module in test_modules:
            success = self.run_test_module(module, verbosity)
            if not success:
                all_success = False

        self.print_summary()
        return all_success

    def run_specific_tests(self, test_names, verbosity=2):
        """Run specific test modules"""
        all_success = True

        for test_name in test_names:
            if not test_name.startswith('test_'):
                test_name = f"test_{test_name}"

            success = self.run_test_module(test_name, verbosity)
            if not success:
                all_success = False

        self.print_summary()
        return all_success

    def print_summary(self):
        """Print test results summary"""
        print(f"\n{'=' * 60}")
        print("TEST RESULTS SUMMARY")
        print(f"{'=' * 60}")

        for module, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = f"{result['duration']:.2f}s"

            print(f"{status} {module:<25} "
                  f"Tests: {result['tests_run']:<3} "
                  f"Failures: {result['failures']:<3} "
                  f"Errors: {result['errors']:<3} "
                  f"Time: {duration}")

            if 'error' in result:
                print(f"     Error: {result['error']}")

        print(f"\n{'=' * 60}")
        success_rate = ((self.total_tests - self.total_failures - self.total_errors) /
                        max(self.total_tests, 1)) * 100

        print(f"TOTAL TESTS:    {self.total_tests}")
        print(f"FAILURES:       {self.total_failures}")
        print(f"ERRORS:         {self.total_errors}")
        print(f"SUCCESS RATE:   {success_rate:.1f}%")

        if self.total_failures == 0 and self.total_errors == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("💥 SOME TESTS FAILED!")

        print(f"{'=' * 60}")


def run_coverage_tests():
    """Run tests with coverage analysis"""
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()

        # Run tests
        runner = ThermalTestRunner()
        success = runner.run_all_tests(verbosity=1)

        cov.stop()
        cov.save()

        print("\n" + "=" * 60)
        print("COVERAGE REPORT")
        print("=" * 60)
        cov.report()

        return success

    except ImportError:
        print("Coverage package not installed. Install with: pip install coverage")
        return False


def create_test_config():
    """Create a test configuration for isolated testing"""
    from config.settings import ThermalConfig

    # Mock configuration to avoid file dependencies
    config = ThermalConfig()
    config.development.demo_mode = True
    config.camera.thermal_device_type = "webcam"
    config.temperature.fever_threshold = 37.5
    config.detection.face_confidence_threshold = 0.5

    return config


def setup_test_environment():
    """Setup test environment with mocks and patches"""
    # Mock OpenCV to avoid camera dependencies
    opencv_mock = patch('cv2.VideoCapture')
    opencv_mock.start()

    # Mock file operations for config
    config_mock = patch('pathlib.Path.exists', return_value=False)
    config_mock.start()

    return [opencv_mock, config_mock]


def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="Thermal Detection System Test Runner")
    parser.add_argument(
        "--module", "-m",
        nargs="+",
        help="Specific test modules to run (e.g., face_detector temperature_analyzer)"
    )
    parser.add_argument(
        "--pattern", "-p",
        default="test_*.py",
        help="Test file pattern (default: test_*.py)"
    )
    parser.add_argument(
        "--verbosity", "-v",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Test output verbosity (0=minimal, 1=normal, 2=verbose)"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage analysis"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick tests only (skip integration tests)"
    )

    args = parser.parse_args()

    # Setup test environment
    mocks = setup_test_environment()

    try:
        if args.coverage:
            success = run_coverage_tests()
        else:
            runner = ThermalTestRunner()

            if args.module:
                success = runner.run_specific_tests(args.module, args.verbosity)
            else:
                pattern = "test_*_unit.py" if args.quick else args.pattern
                success = runner.run_all_tests(args.verbosity, pattern)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    finally:
        # Cleanup mocks
        for mock in mocks:
            mock.stop()


if __name__ == "__main__":
    main()


# Additional test utilities
class TestUtilities:
    """Utility functions for tests"""

    @staticmethod
    def create_test_image(width=640, height=480, channels=3):
        """Create a test image for face detection"""
        import numpy as np
        if channels == 3:
            return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
        else:
            return np.random.randint(0, 255, (height, width), dtype=np.uint8)

    @staticmethod
    def create_test_thermal_data(width=640, height=480, temp_range=(4.0, 5.0)):
        """Create test thermal data"""
        import numpy as np
        return np.random.uniform(temp_range[0], temp_range[1], (height, width)).astype(np.float32)

    @staticmethod
    def create_mock_face_detection(x=200, y=150, width=200, height=200, confidence=0.85):
        """Create a mock face detection"""
        from src.core.face_detector import FaceDetection
        return FaceDetection(x=x, y=y, width=width, height=height, confidence=confidence)

    @staticmethod
    def create_mock_temperature_reading(temp=36.5, is_fever=False):
        """Create a mock temperature reading"""
        from src.core.temperature_analyzer import TemperatureReading
        from datetime import datetime
        return TemperatureReading(
            avg_temperature=temp,
            max_temperature=temp + 0.5,
            min_temperature=temp - 0.5,
            pixel_count=100,
            confidence=0.9,
            timestamp=datetime.now(),
            is_fever=is_fever
        )


# Performance test utilities
class PerformanceTestMixin:
    """Mixin for performance testing"""

    def time_function(self, func, *args, **kwargs):
        """Time function execution"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    def assert_performance(self, func, max_time, *args, **kwargs):
        """Assert function completes within time limit"""
        result, duration = self.time_function(func, *args, **kwargs)
        self.assertLess(duration, max_time,
                        f"Function took {duration:.3f}s, expected < {max_time}s")
        return result