"""
run_tests.py
============
Automated Test Runner for the SMS Scam Detection System.
Executes all unit and integration tests and prints a clean report.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUTOMATED TEST SUITE: SMS SCAM DETECTION")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(BASE_DIR / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run : {result.testsRun}")
    print(f"Passed          : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures        : {len(result.failures)}")
    print(f"Errors          : {len(result.errors)}")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED SUCCESSFULLY! [OK]")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED! [FAILED]")
        sys.exit(1)
