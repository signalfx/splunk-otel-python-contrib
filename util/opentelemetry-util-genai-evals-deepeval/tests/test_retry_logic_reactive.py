
import unittest
from unittest.mock import MagicMock, patch
import os
import time

# Import modules to test (assuming python path is set correctly or running from project root)
# Note: Ensure paths are correct based on your directory structure
# Mock sys.modules to avoid real deepeval/openai dependencies during testing
import sys
from types import ModuleType

# Mock deepeval and openai to prevent import errors
sys.modules["deepeval"] = MagicMock()
sys.modules["deepeval.metrics"] = MagicMock()
sys.modules["deepeval.test_case"] = MagicMock()
sys.modules["deepeval.evaluate"] = MagicMock()
sys.modules["deepeval.evaluate.configs"] = MagicMock()
sys.modules["openai"] = MagicMock()

# Now import our target code
from opentelemetry.util.evaluator.deepeval import _run_with_retry, _is_rate_limit_error

class TestRetryLogic(unittest.TestCase):

    def test_is_rate_limit_error(self):
        """Test if rate limit errors are correctly identified"""
        # Case 1: Standard exceptions with keywords
        self.assertTrue(_is_rate_limit_error(Exception("Rate limit exceeded")))
        self.assertTrue(_is_rate_limit_error(Exception("Too many requests")))
        self.assertTrue(_is_rate_limit_error(Exception("HTTP 429 error")))
        
        # Case 2: Named exception types
        class RateLimitError(Exception): pass
        self.assertTrue(_is_rate_limit_error(RateLimitError("Boom")))
        
        # Case 3: Unrelated exceptions
        self.assertFalse(_is_rate_limit_error(ValueError("Invalid value")))
        self.assertFalse(_is_rate_limit_error(RuntimeError("Unknown error")))

    @patch("opentelemetry.util.evaluator.deepeval._run_deepeval")
    @patch("time.sleep") # Mock sleep to speed up tests
    def test_retry_success_after_failure(self, mock_sleep, mock_run_real):
        """Test: First two attempts fail (rate limit), third succeeds. Verify 3 retries."""
        
        # Simulating: 1st fail, 2nd fail, 3rd success
        mock_run_real.side_effect = [
            Exception("429 Rate limit"),
            Exception("429 Rate limit"),
            "Success"
        ]
        
        valid_logger = MagicMock()
        
        # Execution
        result = _run_with_retry(
            test_case="dummy_case", 
            metrics=[], 
            debug_log=valid_logger, 
            max_retries=3, 
            initial_backoff=0.01 # Short interval for testing
        )
        
        # Verify result
        self.assertEqual(result, "Success")
        
        # Verify call count: Should be 3 (2 failures + 1 success)
        self.assertEqual(mock_run_real.call_count, 3)
        print("\n✅ Test passed: Retried 3 times as expected and succeeded.")

    @patch("opentelemetry.util.evaluator.deepeval._run_deepeval")
    @patch("time.sleep")
    def test_retry_exhaustion(self, mock_sleep, mock_run_real):
        """Test: Always fails until max retries reached, then raises exception."""
        
        # Simulating: Always raising rate limit error
        mock_run_real.side_effect = Exception("429 Rate limit")
        
        with self.assertRaises(Exception) as cm:
            _run_with_retry(
                test_case="dummy_case", 
                metrics=[], 
                debug_log=MagicMock(), 
                max_retries=2, 
                initial_backoff=0.01
            )
        
        self.assertIn("429 Rate limit", str(cm.exception))
        
        # Verify call count: Should be 3 (Initial + Retry 1 + Retry 2) = 3 attempts
        # wait... retry count starts at 0.
        # loop 0: run (fail), sleep, retry_count=1
        # loop 1: run (fail), sleep, retry_count=2
        # loop 2: run (fail), sleep, retry_count=3
        # loop 3: > max_retries(2), raises
        # So actual calls are 3
        self.assertEqual(mock_run_real.call_count, 3) 
        print("\n✅ Test passed: Stopped after exhausting retries.")

    @patch("opentelemetry.util.evaluator.deepeval._run_deepeval")
    def test_no_retry_for_other_errors(self, mock_run_real):
        """Test: For non-rate-limit errors, should not retry, fail fast."""
        
        mock_run_real.side_effect = ValueError("Bad input")
        
        with self.assertRaises(ValueError):
            _run_with_retry(
                test_case="dummy_case", 
                metrics=[], 
                debug_log=MagicMock(), 
                max_retries=3
            )
            
        # Verify called only once
        self.assertEqual(mock_run_real.call_count, 1)
        print("\n✅ Test passed: Did not retry for non-rate-limit errors.")

if __name__ == "__main__":
    unittest.main()
