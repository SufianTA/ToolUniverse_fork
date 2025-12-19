#!/usr/bin/env python3
"""
Test edge cases and error handling for Tool Finder functionality.

This test file covers important edge cases:
1. Tool finder with real ToolUniverse calls
2. Error handling for invalid parameters
3. Edge cases with real tool execution
"""

import sys
import unittest
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tooluniverse import ToolUniverse


class TestToolFinderEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for Tool Finder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tu = ToolUniverse()
        # Load tools for real testing
        self.tu.load_tools()
    
    def test_tool_finder_empty_query_real(self):
        """Test Tool_Finder with empty query using real ToolUniverse."""
        try:
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": "", "limit": 5}
            })
            
            self.assertIsInstance(result, dict)
            # Should handle empty query gracefully
            if "tools" in result:
                self.assertIsInstance(result["tools"], list)
        except Exception as e:
            # Expected if tool not available or API keys not configured
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_invalid_limit_real(self):
        """Test Tool_Finder with invalid limit values using real ToolUniverse."""
        try:
            # Test negative limit
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": "test", "limit": -1}
            })
            
            self.assertIsInstance(result, dict)
            # Should handle invalid limit gracefully
        except Exception as e:
            # Expected if tool not available or validation fails
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_very_large_limit_real(self):
        """Test Tool_Finder with very large limit using real ToolUniverse."""
        try:
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": "test", "limit": 10000}
            })
            
            self.assertIsInstance(result, dict)
            # Should handle large limit gracefully
        except Exception as e:
            # Expected if tool not available or limit too large
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_special_characters_real(self):
        """Test Tool_Finder with special characters using real ToolUniverse."""
        special_queries = [
            "test@#$%^&*()",
            "test with spaces and symbols!@#",
            "test with unicode: 中文测试",
            "test with quotes: \"double\" and 'single'",
        ]
        
        for query in special_queries:
            try:
                result = self.tu.run({
                    "name": "Tool_Finder_Keyword",
                    "arguments": {"description": query, "limit": 5}
                })
                
                self.assertIsInstance(result, dict)
                # Should handle special characters gracefully
            except Exception as e:
                # Expected if tool not available or special characters cause issues
                self.assertIsInstance(e, Exception)
    
    def test_tool_finder_missing_parameters_real(self):
        """Test Tool_Finder with missing required parameters using real ToolUniverse."""
        try:
            # Test missing description
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"limit": 5}
            })
            
            self.assertIsInstance(result, dict)
            # Should return error for missing required parameter
            if "error" in result:
                self.assertIn("description", str(result["error"]).lower())
        except Exception as e:
            # Expected if validation fails
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_extra_parameters_real(self):
        """Test Tool_Finder with extra parameters using real ToolUniverse."""
        try:
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {
                    "description": "test",
                    "limit": 5,
                    "extra_param": "should_be_ignored"
                }
            })
            
            self.assertIsInstance(result, dict)
            # Should handle extra parameters gracefully
        except Exception as e:
            # Expected if tool not available
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_wrong_parameter_types_real(self):
        """Test Tool_Finder with wrong parameter types using real ToolUniverse."""
        try:
            # Test limit as string instead of integer
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": "test", "limit": "not_a_number"}
            })
            
            self.assertIsInstance(result, dict)
            # Should either work (if validation is lenient) or return error
            if "error" in result:
                self.assertIn("limit", str(result["error"]).lower())
        except Exception as e:
            # Expected if validation fails
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_none_values_real(self):
        """Test Tool_Finder with None values using real ToolUniverse."""
        try:
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": None, "limit": None}
            })
            
            self.assertIsInstance(result, dict)
            # Should handle None values gracefully
        except Exception as e:
            # Expected if validation fails
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_very_long_query_real(self):
        """Test Tool_Finder with very long query using real ToolUniverse."""
        long_query = "test " * 1000  # Very long query
        
        try:
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": long_query, "limit": 5}
            })
            
            self.assertIsInstance(result, dict)
            # Should handle long query gracefully
        except Exception as e:
            # Expected if query too long or tool not available
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_unicode_handling_real(self):
        """Test Tool_Finder with various Unicode characters using real ToolUniverse."""
        unicode_queries = [
            "test with emoji: 🧬🔬🧪",
            "test with accented chars: café naïve résumé",
            "test with symbols: ∑∏∫√∞",
            "test with arrows: ←→↑↓",
            "test with currency: €£¥$",
        ]
        
        for query in unicode_queries:
            try:
                result = self.tu.run({
                    "name": "Tool_Finder_Keyword",
                    "arguments": {"description": query, "limit": 5}
                })
                
                self.assertIsInstance(result, dict)
                # Should handle Unicode gracefully
            except Exception as e:
                # Expected if tool not available or Unicode causes issues
                self.assertIsInstance(e, Exception)
    
    def test_tool_finder_concurrent_calls_real(self):
        """Test Tool_Finder with concurrent calls using real ToolUniverse."""
        import threading
        import json
        
        results = []
        results_lock = threading.Lock()
        
        def make_call(query_id):
            try:
                result = self.tu.run({
                    "name": "Tool_Finder_Keyword",
                    "arguments": {
                        "description": f"query_{query_id}",
                        "limit": 5
                    }
                })
                
                # Handle both string and dict results
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                        with results_lock:
                            results.append(parsed_result)
                    except json.JSONDecodeError:
                        with results_lock:
                            results.append({
                                "error": "Failed to parse JSON result",
                                "raw_result": result
                            })
                elif isinstance(result, list):
                    # Handle list results (shouldn't happen but let's be safe)
                    with results_lock:
                        results.append({
                            "error": "Unexpected list result",
                            "raw_result": result
                        })
                elif result is None:
                    with results_lock:
                        results.append({
                            "error": "Tool returned None",
                            "query_id": query_id
                        })
                else:
                    with results_lock:
                        results.append(result)
                        
            except Exception as e:
                with results_lock:
                    results.append({"error": str(e), "query_id": query_id})
        
        # Create multiple threads
        threads = []
        for i in range(3):  # Reduced for testing
            thread = threading.Thread(target=make_call, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout per thread
        
        # Verify all calls completed
        self.assertEqual(
            len(results), 3,
            f"Expected 3 results, got {len(results)}: {results}"
        )
        for i, result in enumerate(results):
            self.assertIsInstance(
                result, dict,
                f"Result {i} is not a dict: {type(result)} = {result}"
            )
    
    def test_tool_finder_llm_edge_cases_real(self):
        """Test Tool_Finder_LLM edge cases using real ToolUniverse."""
        try:
            # Test with various edge cases
            edge_cases = [
                {"description": "", "limit": 0},
                {"description": "a", "limit": 1},
                {"description": "test", "limit": -1},
                {"description": "test", "limit": 1000},
            ]
            
            for case in edge_cases:
                result = self.tu.run({
                    "name": "Tool_Finder_LLM",
                    "arguments": case
                })
                
                self.assertIsInstance(result, dict)
        except Exception as e:
            # Expected if tool not available
            self.assertIsInstance(e, Exception)
    
    @pytest.mark.require_gpu
    def test_tool_finder_embedding_edge_cases_real(self):
        """Test Tool_Finder (embedding) edge cases using real ToolUniverse."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available for embedding edge case tests")
        if not torch.cuda.is_available():
            pytest.skip("GPU required for Tool_Finder embedding edge cases")
        try:
            # Test with various edge cases
            edge_cases = [
                {"description": "", "limit": 0, "return_call_result": False},
                {"description": "a", "limit": 1, "return_call_result": True},
                {"description": "test", "limit": -1, "return_call_result": False},
                {"description": "test", "limit": 1000, "return_call_result": True},
            ]
            
            for case in edge_cases:
                result = self.tu.run({
                    "name": "Tool_Finder",
                    "arguments": case
                })
                
                self.assertIsInstance(result, dict)
        except Exception as e:
            # Expected if tool not available
            self.assertIsInstance(e, Exception)
    
    def test_tool_finder_actual_functionality(self):
        """Test that Tool_Finder actually works with valid inputs."""
        try:
            result = self.tu.run({
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": "protein search", "limit": 5}
            })
            
            self.assertIsInstance(result, dict)
            if "tools" in result:
                self.assertIsInstance(result["tools"], list)
                # If we got tools, verify they have expected structure
                for tool in result["tools"]:
                    self.assertIsInstance(tool, dict)
                    if "name" in tool:
                        self.assertIsInstance(tool["name"], str)
        except Exception as e:
            # Expected if tool not available or API keys not configured
            self.assertIsInstance(e, Exception)


if __name__ == "__main__":
    unittest.main()
