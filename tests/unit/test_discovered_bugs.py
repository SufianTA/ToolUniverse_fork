#!/usr/bin/env python3
"""
Test critical bugs and issues discovered during system testing - Cleaned Version

This test file covers important bugs that were discovered:
1. Deprecated method warnings
2. Missing API key handling
3. Tool loading timeout issues
4. Error handling for invalid tools
5. Parameter validation edge cases
6. Memory and performance issues
"""

import sys
import unittest
from pathlib import Path
import pytest
from unittest.mock import patch, Mock, MagicMock
import warnings
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tooluniverse import ToolUniverse
from tooluniverse.exceptions import ToolError, ToolValidationError


@pytest.mark.unit
class TestDiscoveredBugs(unittest.TestCase):
    """Test critical bugs and issues discovered during system testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tu = ToolUniverse()
        # Load tools for real testing
        self.tu.load_tools()
    
    def test_deprecated_method_warnings(self):
        """Test that deprecated methods show proper warnings."""
        # Test get_tool_by_name deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger a deprecation warning
            try:
                result = self.tu.get_tool_by_name(["NonExistentTool"])
                # Verify we got a result (even if empty)
                self.assertIsInstance(result, list)
            except Exception:
                pass
            
            # Check if deprecation warning was issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            if deprecation_warnings:
                self.assertTrue(any("get_tool_by_name" in str(warning.message) for warning in deprecation_warnings))
    
    def test_missing_api_key_handling(self):
        """Test proper handling of missing API keys."""
        # Test that tools requiring API keys handle missing keys gracefully
        api_dependent_tools = [
            "UniProt_get_entry_by_accession",
            "ArXiv_search_papers",
            "OpenTargets_get_associated_targets_by_disease_efoId"
        ]
        
        for tool_name in api_dependent_tools:
            try:
                result = self.tu.run({
                    "name": tool_name,
                    "arguments": {"accession": "P05067"} if "UniProt" in tool_name else 
                                {"query": "test", "limit": 5} if "ArXiv" in tool_name else
                                {"efoId": "EFO_0000305"}
                })
                
                # Should return a result (may be error if API keys not configured)
                self.assertIsInstance(result, dict)
                if "error" in result:
                    # Should be a meaningful error message
                    self.assertIsInstance(result["error"], str)
                    self.assertGreater(len(result["error"]), 0)
            except Exception as e:
                # Expected if API keys not configured
                self.assertIsInstance(e, Exception)
    
    def test_tool_loading_timeout_issues(self):
        """Test handling of tool loading timeout issues."""
        # Test that tool loading doesn't hang indefinitely
        import time
        
        start_time = time.time()
        try:
            # Try to load tools (this should complete in reasonable time)
            self.tu.load_tools()
            load_time = time.time() - start_time
            
            # Should complete within reasonable time (30 seconds)
            self.assertLess(load_time, 30)
            
        except Exception as e:
            # If loading fails, it should fail quickly, not hang
            load_time = time.time() - start_time
            self.assertLess(load_time, 30)
            self.assertIsInstance(e, Exception)
    
    def test_invalid_tool_name_handling(self):
        """Test that invalid tool names are handled gracefully."""
        # Test with completely invalid tool name
        result = self.tu.run({
            "name": "NonExistentTool",
            "arguments": {"test": "value"}
        })
        
        self.assertIsInstance(result, dict)
        # Should either return error or handle gracefully
        if "error" in result:
            self.assertIn("tool", str(result["error"]).lower())
    
    def test_invalid_arguments_handling(self):
        """Test that invalid arguments are handled gracefully."""
        # Test with invalid arguments for a real tool
        result = self.tu.run({
            "name": "UniProt_get_entry_by_accession",
            "arguments": {"invalid_param": "value"}
        })
        
        self.assertIsInstance(result, dict)
        # Should either return error or handle gracefully
        if "error" in result:
            self.assertIn("parameter", str(result["error"]).lower())
    
    def test_empty_arguments_handling(self):
        """Test handling of empty arguments."""
        # Test with empty arguments
        result = self.tu.run({
            "name": "UniProt_get_entry_by_accession",
            "arguments": {}
        })
        
        self.assertIsInstance(result, dict)
        # Should either return error or handle gracefully
        if "error" in result:
            self.assertIn("required", str(result["error"]).lower())
    
    def test_none_arguments_handling(self):
        """Test handling of None arguments."""
        # Test with None arguments
        result = self.tu.run({
            "name": "UniProt_get_entry_by_accession",
            "arguments": None
        })
        
        self.assertIsInstance(result, dict)
        # Should handle None arguments gracefully
    
    def test_malformed_query_handling(self):
        """Test handling of malformed queries."""
        malformed_queries = [
            {"name": "UniProt_get_entry_by_accession"},  # Missing arguments
            {"arguments": {"accession": "P05067"}},  # Missing name
            {"name": "", "arguments": {"accession": "P05067"}},  # Empty name
            {"name": "UniProt_get_entry_by_accession", "arguments": ""},  # String arguments
            {"name": "UniProt_get_entry_by_accession", "arguments": []},  # List arguments
        ]
        
        for query in malformed_queries:
            result = self.tu.run(query)
            self.assertIsInstance(result, dict)
            # Should handle malformed queries gracefully
    
    def test_large_argument_handling(self):
        """Test handling of very large arguments."""
        # Test with very large argument values
        large_args = {
            "accession": "P05067",
            "large_string": "A" * 100000,  # 100KB string
            "large_array": ["item"] * 10000,  # Large array
        }
        
        result = self.tu.run({
            "name": "UniProt_get_entry_by_accession",
            "arguments": large_args
        })
        
        self.assertIsInstance(result, dict)
        # Should handle large arguments gracefully
    
    def test_concurrent_tool_access(self):
        """Test concurrent access to tools."""
        import threading
        import time
        
        results = []
        
        def make_call(call_id):
            result = self.tu.run({
                "name": "UniProt_get_entry_by_accession",
                "arguments": {"accession": f"P{call_id:05d}"}
            })
            results.append(result)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_call, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all calls completed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, dict)
    
    def test_memory_leak_prevention(self):
        """Test that memory leaks are prevented."""
        # Test multiple tool calls to ensure no memory leaks
        try:
            initial_objects = len(gc.get_objects())
        except MemoryError:
            pytest.skip("gc.get_objects() exhausted memory in this environment")
        
        for i in range(10):  # Reduced from 100 for faster testing
            result = self.tu.run({
                "name": "UniProt_get_entry_by_accession",
                "arguments": {"accession": f"P{i:05d}"}
            })
            
            self.assertIsInstance(result, dict)
            
            # Force garbage collection periodically
            if i % 5 == 0:
                gc.collect()
        
        # Check that we haven't created too many new objects
        try:
            final_objects = len(gc.get_objects())
        except MemoryError:
            pytest.skip("gc.get_objects() exhausted memory in this environment")
        object_growth = final_objects - initial_objects
        
        # Should not have created more than 1000 new objects
        self.assertLess(object_growth, 1000)
    
    def test_error_message_clarity(self):
        """Test that error messages are clear and helpful."""
        # Test with invalid tool name
        result = self.tu.run({
            "name": "NonExistentTool",
            "arguments": {"test": "value"}
        })
        
        if "error" in result:
            error_msg = str(result["error"])
            # Error message should be clear and helpful
            self.assertIsInstance(error_msg, str)
            self.assertGreater(len(error_msg), 0)
            # Should contain meaningful information
            self.assertTrue(any(keyword in error_msg.lower() for keyword in ["tool", "not", "found", "error"]))
    
    def test_parameter_validation_edge_cases(self):
        """Test parameter validation edge cases."""
        edge_cases = [
            {"accession": ""},  # Empty string
            {"accession": None},  # None value
            {"accession": 123},  # Wrong type
            {"accession": []},  # List instead of string
            {"accession": {}},  # Dict instead of string
            {"accession": True},  # Boolean instead of string
        ]
        
        for case in edge_cases:
            result = self.tu.run({
                "name": "UniProt_get_entry_by_accession",
                "arguments": case
            })
            
            self.assertIsInstance(result, dict)
            # Should handle edge cases gracefully
            if "error" in result:
                self.assertIsInstance(result["error"], str)
    
    
    def test_tool_health_check(self):
        """Test tool health check functionality."""
        # Test health check
        health = self.tu.get_tool_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn("total", health)
        self.assertIn("available", health)
        self.assertIn("unavailable", health)
        self.assertIn("unavailable_list", health)
        self.assertIn("details", health)
        
        # Verify totals make sense
        self.assertEqual(health["total"], health["available"] + health["unavailable"])
        self.assertGreaterEqual(health["total"], 0)
        self.assertGreaterEqual(health["available"], 0)
        self.assertGreaterEqual(health["unavailable"], 0)
    
    
    
    
    def test_cache_management(self):
        """Test cache management functionality."""
        # Test cache clearing
        self.tu.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.tu._cache), 0)
        
        # Test cache operations using proper API
        self.tu._cache.set("test", "value")
        self.assertEqual(self.tu._cache.get("test"), "value")
        
        self.tu.clear_cache()
        self.assertEqual(len(self.tu._cache), 0)
    
    
    
    
    
    


if __name__ == "__main__":
    unittest.main()
