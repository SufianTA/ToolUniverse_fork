#!/usr/bin/env python3
"""
Test stdio mode with hooks integration

This test file covers the integration of stdio mode and hooks:
1. stdio mode with hooks enabled
2. MCP protocol with hooks in stdio mode
3. Tool calls with summarization in stdio mode
4. Error handling in stdio + hooks mode
"""

import pytest
import subprocess
import json
import time
import os
import sys
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

SUBPROCESS_ENV = os.environ.copy()
SUBPROCESS_ENV["PYTHONIOENCODING"] = "utf-8"
SUBPROCESS_ENV["PYTHONUTF8"] = "1"


def _start_stderr_reader(process):
    lines = []

    def _reader():
        try:
            for line in process.stderr:
                lines.append(line)
        except Exception:
            return

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return lines


@pytest.mark.integration
@pytest.mark.stdio
@pytest.mark.hooks
class TestStdioHooksIntegration:
    """Test stdio mode with hooks integration"""

    def test_stdio_with_hooks_handshake(self):
        """Test MCP handshake in stdio mode with hooks enabled"""
        # Start server in subprocess with hooks
        process = subprocess.Popen(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from tooluniverse.smcp_server import run_stdio_server
import os
os.environ['TOOLUNIVERSE_STDIO_MODE'] = '1'
sys.argv = ['tooluniverse-smcp-stdio', '--hooks']
run_stdio_server()
"""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=SUBPROCESS_ENV,
            bufsize=1
        )
        stderr_lines = _start_stderr_reader(process)
        
        try:
            # Wait for server to start (hooks take longer)
            time.sleep(8)
            
            # Step 1: Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read response
            response = process.stdout.readline()
            assert response.strip()
            
            # Parse response
            response_data = json.loads(response.strip())
            assert "result" in response_data
            assert response_data["result"]["protocolVersion"] == "2024-11-05"
            
            # Step 2: Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            
            time.sleep(2)
            
            # Step 3: List tools
            list_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            process.stdin.write(json.dumps(list_request) + "\n")
            process.stdin.flush()
            
            # Read tools list response
            response = process.stdout.readline()
            assert response.strip()
            
            # Parse response
            response_data = json.loads(response.strip())
            assert "result" in response_data
            assert "tools" in response_data["result"]
            
            # Check that hook tools are present
            tool_names = [tool["name"] for tool in response_data["result"]["tools"]]
            assert "ToolOutputSummarizer" in tool_names
            assert "OutputSummarizationComposer" in tool_names
            
        finally:
            # Clean up
            process.terminate()
            process.wait(timeout=10)

    def test_stdio_tool_call_with_hooks(self):
        """Test tool call in stdio mode with hooks enabled"""
        # Start server in subprocess with hooks
        process = subprocess.Popen(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from tooluniverse.smcp_server import run_stdio_server
import os
os.environ['TOOLUNIVERSE_STDIO_MODE'] = '1'
sys.argv = ['tooluniverse-smcp-stdio', '--hooks']
run_stdio_server()
"""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=SUBPROCESS_ENV,
            bufsize=1
        )
        stderr_lines = _start_stderr_reader(process)
        
        try:
            # Wait for server to start
            time.sleep(8)
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read init response
            response = process.stdout.readline()
            assert response.strip()
            
            # Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            
            time.sleep(2)
            
            # Call a tool that might generate long output
            tool_call_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "OpenTargets_get_target_gene_ontology_by_ensemblID",
                    "arguments": json.dumps({"ensemblId": "ENSG00000012048"})
                }
            }
            process.stdin.write(json.dumps(tool_call_request) + "\n")
            process.stdin.flush()
            
            # Read tool call response (this might take a while with hooks)
            response = process.stdout.readline()
            assert response.strip()
            
            # Parse response
            response_data = json.loads(response.strip())
            assert "result" in response_data or "error" in response_data
            
            # If successful, check if it's summarized
            if "result" in response_data:
                result_content = response_data["result"].get("content", [])
                if result_content:
                    text_content = result_content[0].get("text", "")
                    # Check if it's a summary (shorter than typical full output)
                    if len(text_content) < 10000:  # Typical full output is much longer
                        assert "summary" in text_content.lower() or "摘要" in text_content.lower()
            
        finally:
            # Clean up
            process.terminate()
            process.wait(timeout=15)

    def test_stdio_hooks_error_handling(self):
        """Test error handling in stdio mode with hooks"""
        # Start server in subprocess with hooks
        process = subprocess.Popen(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from tooluniverse.smcp_server import run_stdio_server
import os
os.environ['TOOLUNIVERSE_STDIO_MODE'] = '1'
sys.argv = ['tooluniverse-smcp-stdio', '--hooks']
run_stdio_server()
"""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=SUBPROCESS_ENV,
            bufsize=1
        )
        stderr_lines = _start_stderr_reader(process)
        
        try:
            # Wait for server to start
            time.sleep(8)
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read init response
            response = process.stdout.readline()
            assert response.strip()
            
            # Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            
            time.sleep(2)
            
            # Call a non-existent tool
            tool_call_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "NonExistentTool",
                    "arguments": "{}"
                }
            }
            process.stdin.write(json.dumps(tool_call_request) + "\n")
            process.stdin.flush()
            
            # Read error response
            response = process.stdout.readline()
            assert response.strip()
            
            # Parse response - should be an error
            response_data = json.loads(response.strip())
            assert "error" in response_data
            
        finally:
            # Clean up
            process.terminate()
            process.wait(timeout=10)

    def test_stdio_hooks_performance(self):
        """Test performance of stdio mode with hooks"""
        # Start server in subprocess with hooks
        process = subprocess.Popen(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from tooluniverse.smcp_server import run_stdio_server
import os
os.environ['TOOLUNIVERSE_STDIO_MODE'] = '1'
sys.argv = ['tooluniverse-smcp-stdio', '--hooks']
run_stdio_server()
"""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=SUBPROCESS_ENV,
            bufsize=1
        )
        stderr_lines = _start_stderr_reader(process)
        
        try:
            # Wait for server to start
            start_time = time.time()
            time.sleep(8)
            startup_time = time.time() - start_time
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read init response
            response = process.stdout.readline()
            assert response.strip()
            
            # Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            
            time.sleep(2)
            
            # Call a simple tool to test response time
            tool_call_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_server_info",
                    "arguments": "{}"
                }
            }
            
            call_start_time = time.time()
            process.stdin.write(json.dumps(tool_call_request) + "\n")
            process.stdin.flush()
            
            # Read response
            response = process.stdout.readline()
            call_end_time = time.time()
            
            call_time = call_end_time - call_start_time
            
            # Should complete within reasonable time
            assert call_time < 30  # Should be much faster
            assert response.strip()
            
        finally:
            # Clean up
            process.terminate()
            process.wait(timeout=10)

    def test_stdio_hooks_logging_separation(self):
        """Test that logs and JSON responses are properly separated in stdio mode with hooks"""
        # Start server in subprocess with hooks
        process = subprocess.Popen(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from tooluniverse.smcp_server import run_stdio_server
import os
os.environ['TOOLUNIVERSE_STDIO_MODE'] = '1'
sys.argv = ['tooluniverse-smcp-stdio', '--hooks']
run_stdio_server()
"""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=SUBPROCESS_ENV,
            bufsize=1
        )
        stderr_lines = _start_stderr_reader(process)
        
        try:
            # Wait for server to start
            time.sleep(8)
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read response - should be valid JSON
            response = process.stdout.readline()
            assert response.strip()
            
            # Try to parse as JSON - should succeed
            response_data = json.loads(response.strip())
            assert "jsonrpc" in response_data
            assert response_data["jsonrpc"] == "2.0"
            
            # Check that stderr contains logs (not stdout)
            time.sleep(1)
            assert stderr_lines  # Should contain logs
            
        finally:
            # Clean up
            process.terminate()
            process.wait(timeout=10)

    def test_stdio_hooks_multiple_tool_calls(self):
        """Test multiple tool calls in stdio mode with hooks"""
        # Start server in subprocess with hooks
        process = subprocess.Popen(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from tooluniverse.smcp_server import run_stdio_server
import os
os.environ['TOOLUNIVERSE_STDIO_MODE'] = '1'
sys.argv = ['tooluniverse-smcp-stdio', '--hooks']
run_stdio_server()
"""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=SUBPROCESS_ENV,
            bufsize=1
        )
        stderr_lines = _start_stderr_reader(process)
        
        try:
            # Wait for server to start
            time.sleep(8)
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read init response
            response = process.stdout.readline()
            assert response.strip()
            
            # Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            
            time.sleep(2)
            
            # Make multiple tool calls
            for i in range(3):
                tool_call_request = {
                    "jsonrpc": "2.0",
                    "id": i + 2,
                    "method": "tools/call",
                    "params": {
                        "name": "get_server_info",
                        "arguments": "{}"
                    }
                }
                process.stdin.write(json.dumps(tool_call_request) + "\n")
                process.stdin.flush()
                
                # Read response
                response = process.stdout.readline()
                assert response.strip()
                
                # Parse response
                response_data = json.loads(response.strip())
                assert "result" in response_data or "error" in response_data
            
        finally:
            # Clean up
            process.terminate()
            process.wait(timeout=15)
