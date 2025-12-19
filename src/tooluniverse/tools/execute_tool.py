"""
execute_tool

Execute a ToolUniverse tool directly with custom arguments. This is the primary way to run any to...
"""

from typing import Any, Optional, Callable
from ._shared_client import get_shared_client


def execute_tool(
    tool_name: str,
    arguments: Optional[dict[str, Any]] = None,
    *,
    stream_callback: Optional[Callable[[str], None]] = None,
    use_cache: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Execute a ToolUniverse tool directly with custom arguments. This is the primary way to run any to...

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute (e.g., 'example_tool_name')
    arguments : dict[str, Any]
        Tool arguments as a JSON object (dictionary), NOT a JSON string. Pass the par...
    stream_callback : Callable, optional
        Callback for streaming output
    use_cache : bool, default False
        Enable caching
    validate : bool, default True
        Validate parameters

    Returns
    -------
    dict[str, Any]
    """
    # Handle mutable defaults to avoid B006 linting error

    return get_shared_client().run_one_function(
        {
            "name": "execute_tool",
            "arguments": {
                "tool_name": tool_name,
                "arguments": arguments
            }
        },
        stream_callback=stream_callback,
        use_cache=use_cache,
        validate=validate
    )


__all__ = ["execute_tool"]
