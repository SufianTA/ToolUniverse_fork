"""
ghost_tool

A ghost tool that exists in code but not in JSON configs.
"""

from typing import Any, Optional, Callable
from ._shared_client import get_shared_client


def ghost_tool(
    msg: Optional[str] = None,
    *,
    stream_callback: Optional[Callable[[str], None]] = None,
    use_cache: bool = False,
    validate: bool = True,
) -> Any:
    """
    A ghost tool that exists in code but not in JSON configs.

    Parameters
    ----------
    msg : str
        
    stream_callback : Callable, optional
        Callback for streaming output
    use_cache : bool, default False
        Enable caching
    validate : bool, default True
        Validate parameters

    Returns
    -------
    Any
    """
    # Handle mutable defaults to avoid B006 linting error

    return get_shared_client().run_one_function(
        {
            "name": "ghost_tool",
            "arguments": {
                "msg": msg
            }
        },
        stream_callback=stream_callback,
        use_cache=use_cache,
        validate=validate
    )


__all__ = ["ghost_tool"]
