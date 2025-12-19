"""
clinvar_search_variants

Search for variants in ClinVar database by gene name, condition, or variant ID. Returns variant i...
"""

from typing import Any, Optional, Callable
from ._shared_client import get_shared_client


def clinvar_search_variants(
    gene: Optional[str] = None,
    condition: Optional[str] = None,
    variant_id: Optional[str] = None,
    max_results: Optional[int] = 20,
    *,
    stream_callback: Optional[Callable[[str], None]] = None,
    use_cache: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Search for variants in ClinVar database by gene name, condition, or variant ID. Returns variant i...

    Parameters
    ----------
    gene : str
        Gene name or symbol (e.g., 'BRCA1', 'BRCA2')
    condition : str
        Disease or condition name (e.g., 'breast cancer', 'diabetes')
    variant_id : str
        ClinVar variant ID (e.g., '12345')
    max_results : int
        Maximum number of results to return
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
            "name": "clinvar_search_variants",
            "arguments": {
                "gene": gene,
                "condition": condition,
                "variant_id": variant_id,
                "max_results": max_results
            }
        },
        stream_callback=stream_callback,
        use_cache=use_cache,
        validate=validate
    )


__all__ = ["clinvar_search_variants"]
