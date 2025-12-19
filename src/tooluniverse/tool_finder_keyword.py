"""
Keyword-based Tool Finder - An advanced keyword search tool for finding relevant tools.

This tool provides sophisticated keyword matching functionality using natural language
processing techniques including tokenization, stop word removal, stemming, and TF-IDF
scoring for improved relevance ranking. It serves as a robust search method when
AI-powered search methods are unavailable.
"""

import json
import re
import math
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
from .base_tool import BaseTool
from .tool_registry import register_tool


@register_tool("ToolFinderKeyword")
class ToolFinderKeyword(BaseTool):
    """
    Advanced keyword-based tool finder that uses sophisticated text processing and TF-IDF scoring.

    This class implements natural language processing techniques for tool discovery including:
    - Tokenization and normalization
    - Stop word removal
    - Basic stemming
    - TF-IDF relevance scoring
    - Semantic phrase matching

    The search operates by parsing user queries to extract key terms, processing them through
    NLP pipelines, and matching against pre-built indices of tool metadata for efficient
    and relevant tool discovery.
    """

    # Common English stop words to filter out
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "to",
        "was",
        "will",
        "with",
        "the",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "said",
        "each",
        "which",
        "their",
        "time",
        "up",
        "use",
        "your",
        "how",
        "all",
        "any",
        "can",
        "do",
        "get",
        "if",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "number",
        "no",
        "find",
        "long",
        "down",
        "day",
        "came",
        "made",
        "part",
    }

    # Simple stemming rules for common suffixes
    STEMMING_RULES = [
        ("ies", "y"),
        ("ied", "y"),
        ("ying", "y"),
        ("ing", ""),
        ("ly", ""),
        ("ed", ""),
        ("ies", "y"),
        ("ier", "y"),
        ("iest", "y"),
        ("s", ""),
        ("es", ""),
        ("er", ""),
        ("est", ""),
        ("tion", "t"),
        ("sion", "s"),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("ful", ""),
        ("less", ""),
        ("ous", ""),
        ("ive", ""),
        ("al", ""),
        ("ic", ""),
        ("ize", ""),
        ("ise", ""),
        ("ate", ""),
        ("fy", ""),
        ("ify", ""),
    ]

    def __init__(self, tool_config, tooluniverse=None):
        """
        Initialize the Advanced Keyword-based Tool Finder.

        Args:
            tool_config (dict): Configuration dictionary for the tool
            tooluniverse: Reference to the ToolUniverse instance containing all tools
        """
        super().__init__(tool_config)
        self.tooluniverse = tooluniverse

        # Extract configuration
        self.name = tool_config.get("name", "ToolFinderKeyword")
        self.description = tool_config.get(
            "description", "Advanced keyword-based tool finder"
        )

        # Tool filtering settings
        self.exclude_tools = tool_config.get(
            "exclude_tools",
            tool_config.get("configs", {}).get(
                "exclude_tools",
                [
                    "Tool_RAG",
                    "Tool_Finder",
                    "Finish",
                    "CallAgent",
                    "ToolFinderLLM",
                    "ToolFinderKeyword",
                ],
            ),
        )
        self.include_categories = tool_config.get("include_categories", None)
        self.exclude_categories = tool_config.get("exclude_categories", None)

        # Initialize tool index for TF-IDF scoring
        self._tool_index = None
        self._document_frequencies = None
        self._total_documents = 0

    def _tokenize_and_normalize(self, text: str) -> List[str]:
        """
        Tokenize text and apply normalization including stop word removal and stemming.

        Args:
            text (str): Input text to tokenize

        Returns
            List[str]: List of processed tokens
        """
        if not text:
            return []

        # Convert to lowercase and extract words (alphanumeric sequences)
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())

        # Remove stop words
        tokens = [token for token in tokens if token not in self.STOP_WORDS]

        # Apply basic stemming
        stemmed_tokens = []
        for token in tokens:
            stemmed = self._apply_stemming(token)
            if len(stemmed) > 2:  # Keep only meaningful terms
                stemmed_tokens.append(stemmed)

        return stemmed_tokens

    def _apply_stemming(self, word: str) -> str:
        """
        Apply basic stemming rules to reduce words to their root form.

        Args:
            word (str): Word to stem

        Returns
            str: Stemmed word
        """
        if len(word) <= 3:
            return word

        for suffix, replacement in self.STEMMING_RULES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)] + replacement

        return word

    def _build_persona_context(self, persona: Dict) -> str:
        if not persona:
            return ""
        identity = persona.get("identity", {})
        aliases = identity.get("aliases") or []
        tags = identity.get("domain_tags") or []
        description = identity.get("description") or ""
        parts = []
        if description:
            parts.append(description)
        if aliases:
            parts.append("aliases: " + " ".join(aliases))
        if tags:
            parts.append("tags: " + " ".join(tags))
        return " ".join(parts)

    def _load_persona_map(self) -> Dict[str, Dict]:
        if hasattr(self, "_persona_map"):
            return self._persona_map
        persona_dir = os.getenv("TOOLUNIVERSE_PERSONA_DIR", "web/personas")
        path = Path(persona_dir)
        if not path.exists():
            self._persona_map = {}
            return self._persona_map
        persona_map = {}
        for persona_file in path.glob("*.json"):
            try:
                data = json.loads(persona_file.read_text(encoding="utf-8"))
                name = data.get("name")
                if name:
                    persona_map[name] = data
            except Exception:
                continue
        self._persona_map = persona_map
        return self._persona_map

    def _extract_phrases(
        self, tokens: List[str], max_phrase_length: int = 3
    ) -> List[str]:
        """
        Extract meaningful phrases from tokens for better semantic matching.

        Args:
            tokens (List[str]): Tokenized words
            max_phrase_length (int): Maximum length of phrases to extract

        Returns
            List[str]: List of phrases and individual tokens
        """
        phrases = []

        # Add individual tokens
        phrases.extend(tokens)

        # Add bigrams and trigrams
        for length in range(2, min(max_phrase_length + 1, len(tokens) + 1)):
            for i in range(len(tokens) - length + 1):
                phrase = " ".join(tokens[i : i + length])
                phrases.append(phrase)

        return phrases

    def _build_tool_index(self, tools: List[Dict]) -> None:
        """
        Build TF-IDF index for all tools to enable efficient relevance scoring.

        Args:
            tools (List[Dict]): List of tool configurations
        """
        self._tool_index = {}
        term_doc_count = defaultdict(int)
        self._total_documents = 0

        persona_map = self._load_persona_map()
        for tool in tools:
            tool_name = tool.get("name", "")
            if tool_name in self.exclude_tools:
                continue

            persona_context = self._build_persona_context(persona_map.get(tool_name))
            # Combine tool metadata for indexing
            searchable_text = " ".join(
                [
                    tool.get("name", ""),
                    tool.get("description", ""),
                    tool.get("type", ""),
                    tool.get("category", ""),
                    # Include parameter names and descriptions
                    " ".join(self._extract_parameter_text(tool.get("parameter", {}))),
                    persona_context,
                ]
            )

            # Tokenize and extract phrases
            tokens = self._tokenize_and_normalize(searchable_text)
            phrases = self._extract_phrases(tokens)

            # Build term frequency map for this tool
            term_freq = Counter(phrases)
            self._tool_index[tool_name] = {
                "tool": tool,
                "terms": term_freq,
                "total_terms": len(phrases),
            }

            # Count document frequency for each term
            unique_terms = set(phrases)
            for term in unique_terms:
                term_doc_count[term] += 1

            self._total_documents += 1

        # Calculate document frequencies
        self._document_frequencies = dict(term_doc_count)

    def _extract_parameter_text(self, parameter_schema: Dict) -> List[str]:
        """
        Extract searchable text from parameter schema.

        Args:
            parameter_schema (Dict): Tool parameter schema

        Returns
            List[str]: List of text elements from parameters
        """
        text_elements = []

        if isinstance(parameter_schema, dict):
            properties = parameter_schema.get("properties", {})
            for prop_name, prop_info in properties.items():
                text_elements.append(prop_name)
                if isinstance(prop_info, dict):
                    desc = prop_info.get("description", "")
                    if desc:
                        text_elements.append(desc)

        return text_elements

    def _calculate_tfidf_score(self, query_terms: List[str], tool_name: str) -> float:
        """
        Calculate TF-IDF relevance score for a tool given query terms.

        Args:
            query_terms (List[str]): Processed query terms and phrases
            tool_name (str): Name of the tool to score

        Returns
            float: TF-IDF relevance score
        """
        if tool_name not in self._tool_index:
            return 0.0

        tool_data = self._tool_index[tool_name]
        tool_terms = tool_data["terms"]
        total_terms = tool_data["total_terms"]

        score = 0.0
        query_term_freq = Counter(query_terms)

        for term, query_freq in query_term_freq.items():
            if term in tool_terms:
                # Term Frequency (TF): frequency of term in tool / total terms in tool
                tf = tool_terms[term] / total_terms

                # Inverse Document Frequency (IDF): log(total docs / docs containing term)
                doc_freq = self._document_frequencies.get(term, 1)
                idf = math.log(self._total_documents / doc_freq)

                # TF-IDF score with query term frequency weighting
                score += tf * idf * math.log(1 + query_freq)

        return score

    def _calculate_exact_match_bonus(self, query: str, tool: Dict) -> float:
        """
        Calculate bonus score for exact matches in tool name or key phrases.

        Args:
            query (str): Original query string
            tool (Dict): Tool configuration

        Returns
            float: Exact match bonus score
        """
        query_lower = query.lower()
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()

        bonus = 0.0

        # Exact tool name match
        if query_lower in tool_name or tool_name in query_lower:
            bonus += 2.0

        # Exact phrase matches in description
        query_words = query_lower.split()
        if len(query_words) > 1:
            query_phrase = " ".join(query_words)
            if query_phrase in tool_desc:
                bonus += 1.5

        # Category or type exact matches
        tool_type = tool.get("type", "").lower()
        tool_category = tool.get("category", "").lower()

        if query_lower in tool_type or query_lower in tool_category:
            bonus += 1.0

        return bonus

    def find_tools(
        self,
        message=None,
        picked_tool_names=None,
        rag_num=5,
        return_call_result=False,
        categories=None,
    ):
        """
        Find relevant tools based on a message or pre-selected tool names.

        This method matches the interface of other tool finders to ensure
        seamless replacement. It uses keyword-based search instead of embedding similarity.

        Args:
            message (str, optional): Query message to find tools for. Required if picked_tool_names is None.
            picked_tool_names (list, optional): Pre-selected tool names to process. Required if message is None.
            rag_num (int, optional): Number of tools to return after filtering. Defaults to 5.
            return_call_result (bool, optional): If True, returns both prompts and tool names. Defaults to False.
            categories (list, optional): List of tool categories to filter by.

        Returns
            str or tuple:
                - If return_call_result is False: Tool prompts as a formatted string
                - If return_call_result is True: Tuple of (tool_prompts, tool_names)

        Raises:
            AssertionError: If both message and picked_tool_names are None
        """
        if picked_tool_names is None:
            assert picked_tool_names is not None or message is not None

            # Use keyword-based tool search (directly call JSON search to avoid recursion)
            search_result = self._run_json_search(
                {"description": message, "categories": categories, "limit": rag_num}
            )

            # Parse JSON result to extract tool names
            try:
                result_data = json.loads(search_result)
                if result_data.get("error"):
                    picked_tool_names = []
                else:
                    picked_tool_names = [
                        tool["name"] for tool in result_data.get("tools", [])
                    ]
            except json.JSONDecodeError:
                picked_tool_names = []

        # Filter out special tools (matching original behavior)
        picked_tool_names_no_special = []
        for tool in picked_tool_names:
            if tool not in self.exclude_tools:
                picked_tool_names_no_special.append(tool)
        picked_tool_names_no_special = picked_tool_names_no_special[:rag_num]
        picked_tool_names = picked_tool_names_no_special[:rag_num]

        # Get tool objects and prepare prompts (matching original behavior)
        picked_tools = self.tooluniverse.get_tool_specification_by_names(
            picked_tool_names
        )
        picked_tools_prompt = self.tooluniverse.prepare_tool_prompts(picked_tools)

        if return_call_result:
            return picked_tools_prompt, picked_tool_names
        return picked_tools_prompt

    def run(self, arguments):
        """
        Find tools using advanced keyword-based search with NLP processing and TF-IDF scoring.

        This method provides a unified interface compatible with other tool finders.

        Args:
            arguments (dict): Dictionary containing:
                - description (str): Search query string (unified parameter name)
                - categories (list, optional): List of categories to filter by
                - limit (int, optional): Maximum number of results to return (default: 10)
                - picked_tool_names (list, optional): Pre-selected tool names to process
                - return_call_result (bool, optional): Whether to return both prompts and names. Defaults to False.

        Returns
            str or tuple:
                - If return_call_result is False: Tool prompts as a formatted string
                - If return_call_result is True: Tuple of (tool_prompts, tool_names)
        """
        # Extract parameters for compatibility
        description = arguments.get("description", arguments.get("query", ""))
        limit = arguments.get("limit", 10)
        return_call_result = arguments.get("return_call_result", False)
        categories = arguments.get("categories", None)
        picked_tool_names = arguments.get("picked_tool_names", None)

        # If we have a unified interface call, delegate to find_tools method
        if return_call_result is not None:
            return self.find_tools(
                message=description,
                picked_tool_names=picked_tool_names,
                rag_num=limit,
                return_call_result=return_call_result,
                categories=categories,
            )

        # Otherwise use original JSON-based interface for backward compatibility
        return self._run_json_search(arguments)

    def _run_json_search(self, arguments):
        """
        Original JSON-based search implementation for backward compatibility.

        Args:
            arguments (dict): Search arguments

        Returns
            str: JSON string containing search results with relevance scores
        """
        try:
            # Extract arguments with unified parameter names
            query = arguments.get(
                "description", arguments.get("query", "")
            )  # Support both names for compatibility
            categories = arguments.get("categories", None)
            limit = arguments.get("limit", 10)

            if not query:
                return json.dumps(
                    {
                        "error": "Description parameter is required",
                        "query": query,
                        "tools": [],
                    },
                    indent=2,
                )

            # Ensure categories is None or a list (handle validation issue)
            if categories is not None and not isinstance(categories, list):
                categories = None

            # Get all tools from tooluniverse
            if not self.tooluniverse:
                return json.dumps(
                    {
                        "error": "ToolUniverse not available",
                        "query": query,
                        "tools": [],
                    },
                    indent=2,
                )

            all_tools = self.tooluniverse.return_all_loaded_tools()

            # Filter by categories if specified
            if categories:
                filtered_tools = self.tooluniverse.select_tools(
                    include_categories=categories
                )
            else:
                filtered_tools = all_tools

            # Build search index if not already built or if tools changed
            if self._tool_index is None or self._total_documents != len(
                [
                    t
                    for t in filtered_tools
                    if t.get("name", "") not in self.exclude_tools
                ]
            ):
                self._build_tool_index(filtered_tools)

            # Process query using NLP techniques
            query_tokens = self._tokenize_and_normalize(query)
            query_phrases = self._extract_phrases(query_tokens)

            if not query_tokens and not query_phrases:
                return json.dumps(
                    {
                        "error": "No meaningful search terms found in query",
                        "query": query,
                        "tools": [],
                    },
                    indent=2,
                )

            # Calculate relevance scores for all tools
            tool_scores = []

            for tool in filtered_tools:
                tool_name = tool.get("name", "")

                # Skip excluded tools
                if tool_name in self.exclude_tools:
                    continue

                # Apply category filters if specified
                tool_category = tool.get("category", "unknown")
                if (
                    self.include_categories
                    and tool_category not in self.include_categories
                ):
                    continue
                if self.exclude_categories and tool_category in self.exclude_categories:
                    continue

                # Calculate TF-IDF score
                tfidf_score = self._calculate_tfidf_score(query_phrases, tool_name)

                # Calculate exact match bonus
                exact_bonus = self._calculate_exact_match_bonus(query, tool)

                # Combined relevance score
                total_score = tfidf_score + exact_bonus

                # Only include tools with positive relevance
                if total_score > 0:
                    tool_info = {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "type": tool.get("type", ""),
                        "category": tool_category,
                        "parameters": tool.get("parameter", {}),
                        "required": tool.get("required", []),
                        "relevance_score": round(total_score, 4),
                        "tfidf_score": round(tfidf_score, 4),
                        "exact_match_bonus": round(exact_bonus, 4),
                    }
                    tool_scores.append(tool_info)

            # Sort by relevance score (highest first) and limit results
            tool_scores.sort(key=lambda x: x["relevance_score"], reverse=True)
            matching_tools = tool_scores[:limit]

            # Remove internal scoring details from final output
            for tool in matching_tools:
                tool.pop("tfidf_score", None)
                tool.pop("exact_match_bonus", None)

            return json.dumps(
                {
                    "query": query,
                    "search_method": "Advanced keyword matching (TF-IDF + NLP)",
                    "total_matches": len(matching_tools),
                    "categories_filtered": categories,
                    "processing_info": {
                        "query_tokens": len(query_tokens),
                        "query_phrases": len(query_phrases),
                        "indexed_tools": self._total_documents,
                    },
                    "tools": matching_tools,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Advanced keyword search error: {str(e)}",
                    "query": arguments.get("query", ""),
                    "tools": [],
                },
                indent=2,
            )


# # Tool configuration for ToolUniverse registration
# TOOL_CONFIG = {
#     "name": "ToolFinderKeyword",
#     "description": "Advanced keyword-based tool finder using NLP techniques, TF-IDF scoring, and semantic phrase matching for precise tool discovery",
#     "type": "tool_finder_keyword",
#     "category": "tool_finder",
#     "parameter": {
#         "type": "object",
#         "properties": {
#             "query": {
#                 "type": "string",
#                 "description": "Search query describing the desired functionality. Uses advanced NLP processing including tokenization, stop word removal, and stemming."
#             },
#             "categories": {
#                 "type": "array",
#                 "items": {"type": "string"},
#                 "description": "Optional list of tool categories to filter by"
#             },
#             "limit": {
#                 "type": "integer",
#                 "description": "Maximum number of tools to return, ranked by TF-IDF relevance score (default: 10)",
#                 "default": 10
#             }
#         },
#         "required": ["query"]
#     },
#     "configs": {
#         "exclude_tools": [
#             "Tool_RAG", "Tool_Finder", "Finish", "CallAgent",
#             "ToolFinderLLM", "ToolFinderKeyword"
#         ],
#         "features": [
#             "tokenization", "stop_word_removal", "stemming",
#             "phrase_extraction", "tfidf_scoring", "exact_match_bonus"
#         ]
#     }
# }
