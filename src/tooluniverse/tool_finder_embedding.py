import copy
import json
import gc
import os
from pathlib import Path
from .utils import get_md5
from .base_tool import BaseTool
from .tool_registry import register_tool


@register_tool("ToolFinderEmbedding")
class ToolFinderEmbedding(BaseTool):
    """
    A tool finder model that uses RAG (Retrieval-Augmented Generation) to find relevant tools
    based on user queries using semantic similarity search.

    This class leverages sentence transformers to encode tool descriptions and find the most
    relevant tools for a given query through embedding-based similarity matching.

    Attributes:
        rag_model_name (str): Name of the sentence transformer model for embeddings
        rag_model (SentenceTransformer): The loaded sentence transformer model
        tool_desc_embedding (torch.Tensor): Cached embeddings of tool descriptions
        tool_name (list): List of available tool names
        tool_embedding_path (str): Path to cached tool embeddings file
        special_tools_name (list): List of special tools to exclude from results
        tooluniverse: Reference to the tool universe containing all tools
    """

    def __init__(self, tool_config, tooluniverse):
        """
        Initialize the ToolFinderEmbedding with configuration and RAG model.

        Args:
            tool_config (dict): Configuration dictionary for the tool
        """
        super().__init__(tool_config)
        self.rag_model = None
        self.tool_desc_embedding = None
        self.tool_name = None
        self.tool_embedding_path = None
        toolfinder_model = tool_config["configs"].get("tool_finder_model")
        self.toolfinder_model = toolfinder_model
        # Get exclude tools from config, with fallback to default list
        self.exclude_tools = tool_config.get(
            "exclude_tools",
            tool_config.get("configs", {}).get(
                "exclude_tools", ["Tool_RAG", "Tool_Finder", "Finish", "CallAgent"]
            ),
        )
        self._dependencies_available = False
        self._dependency_error = None

        try:
            self.load_rag_model()
            print(
                f"Using toolfinder model: {toolfinder_model}, GPU is required for this model for fast speed..."
            )
            self.load_tool_desc_embedding(
                tooluniverse, exclude_names=self.exclude_tools
            )
            self._dependencies_available = True
        except ImportError as e:
            self._dependency_error = e
            # Don't raise - allow tool to be created but mark as unavailable
            import warnings

            warnings.warn(
                "ToolFinderEmbedding initialized without dependencies. "
                "Install missing packages with: pip install tooluniverse[embedding] "
                "or pip install tooluniverse[ml]",
                UserWarning,
                stacklevel=2,
            )

    def load_rag_model(self):
        """
        Load the sentence transformer model for RAG-based tool retrieval.

        Configures the model with appropriate sequence length and tokenizer settings
        for optimal performance in tool description encoding.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "ToolFinderEmbedding requires 'sentence-transformers' package. "
                "Install it with: pip install tooluniverse[embedding] or pip install tooluniverse[ml]"
            ) from e

        self.rag_model = SentenceTransformer(self.toolfinder_model)
        self.rag_model.max_seq_length = 4096
        self.rag_model.tokenizer.padding_side = "right"

    def load_tool_desc_embedding(
        self,
        tooluniverse,
        include_names=None,
        exclude_names=None,
        include_categories=None,
        exclude_categories=None,
    ):
        """
        Load or generate embeddings for tool descriptions from the tool universe.

        This method either loads cached embeddings from disk or generates new ones by encoding
        all tool descriptions. Embeddings are cached to disk for faster subsequent loads.
        Memory is properly cleaned up after embedding generation to avoid OOM issues.

        Args:
            tooluniverse: ToolUniverse instance containing all available tools
            include_names (list, optional): Specific tool names to include
            exclude_names (list, optional): Tool names to exclude
            include_categories (list, optional): Tool categories to include
            exclude_categories (list, optional): Tool categories to exclude
        """
        self.tooluniverse = tooluniverse
        print("Loading tool descriptions and embeddings...")
        self.tool_name, _ = tooluniverse.refresh_tool_name_desc(
            enable_full_desc=True,
            include_names=include_names,
            exclude_names=exclude_names,
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        )

        # Get filtered tools that match the tool_name list
        filtered_tools = []
        tool_name_set = set(self.tool_name)
        for tool in tooluniverse.all_tools:
            if tool["name"] in tool_name_set:
                filtered_tools.append(tool)

        augmented_tools = self._augment_with_persona(filtered_tools)
        all_tools_str = [
            json.dumps(each) for each in tooluniverse.prepare_tool_prompts(augmented_tools)
        ]
        md5_value = get_md5(str(all_tools_str))
        print("get the md value of tools:", md5_value)
        self.tool_embedding_path = (
            self.toolfinder_model.split("/")[-1] + "tool_embedding_" + md5_value + ".pt"
        )

        # Lazy import torch for loading/saving embeddings
        try:
            import torch
        except ImportError:
            raise ImportError(
                "ToolFinderEmbedding requires 'torch' package. "
                "Install it with: pip install tooluniverse[embedding] or pip install tooluniverse[ml]"
            ) from None

        try:
            self.tool_desc_embedding = torch.load(
                self.tool_embedding_path, weights_only=False
            )
            assert len(self.tool_desc_embedding) == len(self.tool_name), (
                "The number of tools in the tool_name list is not equal to the number of tool_desc_embedding."
            )
            print("\033[92mSuccessfully loaded cached embeddings.\033[0m")
        except (RuntimeError, AssertionError, OSError):
            self.tool_desc_embedding = None
            print("\033[92mInferring the tool_desc_embedding.\033[0m")

            # Generate embeddings
            self.tool_desc_embedding = self.rag_model.encode(
                all_tools_str, prompt="", normalize_embeddings=True
            )

            # Save embeddings to disk
            torch.save(self.tool_desc_embedding, self.tool_embedding_path)
            print(
                "\033[92mFinished inferring and saving the tool_desc_embedding.\033[0m"
            )

            # Clean up intermediate variables
            del all_tools_str

            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force CPU memory cleanup
            gc.collect()

            print(
                "\033[92mMemory cleanup completed. Embeddings are ready for use.\033[0m"
            )

    def _augment_with_persona(self, tools):
        persona_map = self._load_persona_map()
        if not persona_map:
            return tools
        augmented = copy.deepcopy(tools)
        for tool in augmented:
            name = tool.get("name")
            persona = persona_map.get(name, {})
            if not persona:
                continue
            identity = persona.get("identity", {})
            aliases = identity.get("aliases") or []
            tags = identity.get("domain_tags") or []
            description = identity.get("description") or ""
            extra = []
            if description:
                extra.append(description)
            if aliases:
                extra.append("aliases: " + ", ".join(aliases))
            if tags:
                extra.append("tags: " + ", ".join(tags))
            if extra:
                base = tool.get("description", "") or ""
                tool["description"] = base + "\n\npersona_context: " + " | ".join(extra)
        return augmented

    def _load_persona_map(self):
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

    def rag_infer(self, query, top_k=5):
        """
        Perform RAG inference to find the most relevant tools for a given query.

        Uses semantic similarity between the query embedding and pre-computed tool embeddings
        to identify the most relevant tools.

        Args:
            query (str): User query or description of desired functionality
            top_k (int, optional): Number of top tools to return. Defaults to 5.

        Returns
            list: List of top-k tool names ranked by relevance to the query

        Raises:
            ImportError: If dependencies are not available.
            SystemExit: If tool_desc_embedding is not loaded
        """
        if not self._dependencies_available:
            raise ImportError(
                "ToolFinderEmbedding requires dependencies. "
                "Install with: pip install tooluniverse[embedding] or "
                "pip install tooluniverse[ml]"
            ) from self._dependency_error

        # Lazy import torch (should already be imported, but for safety)
        try:
            import torch
        except ImportError:
            raise ImportError(
                "ToolFinderEmbedding requires 'torch' package. "
                "Install it with: pip install tooluniverse[embedding] or pip install tooluniverse[ml]"
            ) from None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        queries = [query]
        query_embeddings = self.rag_model.encode(
            queries, prompt="", normalize_embeddings=True
        )
        if self.tool_desc_embedding is None:
            print("No tool_desc_embedding")
            exit()
        scores = self.rag_model.similarity(query_embeddings, self.tool_desc_embedding)
        top_k = min(top_k, len(self.tool_name))
        top_k_indices = torch.topk(scores, top_k).indices.tolist()[0]
        top_k_tool_names = [self.tool_name[i] for i in top_k_indices]
        return top_k_tool_names

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

        This method either uses RAG inference to find tools based on a message or processes
        a list of pre-selected tool names. It filters out special tools and returns tool
        prompts suitable for use in agent workflows.

        Args:
            message (str, optional): Query message to find tools for. Required if picked_tool_names is None.
            picked_tool_names (list, optional): Pre-selected tool names to process. Required if message is None.
            rag_num (int, optional): Number of tools to return after filtering. Defaults to 5.
            return_call_result (bool, optional): If True, returns both prompts and tool names. Defaults to False.
            categories (list, optional): List of tool categories to filter by. Currently not implemented for embedding-based search.

        Returns
            str or tuple:
                - If return_call_result is False: Tool prompts as a formatted string
                - If return_call_result is True: Tuple of (tool_prompts, tool_names)

        Raises:
            AssertionError: If both message and picked_tool_names are None
        """
        extra_factor = 1.5  # Factor to retrieve more than rag_num
        if picked_tool_names is None:
            assert picked_tool_names is not None or message is not None
            picked_tool_names = self.rag_infer(
                message, top_k=int(rag_num * extra_factor)
            )

        picked_tool_names_no_special = []
        for tool in picked_tool_names:
            if tool not in self.exclude_tools:
                picked_tool_names_no_special.append(tool)
        picked_tool_names_no_special = picked_tool_names_no_special[:rag_num]
        picked_tool_names = picked_tool_names_no_special[:rag_num]

        picked_tools = self.tooluniverse.get_tool_specification_by_names(
            picked_tool_names
        )
        picked_tools_prompt = self.tooluniverse.prepare_tool_prompts(picked_tools)
        if return_call_result:
            return picked_tools_prompt, picked_tool_names
        return picked_tools_prompt

    def run(self, arguments):
        """
        Run the tool finder with given arguments following the standard tool interface.

        This is the main entry point for using ToolFinderEmbedding as a standard tool.
        It extracts parameters from the arguments dictionary and delegates to find_tools().

        Args:
            arguments (dict): Dictionary containing:
                - description (str, optional): Query message to find tools for (maps to 'message')
                - limit (int, optional): Number of tools to return (maps to 'rag_num'). Defaults to 5.
                - picked_tool_names (list, optional): Pre-selected tool names to process
                - return_call_result (bool, optional): Whether to return both prompts and names. Defaults to False.
                - categories (list, optional): List of tool categories to filter by
        """
        import copy

        arguments = copy.deepcopy(arguments)

        # Extract parameters from arguments with defaults
        message = arguments.get("description", None)
        rag_num = arguments.get("limit", 5)
        picked_tool_names = arguments.get("picked_tool_names", None)
        return_call_result = arguments.get("return_call_result", False)
        categories = arguments.get("categories", None)

        # Call the existing find_tools method
        return self.find_tools(
            message=message,
            picked_tool_names=picked_tool_names,
            rag_num=rag_num,
            return_call_result=return_call_result,
            categories=categories,
        )
