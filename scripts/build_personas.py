import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _default_persona(tool: Dict, timestamp: str) -> Dict:
    name = tool.get("name", "unknown_tool")
    labels = list(tool.get("label", []) or [])
    # Fold tags, category, and lab into domain tags for richer coverage
    tags = tool.get("tags") or []
    labels.extend(tags)
    category = tool.get("category")
    if category and category not in labels:
        labels.append(category)
    lab = tool.get("lab")
    if lab and lab not in labels:
        labels.append(lab)
    # Prefer the ToolUniverse schema field, fallback to legacy type
    role = tool.get("toolType") or tool.get("type") or "unknown"
    description = tool.get("description", "")
    detailed_description = tool.get("detailed_description", "")
    full_description = (
        f"{description}\n\n{detailed_description}".strip()
        if detailed_description
        else description
    )

    # Capabilities/limitations mapped into persona fields
    capabilities = tool.get("capabilities") or []
    if isinstance(capabilities, dict):
        capabilities = list(capabilities.values())
    limitations = tool.get("limitations") or []
    if isinstance(limitations, dict):
        limitations = list(limitations.values())

    return {
        "name": name,
        "identity": {
            "canonical_name": name,
            "aliases": tool.get("aliases", []),
            "domain_tags": labels,
            "role": role,
            "description": full_description,
            "capabilities": capabilities,
        },
        "contract": {
            "parameter_schema": tool.get("parameter", {}),
            "return_schema": tool.get("return_schema", {}),
            "preconditions": [],
            "postconditions": [],
            "failure_modes": [],
        },
        "telemetry": {
            "success_rate": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "empty_rate": None,
        },
        "domain_validity": {"strong": [], "weak": []},
        "safety": {
            "hard_constraints": [],
            "misuse_cases": limitations,
        },
        "provenance": {
            "tool_version": tool.get("version") or "unknown",
            "data_source_version": tool.get("data_source_version") or "unknown",
            "last_updated": timestamp,
        },
        "embeddings": {},
        "relationships": {
            "used_with": tool.get("related_tools", []),
            "commonly_follows": [],
            "validates_output_of": [],
            "similar_tools": [],
        },
        "workflows": {"observed": []},
        "mcp_call": {
            "name": name,
            "argument_schema": tool.get("parameter", {}),
        },
    }


def _build_embeddings(
    model,
    tool: Dict,
    persona: Dict,
    embedding_dir: Path,
    normalize: bool = True,
) -> (Dict, Dict):
    try:
        import torch
    except ImportError:
        # Skip embedding computation if torch is unavailable
        return persona, {}

    name = persona["name"]
    description = tool.get("description", "") or ""
    detailed = tool.get("detailed_description", "") or ""
    capabilities = tool.get("capabilities") or []
    if isinstance(capabilities, dict):
        capabilities = list(capabilities.values())
    limitations = tool.get("limitations") or []
    if isinstance(limitations, dict):
        limitations = list(limitations.values())
    texts = {
        # Purpose blends short and detailed descriptions for better semantic coverage
        "purpose": "\n\n".join([t for t in [description, detailed] if t]).strip(),
        "domain": " ".join(
            [
                description,
                detailed,
                " ".join(persona["identity"]["domain_tags"]),
                tool.get("category", "") or "",
                " ".join(capabilities),
                " ".join(limitations),
            ]
        ),
        "io_structure": json.dumps(
            {
                "parameter": tool.get("parameter", {}),
                "return_schema": tool.get("return_schema", {}),
            },
            sort_keys=True,
        ),
    }

    vectors: Dict[str, torch.Tensor] = {}
    for key, text in texts.items():
        vec = model.encode([text], normalize_embeddings=normalize)
        tensor = torch.tensor(vec[0])
        vectors[key] = tensor
        file_path = embedding_dir.joinpath(f"{name}_{key}.pt")
        persona["embeddings"][key] = {
            "model": getattr(model, "model_card", None) or getattr(model, "model_id", None) or str(model),
            "dim": tensor.shape[0],
            "path": str(file_path.as_posix()),
        }

    for key, tensor in vectors.items():
        torch.save(tensor, file_path := embedding_dir.joinpath(f"{name}_{key}.pt"))

    return persona, vectors


def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to build personas. "
            "Install with: pip install tooluniverse[embedding]"
        ) from exc

    model = SentenceTransformer(model_name)
    model.max_seq_length = 4096
    model.tokenizer.padding_side = "right"
    return model


def _load_tools(tool_file: Path) -> List[Dict]:
    data = json.loads(tool_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "tools" in data and isinstance(data["tools"], list):
            return data["tools"]
        return list(data.values())
    return []


def build_personas(
    model_name: str,
    tool_filter: Optional[List[str]],
    persona_dir: Path,
    embedding_dir: Path,
    graph_path: Path,
    tool_file: Path,
    rich_out: Path,
) -> None:
    model = None
    try:
        model = _load_model(model_name)
    except Exception:
        # Fall back to no-embedding mode if model cannot load
        model = None
    tools = _load_tools(tool_file)

    _ensure_dir(persona_dir)
    _ensure_dir(embedding_dir)
    graph_nodes: List[Dict] = []
    rich_catalog: List[Dict] = []
    persona_records: List[Dict] = []
    purpose_vectors = []
    tool_names: List[str] = []
    purpose_corpus: List[str] = []

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    skip_types = {"MCPAutoLoaderTool", "MCPClientTool"}

    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        if tool.get("type") in skip_types:
            continue
        if tool_filter and name not in tool_filter:
            continue

        persona = _default_persona(tool, timestamp)

        # Preserve existing embeddings metadata if present
        existing_path = persona_dir.joinpath(f"{name}.json")
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and existing.get("embeddings"):
                    persona["embeddings"] = existing.get("embeddings", {})
            except Exception:
                pass

        vectors = {}
        if model is not None:
            persona, vectors = _build_embeddings(model, tool, persona, embedding_dir)

        # Track for later similarity computation and final writes
        persona_records.append(
            {
                "tool": tool,
                "persona": persona,
                "path": persona_dir.joinpath(f"{name}.json"),
            }
        )
        tool_names.append(name)
        if "purpose" in vectors:
            purpose_vectors.append(vectors["purpose"])
        purpose_corpus.append(persona["identity"]["description"])

        graph_nodes.append({"id": name})
    # Derive similarity-based relationships to seed the graph
    graph_links: List[Dict] = []
    if purpose_vectors:
        try:
            import torch

            purpose_matrix = torch.stack(purpose_vectors)
            sim_matrix = torch.matmul(purpose_matrix, purpose_matrix.T)
            top_k = 5
            threshold = 0.4
            for i, name in enumerate(tool_names):
                sims = sim_matrix[i].clone()
                sims[i] = -1.0  # exclude self
                values, indices = torch.topk(
                    sims, k=min(top_k, len(tool_names) - 1), dim=0
                )
                similar_entries = []
                for score, j in zip(values.tolist(), indices.tolist()):
                    if score < threshold:
                        continue
                    target_name = tool_names[j]
                    similar_entries.append({"tool": target_name, "score": score})
                    graph_links.append(
                        {
                            "source": name,
                            "target": target_name,
                            "type": "similar_purpose",
                            "score": score,
                        }
                    )
                persona_records[i]["persona"]["relationships"]["similar_tools"] = (
                    similar_entries
                )
        except Exception:
            purpose_vectors = []

    if not purpose_vectors and purpose_corpus:
        # Fallback: build similarity edges using TF-IDF cosine on descriptions
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            TfidfVectorizer = None

        if TfidfVectorizer:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform(purpose_corpus)
            sim_matrix = cosine_similarity(tfidf)
            top_k = 5
            threshold = 0.2
            for i, name in enumerate(tool_names):
                sims = sim_matrix[i]
                # get top-k excluding self
                ranked = sorted(
                    [(j, s) for j, s in enumerate(sims) if j != i],
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_k]
                similar_entries = []
                for j, score in ranked:
                    if score < threshold:
                        continue
                    target_name = tool_names[j]
                    similar_entries.append({"tool": target_name, "score": float(score)})
                    graph_links.append(
                        {
                            "source": name,
                            "target": target_name,
                            "type": "similar_purpose",
                            "score": float(score),
                        }
                    )
                persona_records[i]["persona"]["relationships"]["similar_tools"] = (
                    similar_entries
                )

    # Write personas and rich catalog with updated relationships
    for record in persona_records:
        persona = record["persona"]
        tool = record["tool"]
        path = record["path"]

        for edge in persona["relationships"]["used_with"]:
            graph_links.append({"source": tool["name"], "target": edge, "type": "used_with"})
        for edge in persona["relationships"]["commonly_follows"]:
            graph_links.append({"source": edge, "target": tool["name"], "type": "commonly_follows"})
        for edge in persona["relationships"]["validates_output_of"]:
            graph_links.append({"source": edge, "target": tool["name"], "type": "validates_output_of"})

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(persona, ensure_ascii=True, indent=2), encoding="utf-8")

        rich_entry = tool.copy()
        rich_entry["persona"] = {
            "identity": persona.get("identity"),
            "contract": persona.get("contract"),
            "telemetry": persona.get("telemetry"),
            "domain_validity": persona.get("domain_validity"),
            "safety": persona.get("safety"),
            "provenance": persona.get("provenance"),
            "relationships": persona.get("relationships"),
            "workflows": persona.get("workflows"),
            "embeddings": {
                k: {"model": v.get("model"), "dim": v.get("dim"), "path": v.get("path")}
                for k, v in persona.get("embeddings", {}).items()
            },
        }
        rich_catalog.append(rich_entry)

    graph = {"nodes": graph_nodes, "links": graph_links}
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, ensure_ascii=True, indent=2), encoding="utf-8")
    rich_out.parent.mkdir(parents=True, exist_ok=True)
    rich_out.write_text(json.dumps(rich_catalog, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Generated {len(graph_nodes)} personas with embeddings into {persona_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rich personas (metadata + embeddings) for ToolUniverse tools."
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embeddings.",
    )
    parser.add_argument(
        "--tool",
        action="append",
        help="Specific tool(s) to build personas for (can be repeated). Defaults to all tools.",
    )
    parser.add_argument(
        "--persona-dir",
        default=os.path.join("web", "personas"),
        help="Directory to write persona JSON files.",
    )
    parser.add_argument(
        "--embedding-dir",
        default=os.path.join("web", "embeddings"),
        help="Directory to write embedding tensors.",
    )
    parser.add_argument(
        "--graph-path",
        default=os.path.join("web", "persona_graph.json"),
        help="Path to write aggregated persona graph JSON.",
    )
    parser.add_argument(
        "--tool-file",
        default=os.path.join("web", "v5_all_tools_final.json"),
        help="Path to tool catalog JSON used to build personas.",
    )
    parser.add_argument(
        "--rich-out",
        default=os.path.join("web", "v5_all_tools_rich.json"),
        help="Path to write enriched tool catalog with persona metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_personas(
        model_name=args.model,
        tool_filter=args.tool,
        persona_dir=Path(args.persona_dir),
        embedding_dir=Path(args.embedding_dir),
        graph_path=Path(args.graph_path),
        tool_file=Path(args.tool_file),
        rich_out=Path(args.rich_out),
    )


if __name__ == "__main__":
    main()
