#!/usr/bin/env python3
"""
Benchmark two MCP servers using ToolUniverse enriched metadata.

This script measures:
- tools/list tool counts
- find_tools retrieval quality (Hit@k, MRR) using enriched tool descriptions
- find_tools latency stats (p50/p95)
- persona availability (new server only)
- similarity quality from find_similar_tools (new server only)
"""

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_PROTOCOL = "2024-11-05"
DEFAULT_K = 5
DEFAULT_SAMPLE = 50


@dataclass
class BenchmarkResult:
    name: str
    tool_count: int
    persona_tools_present: int
    find_tools_hit_at_k: float
    find_tools_mrr: float
    find_tools_p50_ms: float
    find_tools_p95_ms: float
    find_tools_errors: int
    similarity_overlap: Optional[float]
    similarity_p50_ms: Optional[float]
    similarity_p95_ms: Optional[float]
    persona_graph_nodes: Optional[int]
    persona_graph_links: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark two MCP servers.")
    parser.add_argument("--new", required=True, help="Base URL for new server")
    parser.add_argument("--old", required=True, help="Base URL for old server")
    parser.add_argument("--catalog", default="web/v5_all_tools_rich.json")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="benchmarks/benchmark_results.json")
    parser.add_argument("--csv", default="benchmarks/benchmark_results.csv")
    parser.add_argument(
        "--toolset",
        default=None,
        help="Optional JSON list of tool names to focus the benchmark.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional JSON file with biomedical case studies.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional Markdown report path for case study results.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write per-query traces to JSONL files for auditability.",
    )
    parser.add_argument(
        "--trace-prefix",
        default="benchmarks/traces",
        help="Prefix for trace JSONL files (default: benchmarks/traces).",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_sse_json(text: str) -> Optional[Dict[str, Any]]:
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return None
    return None


class McpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def _ensure_session(self) -> None:
        if self.session_id:
            return
        headers = {"Accept": "application/json, text/event-stream"}
        resp = requests.get(f"{self.base_url}/mcp", headers=headers, timeout=20)
        self.session_id = resp.headers.get("mcp-session-id")
        if not self.session_id:
            raise RuntimeError("Missing mcp-session-id header")
        self._initialize()

    def _initialize(self) -> None:
        headers = {
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self.session_id,
        }
        init_body = {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": DEFAULT_PROTOCOL,
                "clientInfo": {"name": "bench", "version": "0.1"},
                "capabilities": {},
            },
        }
        requests.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json=init_body,
            timeout=30,
        )
        initialized_body = {"jsonrpc": "2.0", "method": "initialized"}
        requests.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json=initialized_body,
            timeout=30,
        )

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        self._ensure_session()
        headers = {
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self.session_id,
        }
        body: Dict[str, Any] = {"jsonrpc": "2.0", "id": "req", "method": method}
        if params is not None:
            body["params"] = params
        resp = requests.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json=body,
            timeout=60,
        )
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return resp.json()
        parsed = _parse_sse_json(resp.text)
        if not parsed:
            raise RuntimeError("Failed to parse MCP response")
        return parsed

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "tools" in data:
            return data["tools"]
        return list(data.values())
    return []


def load_toolset(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(name) for name in data]
    return None


def load_cases(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def build_query(tool: Dict[str, Any]) -> str:
    persona = tool.get("persona", {})
    identity = persona.get("identity", {})
    tags = identity.get("domain_tags", []) or []
    description = identity.get("description") or tool.get("description", "")
    snippet = description.strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:240]
    tag_text = " ".join(tags[:6])
    query = f"{snippet} {tag_text}".strip()
    if not query:
        query = tool.get("name", "")
    return query


def compute_latency_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    values_sorted = sorted(values)
    p50 = statistics.median(values_sorted)
    p95_index = max(0, int(0.95 * (len(values_sorted) - 1)))
    p95 = values_sorted[p95_index]
    return p50, p95


def extract_find_tools(result: Dict[str, Any]) -> List[str]:
    try:
        content = result["result"]["content"][0]["text"]
    except Exception:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        tools = parsed.get("tools", [])
        return [tool.get("name") for tool in tools if isinstance(tool, dict)]
    if isinstance(parsed, list):
        return [tool.get("name") for tool in parsed if isinstance(tool, dict)]
    return []


def extract_similarity(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        content = result["result"]["content"][0]["text"]
    except Exception:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    return parsed.get("neighbors", []) if isinstance(parsed, dict) else []


def evaluate_similarity(
    neighbors: List[Dict[str, Any]],
    persona_map: Dict[str, Dict[str, Any]],
) -> float:
    if not neighbors:
        return 0.0
    scored = 0
    for entry in neighbors:
        name = entry.get("tool")
        if not name:
            continue
        target_tags = set(
            persona_map.get(entry.get("target", ""), {})
            .get("identity", {})
            .get("domain_tags", [])
        )
        neighbor_tags = set(
            persona_map.get(name, {}).get("identity", {}).get("domain_tags", [])
        )
        if target_tags & neighbor_tags:
            scored += 1
    return scored / max(len(neighbors), 1)


def benchmark_endpoint(
    label: str,
    base_url: str,
    tools: List[Dict[str, Any]],
    persona_map: Dict[str, Dict[str, Any]],
    k: int,
    sample: int,
    trace_path: Optional[Path] = None,
) -> BenchmarkResult:
    client = McpClient(base_url)
    tools_list = client.request("tools/list", {})
    tool_entries = tools_list.get("result", {}).get("tools", [])
    tool_count = len(tool_entries)
    tool_names = [t.get("name") for t in tool_entries]
    persona_tools_present = len(
        [
            t
            for t in [
                "get_persona",
                "list_personas",
                "get_persona_graph",
                "find_similar_tools",
            ]
            if t in tool_names
        ]
    )

    query_tools = random.sample(tools, min(sample, len(tools)))
    hit = 0
    mrr_total = 0.0
    latencies = []
    errors = 0

    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if not trace_path.exists():
            trace_path.write_text("", encoding="utf-8")

    for tool in query_tools:
        query = build_query(tool)
        start = time.perf_counter()
        try:
            resp = client.call_tool("find_tools", {"query": query, "limit": k})
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            names = extract_find_tools(resp)
            target = tool.get("name")
            if target in names:
                hit += 1
                rank = names.index(target) + 1
                mrr_total += 1.0 / rank
            if trace_path:
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "endpoint": label,
                                "tool": target,
                                "query": query,
                                "returned": names,
                                "rank": names.index(target) + 1 if target in names else None,
                                "latency_ms": latency_ms,
                            }
                        )
                        + "\n"
                    )
        except Exception:
            errors += 1
            if trace_path:
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "endpoint": label,
                                "tool": tool.get("name"),
                                "query": query,
                                "error": True,
                            }
                        )
                        + "\n"
                    )

    total = max(len(query_tools) - errors, 1)
    hit_at_k = hit / total
    mrr = mrr_total / total
    p50, p95 = compute_latency_stats(latencies)

    similarity_overlap = None
    sim_p50 = None
    sim_p95 = None
    graph_nodes = None
    graph_links = None

    if persona_tools_present:
        sim_latencies = []
        sim_scores = []
        sim_tools = random.sample(query_tools, min(20, len(query_tools)))
        sim_trace_path = None
        if trace_path:
            sim_trace_path = Path(f"{trace_path.as_posix().rsplit('.jsonl', 1)[0]}_similarity.jsonl")
            sim_trace_path.parent.mkdir(parents=True, exist_ok=True)
            if not sim_trace_path.exists():
                sim_trace_path.write_text("", encoding="utf-8")
        for tool in sim_tools:
            start = time.perf_counter()
            try:
                resp = client.call_tool(
                    "find_similar_tools",
                    {"tool_name": tool.get("name"), "top_k": 5, "method": "embedding"},
                )
                sim_latencies.append((time.perf_counter() - start) * 1000)
                neighbors = extract_similarity(resp)
                target_tags = set(
                    persona_map.get(tool.get("name", ""), {})
                    .get("identity", {})
                    .get("domain_tags", [])
                )
                overlap = 0
                for entry in neighbors:
                    name = entry.get("tool")
                    neighbor_tags = set(
                        persona_map.get(name, {})
                        .get("identity", {})
                        .get("domain_tags", [])
                    )
                    if target_tags & neighbor_tags:
                        overlap += 1
                sim_scores.append(overlap / max(len(neighbors), 1))
                if sim_trace_path:
                    sim_trace_path.parent.mkdir(parents=True, exist_ok=True)
                    with sim_trace_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "endpoint": label,
                                    "tool": tool.get("name"),
                                    "neighbors": neighbors,
                                    "overlap": overlap / max(len(neighbors), 1),
                                }
                            )
                            + "\n"
                        )
            except Exception:
                continue
        if sim_scores:
            similarity_overlap = sum(sim_scores) / len(sim_scores)
        sim_p50, sim_p95 = compute_latency_stats(sim_latencies)

        try:
            graph_resp = client.call_tool("get_persona_graph", {})
            content = graph_resp["result"]["content"][0]["text"]
            graph = json.loads(content)
            graph_nodes = len(graph.get("nodes", []))
            graph_links = len(graph.get("links", []))
        except Exception:
            pass

    return BenchmarkResult(
        name=label,
        tool_count=tool_count,
        persona_tools_present=persona_tools_present,
        find_tools_hit_at_k=hit_at_k,
        find_tools_mrr=mrr,
        find_tools_p50_ms=p50,
        find_tools_p95_ms=p95,
        find_tools_errors=errors,
        similarity_overlap=similarity_overlap,
        similarity_p50_ms=sim_p50,
        similarity_p95_ms=sim_p95,
        persona_graph_nodes=graph_nodes,
        persona_graph_links=graph_links,
    )


def benchmark_cases(
    label: str,
    base_url: str,
    cases: List[Dict[str, Any]],
    k: int,
    trace_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not cases:
        return {}
    client = McpClient(base_url)
    tools_list = client.request("tools/list", {})
    tool_entries = tools_list.get("result", {}).get("tools", [])
    tool_names = {t.get("name") for t in tool_entries}
    has_similarity = "find_similar_tools" in tool_names

    step_latencies = []
    step_hits = 0
    step_mrr_total = 0.0
    step_errors = 0
    step_total = 0

    chain_latencies = []
    chain_hits = 0
    chain_total = 0
    chain_errors = 0

    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if not trace_path.exists():
            trace_path.write_text("", encoding="utf-8")

    for case in cases:
        steps = case.get("steps", [])
        for step in steps:
            query = step.get("query", "")
            expected = step.get("expected_tools", [])
            if not query or not expected:
                continue
            step_total += 1
            start = time.perf_counter()
            try:
                resp = client.call_tool("find_tools", {"query": query, "limit": k})
                latency_ms = (time.perf_counter() - start) * 1000
                step_latencies.append(latency_ms)
                names = extract_find_tools(resp)
                hit = False
                best_rank = None
                for tool_name in expected:
                    if tool_name in names:
                        hit = True
                        rank = names.index(tool_name) + 1
                        best_rank = rank if best_rank is None else min(best_rank, rank)
                if hit:
                    step_hits += 1
                    step_mrr_total += 1.0 / best_rank
                if trace_path:
                    with trace_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "endpoint": label,
                                    "case_id": case.get("id"),
                                    "step_query": query,
                                    "expected": expected,
                                    "returned": names,
                                    "rank": best_rank,
                                    "latency_ms": latency_ms,
                                }
                            )
                            + "\n"
                        )
            except Exception:
                step_errors += 1

        seed_tool = case.get("seed_tool")
        chain_expected = case.get("chain_expected", [])
        if seed_tool and chain_expected and has_similarity:
            chain_total += 1
            start = time.perf_counter()
            try:
                resp = client.call_tool(
                    "find_similar_tools",
                    {"tool_name": seed_tool, "top_k": k, "method": "embedding"},
                )
                latency_ms = (time.perf_counter() - start) * 1000
                chain_latencies.append(latency_ms)
                neighbors = extract_similarity(resp)
                neighbor_names = {entry.get("tool") for entry in neighbors}
                if any(tool in neighbor_names for tool in chain_expected):
                    chain_hits += 1
                if trace_path:
                    with trace_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "endpoint": label,
                                    "case_id": case.get("id"),
                                    "seed_tool": seed_tool,
                                    "chain_expected": chain_expected,
                                    "neighbors": neighbors,
                                    "latency_ms": latency_ms,
                                }
                            )
                            + "\n"
                        )
            except Exception:
                chain_errors += 1

    step_p50, step_p95 = compute_latency_stats(step_latencies)
    chain_p50, chain_p95 = compute_latency_stats(chain_latencies)
    step_total_effective = max(step_total - step_errors, 1)
    chain_total_effective = max(chain_total - chain_errors, 1)

    return {
        "step_hit_at_k": step_hits / step_total_effective,
        "step_mrr": step_mrr_total / step_total_effective,
        "step_p50_ms": step_p50,
        "step_p95_ms": step_p95,
        "step_total": step_total,
        "step_errors": step_errors,
        "chain_hit_at_k": None if not has_similarity else chain_hits / chain_total_effective,
        "chain_p50_ms": None if not has_similarity else chain_p50,
        "chain_p95_ms": None if not has_similarity else chain_p95,
        "chain_total": chain_total,
        "chain_errors": chain_errors,
        "similarity_available": has_similarity,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    catalog = load_catalog(Path(args.catalog))
    if not catalog:
        raise SystemExit("Tool catalog is empty")

    toolset = load_toolset(args.toolset)
    if toolset:
        toolset_names = set(toolset)
        catalog = [tool for tool in catalog if tool.get("name") in toolset_names]

    persona_map = {}
    for tool in catalog:
        persona = tool.get("persona")
        name = tool.get("name")
        if name and isinstance(persona, dict):
            persona_map[name] = persona

    new_result = benchmark_endpoint(
        "new",
        args.new,
        catalog,
        persona_map,
        args.k,
        args.sample,
        trace_path=Path(f"{args.trace_prefix}_new.jsonl") if args.trace else None,
    )
    old_result = benchmark_endpoint(
        "old",
        args.old,
        catalog,
        persona_map,
        args.k,
        args.sample,
        trace_path=Path(f"{args.trace_prefix}_old.jsonl") if args.trace else None,
    )

    output = {
        "config": {
            "new": args.new,
            "old": args.old,
            "catalog": args.catalog,
            "sample": args.sample,
            "k": args.k,
            "seed": args.seed,
            "toolset": args.toolset,
            "cases": args.cases,
        },
        "results": [new_result.__dict__, old_result.__dict__],
    }

    cases = load_cases(args.cases)
    if cases:
        output["case_results"] = {
            "new": benchmark_cases(
                "new",
                args.new,
                cases,
                args.k,
                trace_path=Path(f"{args.trace_prefix}_new_cases.jsonl")
                if args.trace
                else None,
            ),
            "old": benchmark_cases(
                "old",
                args.old,
                cases,
                args.k,
                trace_path=Path(f"{args.trace_prefix}_old_cases.jsonl")
                if args.trace
                else None,
            ),
        }

    out_path = Path(args.out)
    _ensure_dir(out_path)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_path = Path(args.csv)
    _ensure_dir(csv_path)
    csv_lines = ["endpoint,tool_count,persona_tools_present,hit_at_k,mrr,p50_ms,p95_ms,errors,sim_overlap,sim_p50_ms,sim_p95_ms,graph_nodes,graph_links"]
    for result in [new_result, old_result]:
        csv_lines.append(
            ",".join(
                [
                    result.name,
                    str(result.tool_count),
                    str(result.persona_tools_present),
                    f"{result.find_tools_hit_at_k:.4f}",
                    f"{result.find_tools_mrr:.4f}",
                    f"{result.find_tools_p50_ms:.2f}",
                    f"{result.find_tools_p95_ms:.2f}",
                    str(result.find_tools_errors),
                    "" if result.similarity_overlap is None else f"{result.similarity_overlap:.4f}",
                    "" if result.similarity_p50_ms is None else f"{result.similarity_p50_ms:.2f}",
                    "" if result.similarity_p95_ms is None else f"{result.similarity_p95_ms:.2f}",
                    "" if result.persona_graph_nodes is None else str(result.persona_graph_nodes),
                    "" if result.persona_graph_links is None else str(result.persona_graph_links),
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    if args.report and cases:
        report_path = Path(args.report)
        _ensure_dir(report_path)
        report_lines = [
            "# Biomedical MCP Benchmark Report",
            "",
            "## Overview",
            f"- New server: {args.new}",
            f"- Old server: {args.old}",
            f"- Sample size: {args.sample}",
            f"- Top-k: {args.k}",
            "",
            "## Case Study Metrics",
        ]
        case_results = output.get("case_results", {})
        for label in ["new", "old"]:
            metrics = case_results.get(label, {})
            report_lines.extend(
                [
                    "",
                    f"### {label.upper()}",
                    f"- step_hit_at_k: {metrics.get('step_hit_at_k')}",
                    f"- step_mrr: {metrics.get('step_mrr')}",
                    f"- step_p50_ms: {metrics.get('step_p50_ms')}",
                    f"- step_p95_ms: {metrics.get('step_p95_ms')}",
                    f"- chain_hit_at_k: {metrics.get('chain_hit_at_k')}",
                    f"- chain_p50_ms: {metrics.get('chain_p50_ms')}",
                    f"- chain_p95_ms: {metrics.get('chain_p95_ms')}",
                ]
            )

        report_lines.extend(["", "## Case Studies"])
        for case in cases:
            report_lines.append(f"- {case.get('id')}: {case.get('title')}")
            context = case.get("context")
            if context:
                report_lines.append(f"  - context: {context}")
            steps = case.get("steps", [])
            for step in steps:
                report_lines.append(f"  - query: {step.get('query')}")
                report_lines.append(f"  - expected_tools: {step.get('expected_tools')}")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
