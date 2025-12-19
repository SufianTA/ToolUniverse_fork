import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PersonaManager:
    def __init__(
        self,
        persona_dir: Path,
        embedding_dir: Path,
        graph_path: Path,
    ):
        self.persona_dir = persona_dir
        self.embedding_dir = embedding_dir
        self.graph_path = graph_path
        self.personas: Dict[str, Dict] = {}
        self.embeddings: Dict[Tuple[str, str], object] = {}
        self.graph: Dict = {"nodes": [], "links": []}

    def load(self) -> None:
        self._load_personas()
        self._load_graph()
        self._load_embeddings()

    def _load_personas(self) -> None:
        if not self.persona_dir.exists():
            return
        for path in self.persona_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                name = data.get("name")
                if name:
                    self.personas[name] = data
            except Exception:
                continue

    def _load_graph(self) -> None:
        if self.graph_path.exists():
            try:
                self.graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
            except Exception:
                self.graph = {"nodes": [], "links": []}

    def _load_embeddings(self) -> None:
        try:
            import torch
        except ImportError:
            return

        for name, persona in self.personas.items():
            emb_info = persona.get("embeddings", {})
            for key, meta in emb_info.items():
                path = Path(meta.get("path", ""))
                if not path.exists():
                    # Also try the shared embedding directory with naming convention
                    fallback = self.embedding_dir / f"{name}_{key}.pt"
                    if fallback.exists():
                        path = fallback
                    else:
                        continue
                try:
                    tensor = torch.load(path, weights_only=False)
                    self.embeddings[(name, key)] = tensor
                except Exception:
                    continue

    def get_persona(self, tool_name: str) -> Optional[Dict]:
        return self.personas.get(tool_name)

    def list_personas(self) -> List[Dict]:
        return list(self.personas.values())

    def get_graph(self) -> Dict:
        return self.graph

    def find_similar(
        self,
        tool_name: str,
        vector_type: str = "purpose",
        top_k: int = 5,
        method: str = "auto",
    ) -> List[Dict]:
        precomputed = []
        if tool_name in self.personas:
            precomputed = (
                self.personas[tool_name]
                .get("relationships", {})
                .get("similar_tools", [])
            )

        method = (method or "auto").lower()
        if method == "precomputed":
            return precomputed[:top_k]

        target = self.embeddings.get((tool_name, vector_type))
        if method == "auto" and target is None and precomputed:
            return precomputed[:top_k]

        if target is None:
            return []

        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return precomputed[:top_k] if method == "auto" else []

        scores = []
        for (name, vtype), tensor in self.embeddings.items():
            if vtype != vector_type or name == tool_name:
                continue
            if tensor.shape != target.shape:
                continue
            sim = F.cosine_similarity(
                target.view(1, -1), tensor.view(1, -1)
            ).item()
            scores.append({"tool": name, "score": sim})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]
