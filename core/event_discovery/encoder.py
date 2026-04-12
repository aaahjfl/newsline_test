"""Embedding stage for the formal event discovery pipeline."""

from __future__ import annotations

from functools import lru_cache
import os

import numpy as np

from configs.model_config import MODEL_CONFIG

EMBEDDING_PROMPT = (
    "请为新闻标题生成用于事件发现的聚类向量。"
    "只有当两条标题描述同一现实世界中的单一具体事件时，向量才应高度相似。"
    "请忽略媒体立场、措辞差异和中英文表达差异。"
)


def pick_device() -> str:
    """Pick the best available inference device."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is expected in runtime env.
        return "cpu"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embedding_model_name() -> str:
    """Return the fixed embedding model name, allowing deployment-path override."""
    return os.getenv("NEWSLINE_QWEN_EMBEDDING_MODEL", MODEL_CONFIG["embedding_model"])


@lru_cache(maxsize=1)
def load_embedding_model():
    """Lazy-load the fixed Qwen embedding model."""
    from sentence_transformers import SentenceTransformer

    model_name = get_embedding_model_name()
    device = pick_device()

    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except Exception as exc:
        if device == "cpu":
            raise RuntimeError(f"Failed to load embedding model '{model_name}'.") from exc
        return SentenceTransformer(model_name, device="cpu", trust_remote_code=True)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / safe_norms


def encode_titles(titles: list[str]) -> np.ndarray:
    """Encode titles into L2-normalized embeddings."""
    cleaned_titles = [str(title).strip() for title in titles if str(title).strip()]
    if not cleaned_titles:
        return np.empty((0, 0), dtype=np.float32)

    model = load_embedding_model()
    embeddings = model.encode(
        cleaned_titles,
        prompt=EMBEDDING_PROMPT,
        show_progress_bar=len(cleaned_titles) >= 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return _normalize_embeddings(array)
