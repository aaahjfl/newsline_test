"""Model names and related switches for the formal pipeline skeleton."""

MODEL_CONFIG = {
    "embedding_model": "Qwen/Qwen3-Embedding-4B",
    "topic_translation_model": "facebook/nllb-200-distilled-600M",
    "reasoning_model": "qwen3:8b",
    "time_parser_primary": "spaCy",
    "time_parser_fallback": "HeidelTime",
}
