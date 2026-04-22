"""Shared pipeline-level defaults for the formal architecture layer."""

PIPELINE_CONFIG = {
    "topic_mode": "single_topic",
    "save_outputs": True,
    "topic_expansion_enabled": True,
    "topic_expansion_langs": ["en", "zh-cn", "es", "ko", "fr", "ru", "uk", "sw"],
    "topic_expansion_min_language_share": 0.02,
    "topic_alias_backend": "ollama",
    "topic_alias_ollama_url": "http://localhost:11434/api/generate",
    "timeline_reasoning_ollama_url": "http://127.0.0.1:11434/api/generate",
    "topic_alias_request_timeout_seconds": 45,
    "topic_alias_ollama_keep_alive": "0s",
    "topic_alias_ollama_think": False,
    "topic_alias_ollama_num_ctx": 2048,
    "topic_alias_ollama_num_predict": 768,
    "topic_alias_per_language_limit": 4,
    "topic_alias_total_limit": 40,
    "event_discovery_candidate_warning_count": 8000,
    # Paths in the new architecture are resolved from the project root.
    "output_root": "outputs",
    # Legacy directories are preserved for compatibility and comparison only.
    "legacy_code_root": "code",
    "archive_root": "archive_mvp",
    # `newsdata/` is currently retained as a historical/raw data directory.
    "newsdata_root": "newsdata",
    # `archive_mvp/newsdata_for_test/` remains a historical experiment dataset.
    "archive_test_data_root": "archive_mvp/newsdata_for_test",
    # `config.props` is an external legacy config kept only for HeidelTime-era compatibility.
    "legacy_external_config": "config.props",
}
