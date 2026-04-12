"""Shared pipeline-level defaults for the formal architecture layer."""

PIPELINE_CONFIG = {
    "topic_mode": "single_topic",
    "save_outputs": True,
    "topic_expansion_enabled": True,
    "topic_expansion_langs": ["en", "zh-cn", "es", "ko", "fr", "ru", "uk", "sw"],
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
