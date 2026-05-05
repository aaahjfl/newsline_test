-- Minimal schema placeholders for the formal project structure.
-- Existing production tables remain managed by legacy scripts until migration is complete.

CREATE TABLE IF NOT EXISTS raw_news_data (
    id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    raw_time VARCHAR(64),
    standard_timestamp DATETIME,
    source VARCHAR(255),
    url TEXT
);

CREATE TABLE IF NOT EXISTS parser_newsdata (
    id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    raw_time VARCHAR(64),
    standard_timestamp DATETIME,
    event_timestamp DATETIME,
    event_time_start DATETIME,
    event_time_end DATETIME,
    time_granularity VARCHAR(64),
    source VARCHAR(255),
    url TEXT,
    is_noise BOOLEAN DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS event_discovery_events (
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(191) NOT NULL,
    event_id VARCHAR(191) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    cluster_size INT NOT NULL,
    canonical_title TEXT NULL,
    representative_news_id VARCHAR(128) NULL,
    member_news_ids LONGTEXT NOT NULL,
    event_time_start DATETIME NULL,
    event_time_end DATETIME NULL,
    event_time_anchor DATETIME NULL,
    source_count INT NOT NULL DEFAULT 0,
    confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
    system_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
    noise_reason VARCHAR(64) NULL,
    risk_flags LONGTEXT NULL,
    quality_metrics LONGTEXT NULL,
    generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_discovery_assignments (
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(191) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    event_id VARCHAR(191) NOT NULL,
    news_id VARCHAR(128) NOT NULL,
    title TEXT NULL,
    source VARCHAR(255) NULL,
    url TEXT NULL,
    event_time_anchor DATETIME NULL,
    cluster_size INT NOT NULL DEFAULT 0,
    canonical_title TEXT NULL,
    system_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
    noise_reason VARCHAR(64) NULL,
    generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_discovery_graph (
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(191) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    left_news_id VARCHAR(128) NOT NULL,
    right_news_id VARCHAR(128) NOT NULL,
    left_event_id VARCHAR(191) NULL,
    right_event_id VARCHAR(191) NULL,
    similarity DECIMAL(8,6) NOT NULL,
    time_gap_days DECIMAL(10,3) NULL,
    edge_reason VARCHAR(32) NULL,
    generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
