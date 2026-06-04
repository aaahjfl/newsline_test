CREATE TABLE parser_newsdata (
  id varchar(64) NOT NULL,
  title text NOT NULL,
  raw_time varchar(255) DEFAULT NULL,
  standard_timestamp datetime DEFAULT NULL,
  event_timestamp datetime DEFAULT NULL,
  event_time_start datetime DEFAULT NULL,
  event_time_end datetime DEFAULT NULL,
  time_granularity varchar(32) DEFAULT NULL,
  parse_mode varchar(16) DEFAULT NULL,
  source varchar(100) NOT NULL,
  url varchar(512) NOT NULL,
  true_order int DEFAULT NULL,
  is_noise tinyint(1) DEFAULT NULL,
  created_at timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY url (url)
) DEFAULT CHARSET=utf8mb4;
