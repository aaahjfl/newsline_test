# NewsLine News Metadata Dataset

This directory contains the open dataset used by NewsLine's event discovery and timeline reconstruction pipeline.

## Contents

- `parser_newsdata.csv`: UTF-8 CSV export of `parser_newsdata`.
- `parser_newsdata.jsonl`: JSON Lines export with the same fields.
- `parser_newsdata.csv.gz`: compressed CSV copy.
- `parser_newsdata.jsonl.gz`: compressed JSONL copy.
- `schema.sql`: MySQL schema for the exported table.
- `metadata.json`: row count, time range, source distribution, and field list.
- `SHA256SUMS`: checksum file for verifying exported artifacts.
- `LICENSE-DATA.md`: data licensing and reuse notes.

## Scope

- Rows: 49130
- Distinct sources: 6
- Parsed time range: 1900-01-01 00:00:00 to 2080-12-31 00:00:00

The dataset contains news metadata only: title, source, URL, raw time expression, normalized timestamps, parser mode, ordering/noise labels, and creation time. It does not include article body text.

## Data Quality Notes

The normalized time fields are parser outputs rather than guaranteed ground truth. Boundary or fallback values such as `1900-01-01` and `2080-12-31` are retained in the export, so downstream experiments should filter or audit them according to their task requirements.

## Fields

| Field | Description |
| --- | --- |
| `id` | Stable row identifier used by the project. |
| `title` | News headline/title. |
| `source` | News source name. |
| `url` | Original article URL. |
| `raw_time` | Raw time expression detected or associated during parsing. |
| `standard_timestamp` | Normalized publication/reference timestamp when available. |
| `event_timestamp` | Single event timestamp when the event can be anchored to one time. |
| `event_time_start` | Event start time for interval-style events. |
| `event_time_end` | Event end time for interval-style events. |
| `time_granularity` | Granularity of the normalized event time. |
| `parse_mode` | Parser route or fallback mode used by the preprocessing layer. |
| `true_order` | Optional manual/order label used in experiments. |
| `is_noise` | Optional noise flag used by downstream filtering. |
| `created_at` | Local database insertion timestamp. |

## Import

Create the table:

```bash
mysql -u <user> -p <database> < schema.sql
```

Import CSV after creating the table:

```sql
LOAD DATA LOCAL INFILE 'parser_newsdata.csv'
INTO TABLE parser_newsdata
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(id,title,source,url,@raw_time,@standard_timestamp,@event_timestamp,@event_time_start,@event_time_end,@time_granularity,@parse_mode,@true_order,@is_noise,@created_at)
SET
  raw_time = NULLIF(@raw_time, ''),
  standard_timestamp = NULLIF(@standard_timestamp, ''),
  event_timestamp = NULLIF(@event_timestamp, ''),
  event_time_start = NULLIF(@event_time_start, ''),
  event_time_end = NULLIF(@event_time_end, ''),
  time_granularity = NULLIF(@time_granularity, ''),
  parse_mode = NULLIF(@parse_mode, ''),
  true_order = NULLIF(@true_order, ''),
  is_noise = NULLIF(@is_noise, ''),
  created_at = NULLIF(@created_at, '');
```

For Python workflows, `parser_newsdata.jsonl` is usually easier to load line by line.

## Source and Use Notes

This dataset is a metadata-level research dataset assembled for NewsLine. Original article content remains with the linked publishers. Please cite the original publisher URLs where appropriate and do not treat this dataset as a redistribution of full news articles.
