import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pymysql


def load_eval_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("event_eval_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载评估脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_db_config(args) -> Dict:
    return {
        "host": args.db_host,
        "port": args.db_port,
        "user": args.db_user,
        "password": args.db_password,
        "database": args.db_name,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


def calc_delta(v1: Optional[float], v2: Optional[float]) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    return round(v2 - v1, 4)


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_sampling_parse_mode(row: Dict, strata_on: str) -> str:
    if strata_on == "v1":
        return row["parse_mode_v1"]
    if strata_on == "both":
        return f"{row['parse_mode_v1']}->{row['parse_mode_v2']}"
    return row["parse_mode_v2"]


def main():
    parser = argparse.ArgumentParser(description="一键对比 v1/v2 时间解析效果并导出对比表")
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument("--db-password", default="123456", help="MySQL password")
    parser.add_argument("--db-name", default="news_db", help="MySQL database")
    parser.add_argument("--table-v1", default="parser_newsdata_com", help="v1 结果表")
    parser.add_argument("--table-v2", default="parser_newsdata", help="v2 结果表")
    parser.add_argument("--sample-size", type=int, default=1200, help="对比样本量")
    parser.add_argument("--strata-mode", choices=["language", "language_parse_mode"], default="language_parse_mode", help="分层维度")
    parser.add_argument("--strata-on", choices=["v1", "v2", "both"], default="v2", help="language_parse_mode 时按哪版 parse_mode 分层")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--day-tolerance", type=int, default=1, help="日粒度 relaxed 允许误差天数")
    parser.add_argument("--gold-mode", choices=["strict", "balanced", "loose"], default="balanced", help="伪标注规则强度")
    parser.add_argument("--max-rows", type=int, default=0, help="仅调试用，限制每张表读取条数")
    parser.add_argument("--output-dir", default="code/script/reports", help="输出目录")
    parser.add_argument("--eval-script", default="code/script/eval_event_timestamp_accuracy.py", help="弱监督评估脚本路径")
    args = parser.parse_args()

    eval_script_path = Path(args.eval_script).resolve()
    if not eval_script_path.exists():
        raise FileNotFoundError(f"评估脚本不存在: {eval_script_path}")

    eval_mod = load_eval_module(eval_script_path)
    db_config = build_db_config(args)
    max_rows = args.max_rows if args.max_rows > 0 else None

    print("读取 v1 表数据...")
    rows_v1 = eval_mod.fetch_parser_newsdata(db_config, args.table_v1, max_rows)
    print("读取 v2 表数据...")
    rows_v2 = eval_mod.fetch_parser_newsdata(db_config, args.table_v2, max_rows)

    if not rows_v1:
        raise RuntimeError(f"v1 表无可用数据: {args.table_v1}")
    if not rows_v2:
        raise RuntimeError(f"v2 表无可用数据: {args.table_v2}")

    map_v1 = {r["id"]: r for r in rows_v1}
    map_v2 = {r["id"]: r for r in rows_v2}
    common_ids = sorted(set(map_v1.keys()) & set(map_v2.keys()))

    if not common_ids:
        raise RuntimeError("v1/v2 两张表没有相同 id，无法对比。")

    aligned_rows = []
    standard_mismatch = 0

    for news_id in common_ids:
        r1 = map_v1[news_id]
        r2 = map_v2[news_id]

        std_1 = r1["standard_timestamp"]
        std_2 = r2["standard_timestamp"]
        if std_1 and std_2 and std_1 != std_2:
            standard_mismatch += 1

        standard_timestamp = std_2 if std_2 is not None else std_1
        title = r2["title"] if r2.get("title") else r1.get("title")
        lang = r2.get("lang") if r2.get("lang") not in {None, "", "unknown"} else r1.get("lang", "unknown")

        aligned_rows.append({
            "id": news_id,
            "title": title,
            "standard_timestamp": standard_timestamp,
            "lang": lang,
            "event_timestamp_v1": r1.get("event_timestamp"),
            "event_timestamp_v2": r2.get("event_timestamp"),
            "event_time_start_v1": r1.get("event_time_start"),
            "event_time_end_v1": r1.get("event_time_end"),
            "time_granularity_v1": r1.get("time_granularity"),
            "event_time_start_v2": r2.get("event_time_start"),
            "event_time_end_v2": r2.get("event_time_end"),
            "time_granularity_v2": r2.get("time_granularity"),
            "parse_mode_v1": r1.get("parse_mode", "unknown"),
            "parse_mode_v2": r2.get("parse_mode", "unknown"),
        })

    sample_base_rows = []
    for row in aligned_rows:
        sample_base_rows.append({
            "id": row["id"],
            "title": row["title"],
            "standard_timestamp": row["standard_timestamp"],
            "event_timestamp": row["event_timestamp_v2"],
            "lang": row["lang"],
            "parse_mode": choose_sampling_parse_mode(row, args.strata_on),
        })

    print(f"开始分层抽样，共交集 {len(sample_base_rows)} 条...")
    sampled_base = eval_mod.stratified_sample(sample_base_rows, args.sample_size, args.strata_mode, args.seed)
    sampled_ids = {r["id"] for r in sampled_base}

    sampled_v1 = []
    sampled_v2 = []
    aligned_map = {row["id"]: row for row in aligned_rows}

    for news_id in sampled_ids:
        row = aligned_map[news_id]
        sampled_v1.append({
            "id": news_id,
            "title": row["title"],
            "standard_timestamp": row["standard_timestamp"],
            "event_timestamp": row["event_timestamp_v1"],
            "event_time_start": row["event_time_start_v1"],
            "event_time_end": row["event_time_end_v1"],
            "time_granularity": row["time_granularity_v1"],
            "lang": row["lang"],
            "parse_mode": row["parse_mode_v1"],
        })
        sampled_v2.append({
            "id": news_id,
            "title": row["title"],
            "standard_timestamp": row["standard_timestamp"],
            "event_timestamp": row["event_timestamp_v2"],
            "event_time_start": row["event_time_start_v2"],
            "event_time_end": row["event_time_end_v2"],
            "time_granularity": row["time_granularity_v2"],
            "lang": row["lang"],
            "parse_mode": row["parse_mode_v2"],
        })

    print(f"抽样完成 {len(sampled_ids)} 条，开始分别评估 v1 / v2...")
    summary_v1, detail_v1, lang_v1 = eval_mod.evaluate_sample(sampled_v1, args.day_tolerance, gold_mode=args.gold_mode)
    summary_v2, detail_v2, lang_v2 = eval_mod.evaluate_sample(sampled_v2, args.day_tolerance, gold_mode=args.gold_mode)

    detail_map_v1 = {row["id"]: row for row in detail_v1}
    detail_map_v2 = {row["id"]: row for row in detail_v2}

    comparison_detail_rows = []
    exact_v2_better = 0
    exact_v1_better = 0
    exact_tie = 0

    relaxed_v2_better = 0
    relaxed_v1_better = 0
    relaxed_tie = 0
    range_v2_better = 0
    range_v1_better = 0
    range_tie = 0

    for news_id in sorted(sampled_ids):
        d1 = detail_map_v1.get(news_id)
        d2 = detail_map_v2.get(news_id)
        if not d1 or not d2:
            continue

        exact_cmp = "na"
        relaxed_cmp = "na"
        range_cmp = "na"

        if d1["is_evaluable"] == 1 and d2["is_evaluable"] == 1:
            if d2["is_exact"] > d1["is_exact"]:
                exact_cmp = "v2_better"
                exact_v2_better += 1
            elif d2["is_exact"] < d1["is_exact"]:
                exact_cmp = "v1_better"
                exact_v1_better += 1
            else:
                exact_cmp = "tie"
                exact_tie += 1

            if d2["is_relaxed"] > d1["is_relaxed"]:
                relaxed_cmp = "v2_better"
                relaxed_v2_better += 1
            elif d2["is_relaxed"] < d1["is_relaxed"]:
                relaxed_cmp = "v1_better"
                relaxed_v1_better += 1
            else:
                relaxed_cmp = "tie"
                relaxed_tie += 1

            if d2.get("is_range_hit", 0) > d1.get("is_range_hit", 0):
                range_cmp = "v2_better"
                range_v2_better += 1
            elif d2.get("is_range_hit", 0) < d1.get("is_range_hit", 0):
                range_cmp = "v1_better"
                range_v1_better += 1
            else:
                range_cmp = "tie"
                range_tie += 1

        comparison_detail_rows.append({
            "id": news_id,
            "lang": d2["lang"],
            "title": d2["title"],
            "standard_timestamp": d2["standard_timestamp"],
            "event_timestamp_v1": d1["event_timestamp"],
            "event_timestamp_v2": d2["event_timestamp"],
            "event_time_start_v1": d1.get("event_time_start", ""),
            "event_time_end_v1": d1.get("event_time_end", ""),
            "event_time_start_v2": d2.get("event_time_start", ""),
            "event_time_end_v2": d2.get("event_time_end", ""),
            "time_granularity_v1": d1.get("time_granularity", ""),
            "time_granularity_v2": d2.get("time_granularity", ""),
            "pseudo_time": d2["pseudo_time"],
            "pseudo_range_start": d2.get("pseudo_range_start", ""),
            "pseudo_range_end": d2.get("pseudo_range_end", ""),
            "pseudo_granularity": d2["pseudo_granularity"],
            "pseudo_source_text": d2["pseudo_source_text"],
            "is_evaluable_v1": d1["is_evaluable"],
            "is_evaluable_v2": d2["is_evaluable"],
            "is_exact_v1": d1["is_exact"],
            "is_exact_v2": d2["is_exact"],
            "is_relaxed_v1": d1["is_relaxed"],
            "is_relaxed_v2": d2["is_relaxed"],
            "is_range_hit_v1": d1.get("is_range_hit", 0),
            "is_range_hit_v2": d2.get("is_range_hit", 0),
            "is_anchor_in_gold_range_v1": d1.get("is_anchor_in_gold_range", 0),
            "is_anchor_in_gold_range_v2": d2.get("is_anchor_in_gold_range", 0),
            "is_interval_gold": d2.get("is_interval_gold", 0),
            "exact_compare": exact_cmp,
            "relaxed_compare": relaxed_cmp,
            "range_compare": range_cmp,
        })

    language_rows = []
    all_langs = sorted(set(lang_v1.keys()) | set(lang_v2.keys()))
    for lang in all_langs:
        l1 = lang_v1.get(lang, {})
        l2 = lang_v2.get(lang, {})
        language_rows.append({
            "lang": lang,
            "sample_count": l2.get("sample_count", l1.get("sample_count", 0)),
            "evaluable_count_v1": l1.get("evaluable_count", 0),
            "evaluable_count_v2": l2.get("evaluable_count", 0),
            "exact_accuracy_v1": l1.get("exact_accuracy"),
            "exact_accuracy_v2": l2.get("exact_accuracy"),
            "exact_delta_v2_minus_v1": calc_delta(l1.get("exact_accuracy"), l2.get("exact_accuracy")),
            "relaxed_accuracy_v1": l1.get("relaxed_accuracy"),
            "relaxed_accuracy_v2": l2.get("relaxed_accuracy"),
            "relaxed_delta_v2_minus_v1": calc_delta(l1.get("relaxed_accuracy"), l2.get("relaxed_accuracy")),
            "range_hit_accuracy_v1": l1.get("range_hit_accuracy"),
            "range_hit_accuracy_v2": l2.get("range_hit_accuracy"),
            "range_hit_delta_v2_minus_v1": calc_delta(l1.get("range_hit_accuracy"), l2.get("range_hit_accuracy")),
            "interval_range_hit_accuracy_v1": l1.get("interval_range_hit_accuracy"),
            "interval_range_hit_accuracy_v2": l2.get("interval_range_hit_accuracy"),
            "interval_range_hit_delta_v2_minus_v1": calc_delta(l1.get("interval_range_hit_accuracy"), l2.get("interval_range_hit_accuracy")),
        })

    overall_delta = {
        "exact_accuracy_delta_v2_minus_v1": calc_delta(summary_v1.get("exact_accuracy"), summary_v2.get("exact_accuracy")),
        "relaxed_accuracy_delta_v2_minus_v1": calc_delta(summary_v1.get("relaxed_accuracy"), summary_v2.get("relaxed_accuracy")),
        "range_hit_accuracy_delta_v2_minus_v1": calc_delta(summary_v1.get("range_hit_accuracy"), summary_v2.get("range_hit_accuracy")),
        "interval_range_hit_accuracy_delta_v2_minus_v1": calc_delta(summary_v1.get("interval_range_hit_accuracy"), summary_v2.get("interval_range_hit_accuracy")),
        "evaluable_coverage_delta_v2_minus_v1": calc_delta(summary_v1.get("evaluable_coverage"), summary_v2.get("evaluable_coverage")),
    }

    payload = {
        "config": {
            "table_v1": args.table_v1,
            "table_v2": args.table_v2,
            "sample_size": args.sample_size,
            "strata_mode": args.strata_mode,
            "strata_on": args.strata_on,
            "seed": args.seed,
            "day_tolerance": args.day_tolerance,
            "gold_mode": args.gold_mode,
            "max_rows": args.max_rows,
        },
        "dataset": {
            "v1_count": len(rows_v1),
            "v2_count": len(rows_v2),
            "overlap_count": len(common_ids),
            "standard_timestamp_mismatch_count": standard_mismatch,
            "sampled_count": len(sampled_ids),
        },
        "overall_v1": summary_v1,
        "overall_v2": summary_v2,
        "overall_delta": overall_delta,
        "head_to_head": {
            "exact": {
                "v2_better": exact_v2_better,
                "v1_better": exact_v1_better,
                "tie": exact_tie,
            },
            "relaxed": {
                "v2_better": relaxed_v2_better,
                "v1_better": relaxed_v1_better,
                "tie": relaxed_tie,
            },
            "range_hit": {
                "v2_better": range_v2_better,
                "v1_better": range_v1_better,
                "tie": range_tie,
            },
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = out_dir / f"event_time_compare_summary_{ts}.json"
    language_path = out_dir / f"event_time_compare_by_language_{ts}.csv"
    detail_path = out_dir / f"event_time_compare_detail_{ts}.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    save_csv(language_path, language_rows)
    save_csv(detail_path, comparison_detail_rows)

    print("\n===== v1/v2 对比完成 =====")
    print(json.dumps(payload["overall_v1"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["overall_v2"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["overall_delta"], ensure_ascii=False, indent=2))
    print(f"对比汇总: {summary_path}")
    print(f"分语种对比: {language_path}")
    print(f"样本明细对比: {detail_path}")


if __name__ == "__main__":
    main()
