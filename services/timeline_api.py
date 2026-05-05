"""HTTP API and static frontend for the NewsLine timeline display layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Literal
import uuid

from pydantic import BaseModel, Field

from database.db_utils import get_db_connection
from core.timeline_reasoning.persistence import ARTICLE_TABLE, NODE_TABLE, RUN_TABLE

try:  # FastAPI is an optional runtime dependency until requirements are installed.
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover - keeps imports friendly in minimal envs.
    FastAPI = None
    HTTPException = None
    CORSMiddleware = None
    FileResponse = None
    StaticFiles = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
JOB_RUNNER = PROJECT_ROOT / "code" / "script" / "run_timeline_web_job.py"
JOB_EVENT_PREFIX = "NEWSLINE_JOB_EVENT "
MAX_LOG_LINES = 160
MODE_ESTIMATES_SECONDS = {
    "fast": 240,
    "standard": 420,
    "full": 720,
}
DATASET_START_DATE = "2025-06-01"
DATASET_END_DATE = "2026-04-01"


class CreateTimelineJobRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=255)
    mode: Literal["fast", "standard", "full"] = "fast"
    start_date: date | None = None
    end_date: date | None = None
    force_regenerate: bool = False


@dataclass
class TimelineJob:
    job_id: str
    topic: str
    mode: str
    start_date: str | None = None
    end_date: str | None = None
    status: str = "queued"
    progress: int = 0
    stage: str = "已加入队列"
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    discovery_run_id: str | None = None
    reasoning_run_id: str | None = None
    candidate_count: int | None = None
    filtered_count: int | None = None
    timeline_count: int | None = None
    error: str | None = None
    hint: str | None = None
    cache_hit: bool = False
    returncode: int | None = None
    cancel_requested: bool = False
    process: subprocess.Popen | None = field(default=None, repr=False)
    logs: list[str] = field(default_factory=list, repr=False)

    def update(self, **changes) -> None:
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = time.time()

    def append_log(self, line: str) -> None:
        clean = line.rstrip()
        if not clean:
            return
        self.logs.append(clean)
        if len(self.logs) > MAX_LOG_LINES:
            del self.logs[: len(self.logs) - MAX_LOG_LINES]
        self.updated_at = time.time()

    def public_dict(self, *, include_logs: bool = False) -> dict:
        now = time.time()
        started_at = self.started_at or self.created_at
        finished_at = self.completed_at if self.status in {"completed", "failed", "cancelled"} else now
        elapsed_seconds = max(0, int(finished_at - started_at))
        estimate_total_seconds = MODE_ESTIMATES_SECONDS.get(self.mode, MODE_ESTIMATES_SECONDS["standard"])
        if self.status == "running" and self.progress > 8:
            progress_ratio = max(self.progress / 100, 0.08)
            estimate_total_seconds = max(estimate_total_seconds, int(elapsed_seconds / progress_ratio))
        if self.status == "completed":
            estimate_total_seconds = elapsed_seconds
        remaining_seconds = max(0, estimate_total_seconds - elapsed_seconds)

        payload = {
            "job_id": self.job_id,
            "topic": self.topic,
            "mode": self.mode,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "discovery_run_id": self.discovery_run_id,
            "reasoning_run_id": self.reasoning_run_id,
            "candidate_count": self.candidate_count,
            "filtered_count": self.filtered_count,
            "timeline_count": self.timeline_count,
            "error": self.error,
            "hint": self.hint,
            "cache_hit": self.cache_hit,
            "elapsed_seconds": elapsed_seconds,
            "estimate_total_seconds": estimate_total_seconds,
            "remaining_seconds": remaining_seconds,
            "returncode": self.returncode,
            "cancel_requested": self.cancel_requested,
            "created_at": _format_epoch(self.created_at),
            "updated_at": _format_epoch(self.updated_at),
        }
        if include_logs:
            payload["logs"] = list(self.logs)
        return payload


jobs: dict[str, TimelineJob] = {}
jobs_lock = threading.RLock()


def _format_epoch(value: float) -> str:
    return datetime.fromtimestamp(value).isoformat(sep=" ", timespec="seconds")


def _json_value(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _normalize_row(row: dict) -> dict:
    return {key: _json_value(value) for key, value in row.items()}


def _date_key(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return text[:10] if text else None


def _date_range_label(start_date: str | None, end_date: str | None) -> str:
    if start_date and end_date:
        return f"{start_date} 至 {end_date}"
    if start_date:
        return f"{start_date} 之后"
    if end_date:
        return f"{end_date} 之前"
    return "全时段"


def _config_matches_date_range(config_json: str | None, start_date: str | None, end_date: str | None) -> bool:
    requested_start = start_date
    requested_end = end_date
    requested_is_full_dataset = requested_start == DATASET_START_DATE and requested_end == DATASET_END_DATE

    if not config_json:
        return (requested_start is None and requested_end is None) or requested_is_full_dataset
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        return (requested_start is None and requested_end is None) or requested_is_full_dataset

    config_start = _date_key(config.get("start_date"))
    config_end = _date_key(config.get("end_date"))
    config_is_full_dataset = config_start == DATASET_START_DATE and config_end == DATASET_END_DATE
    if requested_is_full_dataset and config_start is None and config_end is None:
        return True
    if config_is_full_dataset and requested_start is None and requested_end is None:
        return True
    return config_start == requested_start and config_end == requested_end


def _friendly_error_hint(raw_text: str | None) -> str | None:
    if not raw_text:
        return None

    text = raw_text.lower()
    if any(token in text for token in ("mysql", "pymysql", "operationalerror", "can't connect", "access denied", "unknown database")):
        return "MySQL 当前不可用。请确认数据库服务已启动，`configs/db_config.py` 中的连接信息正确，并且 `parser_newsdata` 与时间线相关表可访问。"
    if any(token in text for token in ("ollama", "11434", "connection refused", "max retries exceeded")):
        return "Ollama 当前不可用。请先启动 Ollama，并确认本地模型服务在 `http://localhost:11434` 可访问。"
    if any(token in text for token in ("qwen", "embedding", "model", "safetensors", "cuda", "mps", "out of memory")):
        return "本地模型环境可能未就绪。请确认 embedding 模型和 LLM 模型已下载，运行设备内存充足，必要时先用后端 CLI 单独验证模型加载。"
    if "modulenotfounderror" in text:
        return "后端脚本导入失败。通常是 Python 启动路径或虚拟环境不一致导致的，请确认从项目根目录启动服务。"
    if any(token in text for token in ("no event discovery run found", "no candidate", "0 个候选")):
        return "没有找到可用于生成时间线的候选事件。可以换一个 topic，或检查数据库中是否存在包含该 topic 的新闻标题。"
    return "后端生成过程中出现异常。可以在终端查看 uvicorn 输出，或先用后端 CLI 单独运行同一个 topic 定位问题。"


def _is_missing_cache_table_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in ("doesn't exist", "unknown table", "1146"))


def _count_timeline_nodes(reasoning_run_id: str) -> int:
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*) AS node_count
                FROM {NODE_TABLE}
                WHERE reasoning_run_id = %s
                """,
                (reasoning_run_id,),
            )
            row = cursor.fetchone()
            return int(row.get("node_count") or 0)
    finally:
        connection.close()


def find_cached_timeline_run(
    topic: str,
    mode: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict | None:
    """Return the newest completed run for a fixed-dataset topic/mode pair."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM {RUN_TABLE}
                WHERE topic = %s
                  AND mode = %s
                  AND status = 'completed'
                ORDER BY generated_at DESC, id DESC
                LIMIT 100
                """,
                (topic, mode),
            )
            rows = cursor.fetchall()
            for row in rows:
                if _config_matches_date_range(row.get("config_json"), start_date, end_date):
                    return _normalize_row(row)
            return None
    finally:
        connection.close()


def _build_cached_job(
    topic: str,
    mode: str,
    cached_run: dict,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> TimelineJob:
    reasoning_run_id = str(cached_run["reasoning_run_id"])
    try:
        node_count = _count_timeline_nodes(reasoning_run_id)
    except Exception:
        node_count = None
    return TimelineJob(
        job_id=uuid.uuid4().hex[:12],
        topic=topic,
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        status="completed",
        progress=100,
        stage="已找到历史时间线",
        message=f"数据集固定，已直接复用 MySQL 中同 topic、mode 和日期范围（{_date_range_label(start_date, end_date)}）的最新完成结果。",
        discovery_run_id=cached_run.get("discovery_run_id"),
        reasoning_run_id=reasoning_run_id,
        timeline_count=node_count,
        cache_hit=True,
        started_at=time.time(),
        completed_at=time.time(),
    )


def _set_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.update(**changes)


def _append_job_log(job_id: str, line: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.append_log(line)


def _handle_job_event(job_id: str, payload: dict) -> None:
    event = payload.get("event")
    changes = {
        "progress": int(payload.get("progress") or 0),
        "stage": payload.get("stage") or "",
        "message": payload.get("message") or "",
        "discovery_run_id": payload.get("discovery_run_id"),
        "reasoning_run_id": payload.get("reasoning_run_id"),
        "candidate_count": payload.get("candidate_count"),
        "filtered_count": payload.get("filtered_count"),
        "timeline_count": payload.get("timeline_count"),
    }
    changes = {key: value for key, value in changes.items() if value not in (None, "")}
    if event == "stage":
        changes["status"] = "running"
    elif event == "done":
        changes["status"] = "completed"
        changes["completed_at"] = time.time()
    elif event == "error":
        changes["status"] = "failed"
        changes["completed_at"] = time.time()
        raw_error = payload.get("error") or payload.get("message") or "Unknown job error."
        changes["error"] = raw_error
        changes["hint"] = _friendly_error_hint(raw_error + "\n" + str(payload.get("traceback") or ""))
    _set_job(job_id, **changes)


def _run_job_process(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        topic = job.topic
        mode = job.mode
        start_date = job.start_date
        end_date = job.end_date

    command = [
        sys.executable,
        str(JOB_RUNNER),
        "--topic",
        topic,
        "--mode",
        mode,
    ]
    if start_date:
        command.extend(["--start-date", start_date])
    if end_date:
        command.extend(["--end-date", end_date])

    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            progress=0,
            stage="无法启动生成进程",
            error=str(exc),
            hint=_friendly_error_hint(str(exc)),
        )
        return

    _set_job(job_id, status="running", progress=3, stage="生成进程已启动", process=process, started_at=time.time())

    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip("\n")
        if stripped.startswith(JOB_EVENT_PREFIX):
            try:
                payload = json.loads(stripped[len(JOB_EVENT_PREFIX) :])
            except json.JSONDecodeError:
                _append_job_log(job_id, stripped)
                continue
            _handle_job_event(job_id, payload)
        else:
            _append_job_log(job_id, stripped)

    returncode = process.wait()
    with jobs_lock:
        job = jobs[job_id]
        job.returncode = returncode
        job.process = None
        job.completed_at = time.time()
        job.updated_at = time.time()
        if job.cancel_requested:
            job.status = "cancelled"
            job.progress = min(job.progress, 99)
            job.stage = "生成已停止"
            job.message = "用户已停止当前生成任务。"
        elif returncode != 0 and job.status not in {"failed", "cancelled"}:
            job.status = "failed"
            last_error_line = next(
                (line for line in reversed(job.logs) if "Error" in line or "Exception" in line),
                None,
            )
            job.error = job.error or last_error_line or f"Timeline job exited with code {returncode}."
            job.hint = job.hint or _friendly_error_hint("\n".join(job.logs[-40:]) or job.error)
            job.stage = "生成失败"


def _start_job(topic: str, mode: str, *, start_date: str | None = None, end_date: str | None = None) -> TimelineJob:
    job_id = uuid.uuid4().hex[:12]
    job = TimelineJob(job_id=job_id, topic=topic.strip(), mode=mode, start_date=start_date, end_date=end_date)
    with jobs_lock:
        jobs[job_id] = job

    thread = threading.Thread(target=_run_job_process, args=(job_id,), daemon=True)
    thread.start()
    return job


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            process.terminate()
        else:
            process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return


def load_timeline_result_from_db(reasoning_run_id: str) -> dict:
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM {RUN_TABLE}
                WHERE reasoning_run_id = %s
                LIMIT 1
                """,
                (reasoning_run_id,),
            )
            run = cursor.fetchone()
            if not run:
                raise KeyError(reasoning_run_id)

            cursor.execute(
                f"""
                SELECT *
                FROM {NODE_TABLE}
                WHERE reasoning_run_id = %s
                ORDER BY order_index ASC, id ASC
                """,
                (reasoning_run_id,),
            )
            nodes = [_normalize_row(row) for row in cursor.fetchall()]

            cursor.execute(
                f"""
                SELECT *
                FROM {ARTICLE_TABLE}
                WHERE reasoning_run_id = %s
                ORDER BY event_id ASC, sort_index ASC, id ASC
                """,
                (reasoning_run_id,),
            )
            articles_by_event: dict[str, list[dict]] = {}
            for row in cursor.fetchall():
                article = _normalize_row(row)
                articles_by_event.setdefault(str(article["event_id"]), []).append(article)
    finally:
        connection.close()

    timeline = []
    for node in nodes:
        event_id = str(node["event_id"])
        node["articles"] = articles_by_event.get(event_id, [])
        for json_field in ("member_news_ids", "risk_flags"):
            raw_value = node.get(json_field)
            if isinstance(raw_value, str):
                try:
                    node[json_field] = json.loads(raw_value)
                except json.JSONDecodeError:
                    node[json_field] = []
        timeline.append(node)

    normalized_run = _normalize_row(run)
    config_json = normalized_run.get("config_json")
    config = {}
    if isinstance(config_json, str) and config_json.strip():
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            config = {}
    return {
        "topic": normalized_run.get("topic"),
        "discovery_run_id": normalized_run.get("discovery_run_id"),
        "reasoning_run_id": normalized_run.get("reasoning_run_id"),
        "model_name": normalized_run.get("model_name"),
        "mode": normalized_run.get("mode"),
        "prompt_version": normalized_run.get("prompt_version"),
        "generated_at": normalized_run.get("generated_at"),
        "status": normalized_run.get("status"),
        "start_date": _date_key(config.get("start_date")),
        "end_date": _date_key(config.get("end_date")),
        "summary": {
            "input_event_count": normalized_run.get("input_event_count"),
            "review_event_count": normalized_run.get("review_event_count"),
            "accepted_event_count": normalized_run.get("accepted_event_count"),
            "rejected_event_count": normalized_run.get("rejected_event_count"),
        },
        "timeline": timeline,
    }


def list_recent_timeline_runs(limit: int = 6) -> list[dict]:
    bounded_limit = max(1, min(int(limit or 6), 12))
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    runs.*,
                    COALESCE(node_counts.node_count, 0) AS node_count
                FROM {RUN_TABLE} AS runs
                LEFT JOIN (
                    SELECT reasoning_run_id, COUNT(*) AS node_count
                    FROM {NODE_TABLE}
                    GROUP BY reasoning_run_id
                ) AS node_counts
                  ON node_counts.reasoning_run_id = runs.reasoning_run_id
                WHERE runs.status = 'completed'
                ORDER BY runs.generated_at DESC, runs.id DESC
                LIMIT %s
                """,
                (bounded_limit,),
            )
            rows = [_normalize_row(row) for row in cursor.fetchall()]
    finally:
        connection.close()

    records = []
    for row in rows:
        config = {}
        config_json = row.get("config_json")
        if isinstance(config_json, str) and config_json.strip():
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                config = {}
        records.append(
            {
                "topic": row.get("topic"),
                "mode": row.get("mode"),
                "reasoning_run_id": row.get("reasoning_run_id"),
                "generated_at": row.get("generated_at"),
                "node_count": row.get("node_count"),
                "start_date": _date_key(config.get("start_date")),
                "end_date": _date_key(config.get("end_date")),
            }
        )
    return records


def create_app():
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Run `pip install -r requirements.txt` first.")

    api = FastAPI(title="NewsLine Timeline API")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/api/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok", "service": "newsline-timeline"}

    @api.post("/api/timeline/jobs")
    def create_timeline_job(request: CreateTimelineJobRequest) -> dict:
        topic = request.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="topic must not be empty.")
        start_date = _date_key(request.start_date)
        end_date = _date_key(request.end_date)
        if start_date and end_date and start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be earlier than or equal to end_date.")
        cached_run = None
        try:
            if not request.force_regenerate:
                cached_run = find_cached_timeline_run(
                    topic,
                    request.mode,
                    start_date=start_date,
                    end_date=end_date,
                )
        except Exception as exc:
            if _is_missing_cache_table_error(exc):
                cached_run = None
            else:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "无法连接 MySQL 查询历史时间线。",
                        "hint": _friendly_error_hint(str(exc)),
                    },
                ) from exc
        if cached_run is not None:
            job = _build_cached_job(
                topic,
                request.mode,
                cached_run,
                start_date=start_date,
                end_date=end_date,
            )
            with jobs_lock:
                jobs[job.job_id] = job
            return job.public_dict()
        job = _start_job(topic, request.mode, start_date=start_date, end_date=end_date)
        return job.public_dict()

    @api.get("/api/timeline/jobs/{job_id}/status")
    def get_timeline_job_status(job_id: str, logs: bool = False) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            return job.public_dict(include_logs=logs)

    @api.post("/api/timeline/jobs/{job_id}/cancel")
    def cancel_timeline_job(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            if job.status in {"completed", "failed", "cancelled"}:
                return job.public_dict()
            job.cancel_requested = True
            job.status = "cancelling"
            job.stage = "正在停止生成"
            job.message = "正在终止当前后端任务。"
            job.updated_at = time.time()
            process = job.process

        if process is not None:
            _terminate_process(process)

        with jobs_lock:
            return jobs[job_id].public_dict()

    @api.get("/api/timeline/jobs/{job_id}/result")
    def get_timeline_job_result(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            if job.status != "completed" or not job.reasoning_run_id:
                raise HTTPException(status_code=409, detail="Job is not completed yet.")
            reasoning_run_id = job.reasoning_run_id
            job_status = job.public_dict()

        try:
            result = load_timeline_result_from_db(reasoning_run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Timeline result not found in MySQL.") from None
        result["elapsed_seconds"] = job_status.get("elapsed_seconds")
        result["cache_hit"] = job_status.get("cache_hit")
        return result

    @api.get("/api/timeline/results/{reasoning_run_id}")
    def get_timeline_result_by_run(reasoning_run_id: str) -> dict:
        try:
            return load_timeline_result_from_db(reasoning_run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Timeline result not found in MySQL.") from None

    @api.get("/api/timeline/recent")
    def get_recent_timeline_runs(limit: int = 6) -> dict:
        try:
            return {"items": list_recent_timeline_runs(limit)}
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "无法读取最近生成记录。",
                    "hint": _friendly_error_hint(str(exc)),
                },
            ) from exc

    if STATIC_DIR.exists():
        api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @api.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        @api.get("/{path:path}")
        def frontend_fallback(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found.")
            return FileResponse(STATIC_DIR / "index.html")

    return api


app = create_app() if FastAPI is not None else None
