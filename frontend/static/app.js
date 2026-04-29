const shell = document.querySelector(".app-shell");
const form = document.querySelector("#topicForm");
const topicInput = document.querySelector("#topicInput");
const modeSelect = document.querySelector("#modeSelect");
const submitButton = document.querySelector("#submitButton");
const progressStage = document.querySelector("#progressStage");
const progressPercent = document.querySelector("#progressPercent");
const progressFill = document.querySelector("#progressFill");
const progressMessage = document.querySelector("#progressMessage");
const elapsedTime = document.querySelector("#elapsedTime");
const estimateTime = document.querySelector("#estimateTime");
const remainingTime = document.querySelector("#remainingTime");
const noticePanel = document.querySelector("#noticePanel");
const noticeTitle = document.querySelector("#noticeTitle");
const noticeBody = document.querySelector("#noticeBody");
const resultTitle = document.querySelector("#resultTitle");
const timelineStats = document.querySelector("#timelineStats");
const timelineFrame = document.querySelector("#timelineFrame");
const timelineScroller = document.querySelector("#timelineScroller");
const timelineRail = document.querySelector("#timelineRail");
const articlePopover = document.querySelector("#articlePopover");
const backButton = document.querySelector("#backButton");
const VISIBLE_TIMELINE_NODES = 6;

let currentJobId = null;
let pollTimer = null;
let scrollVelocity = 0;
let scrollAnimation = null;
let displayedProgress = 0;
let targetProgress = 0;
let softProgressCap = 0;
let progressLoop = null;
let lastProgressTick = 0;

function setView(view) {
  shell.dataset.view = view;
}

function setProgress(progress, stage, message, status = {}) {
  const normalized = Math.max(0, Math.min(100, Number(progress) || 0));
  targetProgress = normalized;
  softProgressCap = getSoftProgressCap(normalized, status.status);
  progressStage.textContent = stage || "正在生成";
  progressMessage.textContent = message || "后端任务正在运行。";
  updateTimeMeta(status);

  if (normalized === 0 || normalized === 100 || ["failed", "cancelled", "completed"].includes(status.status)) {
    displayedProgress = normalized;
    renderDisplayedProgress();
    return;
  }
  ensureProgressLoop();
}

function getSoftProgressCap(progress, status) {
  if (status === "completed") return 100;
  if (status === "failed" || status === "cancelled") return progress;
  if (progress < 8) return 7;
  if (progress < 16) return 15;
  if (progress < 64) return 63;
  if (progress < 72) return 71;
  if (progress < 100) return 96;
  return 100;
}

function ensureProgressLoop() {
  if (progressLoop) return;
  const step = (timestamp) => {
    if (!lastProgressTick) lastProgressTick = timestamp;
    const interval = displayedProgress < targetProgress ? 180 : 1250;
    if (timestamp - lastProgressTick >= interval) {
      lastProgressTick = timestamp;
      const cap = Math.max(targetProgress, softProgressCap);
      if (displayedProgress < cap) {
        displayedProgress += 1;
        renderDisplayedProgress();
      }
    }

    if (displayedProgress < Math.max(targetProgress, softProgressCap)) {
      progressLoop = window.requestAnimationFrame(step);
    } else {
      progressLoop = null;
      lastProgressTick = 0;
    }
  };
  progressLoop = window.requestAnimationFrame(step);
}

function renderDisplayedProgress() {
  const rounded = Math.round(displayedProgress);
  progressPercent.textContent = `${rounded}%`;
  progressFill.style.width = `${rounded}%`;
}

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return `${Math.round(value)} 秒`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  if (minutes < 10 && rest > 0) return `${minutes} 分 ${rest} 秒`;
  return `${Math.round(value / 60)} 分钟`;
}

function updateTimeMeta(status = {}) {
  const elapsed = status.elapsed_seconds;
  const estimate = status.estimate_total_seconds;
  const remaining = status.remaining_seconds;
  elapsedTime.textContent = elapsed == null ? "0 秒" : formatDuration(elapsed);
  estimateTime.textContent = estimate == null ? "计算中" : `约 ${formatDuration(estimate)}`;
  remainingTime.textContent = remaining == null ? "计算中" : formatDuration(remaining);
}

function showNotice(title, body) {
  noticeTitle.textContent = title || "提示";
  noticeBody.textContent = body || "";
  noticePanel.dataset.open = "true";
}

function hideNotice() {
  noticePanel.dataset.open = "false";
  noticeTitle.textContent = "";
  noticeBody.textContent = "";
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "object" && detail !== null
        ? detail.message || detail.hint
        : detail || data.error || `Request failed: ${response.status}`;
    const error = new Error(message);
    if (typeof detail === "object" && detail !== null) {
      error.hint = detail.hint;
    }
    throw error;
  }
  return data;
}

async function createJob(topic, mode) {
  return apiFetch("/api/timeline/jobs", {
    method: "POST",
    body: JSON.stringify({ topic, mode }),
  });
}

async function cancelJob(jobId) {
  return apiFetch(`/api/timeline/jobs/${jobId}/cancel`, { method: "POST" });
}

async function getJobStatus(jobId) {
  return apiFetch(`/api/timeline/jobs/${jobId}/status`);
}

async function getJobResult(jobId) {
  return apiFetch(`/api/timeline/jobs/${jobId}/result`);
}

function startPolling(jobId) {
  stopPolling();
  pollTimer = window.setInterval(async () => {
    try {
      const status = await getJobStatus(jobId);
      setProgress(status.progress, status.stage, status.message, status);

      if (status.status === "completed") {
        stopPolling();
        const result = await getJobResult(jobId);
        renderTimeline(result);
        setView("result");
        submitButton.disabled = false;
      }

      if (["failed", "cancelled"].includes(status.status)) {
        stopPolling();
        setProgress(status.progress, status.stage, status.error || status.message, status);
        if (status.status === "failed") {
          showNotice("环境或后端任务异常", status.hint || status.error || "请查看终端日志获取更多信息。");
        }
        submitButton.disabled = false;
        submitButton.textContent = "生成时间线";
        currentJobId = null;
      }
    } catch (error) {
      stopPolling();
      setProgress(0, "状态查询失败", error.message, { status: "failed" });
      showNotice("状态查询失败", error.hint || error.message);
      submitButton.disabled = false;
    }
  }, 900);
}

function stopPolling() {
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = topicInput.value.trim();
  const mode = modeSelect.value;
  if (!topic) return;

  if (currentJobId) {
    submitButton.disabled = true;
    try {
      await cancelJob(currentJobId);
      stopPolling();
      setProgress(0, "生成已停止", "用户已停止当前生成任务。", { status: "cancelled" });
      hideNotice();
      currentJobId = null;
      submitButton.textContent = "生成时间线";
      setView("idle");
    } catch (error) {
      setProgress(0, "停止失败", error.message);
    } finally {
      submitButton.disabled = false;
    }
    return;
  }

  submitButton.disabled = true;
  hideNotice();
  setView("running");
  setProgress(3, "正在创建任务", "准备启动 SBERT 与 LLM 后端流程。", {
    status: "running",
    elapsed_seconds: 0,
    estimate_total_seconds: getModeEstimateSeconds(mode),
    remaining_seconds: getModeEstimateSeconds(mode),
  });
  try {
    const job = await createJob(topic, mode);
    currentJobId = job.job_id;
    submitButton.textContent = "停止生成";
    submitButton.disabled = false;
    setProgress(job.progress, job.stage, job.message, job);
    if (job.cache_hit) {
      showNotice("已复用历史结果", "MySQL 中存在同 topic 和 mode 的已完成时间线，本次没有重复运行 SBERT 和 LLM。");
      const result = await getJobResult(job.job_id);
      renderTimeline(result);
      window.setTimeout(() => {
        hideNotice();
        setView("result");
      }, 450);
      return;
    }
    startPolling(job.job_id);
  } catch (error) {
    setProgress(0, "任务创建失败", error.message, { status: "failed" });
    showNotice("无法启动生成", error.hint || error.message);
    submitButton.disabled = false;
    submitButton.textContent = "生成时间线";
    currentJobId = null;
  }
});

function getModeEstimateSeconds(mode) {
  if (mode === "full") return 720;
  if (mode === "standard") return 420;
  return 240;
}

backButton.addEventListener("click", () => {
  hidePopover();
  hideNotice();
  currentJobId = null;
  submitButton.textContent = "生成时间线";
  submitButton.disabled = false;
  setView("idle");
});

function getNodeTitle(node) {
  return node.display_title || node.canonical_title || "Untitled event";
}

function renderTimeline(result) {
  const nodes = Array.isArray(result.timeline) ? result.timeline : [];
  resultTitle.textContent = `${result.topic || "Topic"} 时间线`;
  timelineStats.textContent = `${nodes.length} 个时间线节点 · ${result.mode || "fast"} 模式 · ${result.reasoning_run_id || ""}`;
  timelineRail.innerHTML = "";
  timelineRail.style.justifyContent = nodes.length <= VISIBLE_TIMELINE_NODES ? "center" : "flex-start";

  const frameWidth = Math.max(320, timelineFrame.clientWidth || window.innerWidth);
  const nodeWidth = Math.max(188, Math.min(260, (frameWidth - 80) / VISIBLE_TIMELINE_NODES));
  timelineRail.style.setProperty("--node-width", `${nodeWidth}px`);

  nodes.forEach((node, index) => {
    const item = document.createElement("article");
    item.className = "timeline-node";
    item.dataset.index = String(index);

    const inner = document.createElement("div");
    inner.className = "timeline-node-inner";

    const date = document.createElement("div");
    date.className = "timeline-date";
    date.textContent = node.display_date || node.resolved_time_anchor || "No date";

    const dot = document.createElement("div");
    dot.className = "timeline-dot";

    const title = document.createElement("button");
    title.className = "timeline-title";
    title.type = "button";
    title.textContent = getNodeTitle(node);
    title.addEventListener("mouseenter", () => showPopover(title, node));
    title.addEventListener("focus", () => showPopover(title, node));
    title.addEventListener("mouseleave", scheduleHidePopover);
    title.addEventListener("blur", scheduleHidePopover);

    inner.append(date, dot, title);
    item.append(inner);
    timelineRail.append(item);
  });

  timelineScroller.scrollLeft = 0;
}

let hideTimer = null;

function showPopover(anchor, node) {
  window.clearTimeout(hideTimer);
  const articles = Array.isArray(node.articles) ? node.articles : [];
  const links = articles
    .map((article) => {
      const title = escapeHtml(article.title || "Untitled article");
      const source = escapeHtml(article.source || "Unknown source");
      const url = article.url || "#";
      return `<a href="${escapeAttribute(url)}" target="_blank" rel="noreferrer">${title}<span class="article-source">${source}</span></a>`;
    })
    .join("");

  articlePopover.innerHTML = `
    <h3>${escapeHtml(getNodeTitle(node))}</h3>
    <p class="article-count">共 ${articles.length} 条相关新闻</p>
    <div class="article-list">${links || "<p>暂无新闻标题。</p>"}</div>
  `;
  articlePopover.dataset.open = "true";
  articlePopover.setAttribute("aria-hidden", "false");

  const rect = anchor.getBoundingClientRect();
  const popoverRect = articlePopover.getBoundingClientRect();
  const left = Math.min(
    window.innerWidth - popoverRect.width - 16,
    Math.max(16, rect.left + rect.width / 2 - popoverRect.width / 2),
  );
  const top = Math.min(
    window.innerHeight - popoverRect.height - 16,
    Math.max(16, rect.bottom + 14),
  );
  articlePopover.style.left = `${left}px`;
  articlePopover.style.top = `${top}px`;
}

function scheduleHidePopover() {
  window.clearTimeout(hideTimer);
  hideTimer = window.setTimeout(hidePopover, 140);
}

function hidePopover() {
  articlePopover.dataset.open = "false";
  articlePopover.setAttribute("aria-hidden", "true");
}

articlePopover.addEventListener("mouseenter", () => window.clearTimeout(hideTimer));
articlePopover.addEventListener("mouseleave", scheduleHidePopover);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

timelineFrame.addEventListener("mousemove", (event) => {
  if (shell.dataset.view !== "result") return;
  const rect = timelineFrame.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const zone = rect.width * 0.18;
  if (x < zone) {
    scrollVelocity = -Math.ceil(10 * (1 - x / zone));
  } else if (x > rect.width - zone) {
    scrollVelocity = Math.ceil(10 * ((x - (rect.width - zone)) / zone));
  } else {
    scrollVelocity = 0;
  }
  ensureScrollLoop();
});

timelineFrame.addEventListener("mouseleave", () => {
  scrollVelocity = 0;
});

function ensureScrollLoop() {
  if (scrollAnimation) return;
  const step = () => {
    if (scrollVelocity !== 0) {
      timelineScroller.scrollLeft += scrollVelocity;
      scrollAnimation = window.requestAnimationFrame(step);
      return;
    }
    scrollAnimation = null;
  };
  scrollAnimation = window.requestAnimationFrame(step);
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * scale));
  const height = Math.max(1, Math.floor(rect.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, scale };
}

function drawGlobe() {
  const canvas = document.querySelector("#globeCanvas");
  const ctx = canvas.getContext("2d");
  let t = 0;

  function project(lat, lon, radius, cx, cy) {
    const phi = (lat * Math.PI) / 180;
    const theta = ((lon + t) * Math.PI) / 180;
    const x = radius * Math.cos(phi) * Math.sin(theta);
    const y = radius * Math.sin(phi);
    const z = radius * Math.cos(phi) * Math.cos(theta);
    return { x: cx + x, y: cy - y, z };
  }

  function lineForLatitude(lat, radius, cx, cy) {
    ctx.beginPath();
    for (let lon = -180; lon <= 180; lon += 5) {
      const p = project(lat, lon, radius, cx, cy);
      if (lon === -180) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  }

  function lineForLongitude(lon, radius, cx, cy) {
    ctx.beginPath();
    for (let lat = -82; lat <= 82; lat += 4) {
      const p = project(lat, lon, radius, cx, cy);
      if (lat === -82) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  }

  function arc(latA, lonA, latB, lonB, radius, cx, cy, phase) {
    const steps = 42;
    ctx.beginPath();
    for (let i = 0; i <= steps; i += 1) {
      const u = i / steps;
      const lift = Math.sin(u * Math.PI) * radius * 0.22;
      const p = project(
        latA + (latB - latA) * u,
        lonA + (lonB - lonA) * u + Math.sin(phase + u) * 8,
        radius + lift,
        cx,
        cy,
      );
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  }

  function frame() {
    const { width, height } = resizeCanvas(canvas);
    const radius = Math.min(width, height) * 0.34;
    const cx = width * 0.5;
    const cy = height * 0.52;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const gradient = ctx.createRadialGradient(cx - radius * 0.25, cy - radius * 0.3, radius * 0.1, cx, cy, radius);
    gradient.addColorStop(0, "rgba(255,255,255,0.9)");
    gradient.addColorStop(0.55, "rgba(221,225,230,0.34)");
    gradient.addColorStop(1, "rgba(156,164,174,0.1)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "rgba(17, 17, 19, 0.105)";
    ctx.lineWidth = Math.max(1, width / 1200);
    [-60, -35, -15, 15, 35, 60].forEach((lat) => lineForLatitude(lat, radius, cx, cy));
    for (let lon = -150; lon <= 150; lon += 30) lineForLongitude(lon, radius, cx, cy);

    ctx.strokeStyle = "rgba(10, 132, 255, 0.24)";
    ctx.lineWidth = Math.max(1.2, width / 900);
    arc(34, -118, 35, 139, radius, cx, cy, t / 32);
    arc(51, -1, 31, 121, radius, cx, cy, t / 45);
    arc(-23, -46, 1, 104, radius, cx, cy, t / 36);

    ctx.strokeStyle = "rgba(17, 17, 19, 0.18)";
    ctx.lineWidth = Math.max(1, width / 1000);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
    t = (t + 0.16) % 360;
    window.requestAnimationFrame(frame);
  }

  frame();
}

drawGlobe();

window.addEventListener("resize", () => {
  if (shell.dataset.view === "result" && timelineRail.children.length > 0) {
    const frameWidth = Math.max(320, timelineFrame.clientWidth || window.innerWidth);
    const nodeWidth = Math.max(188, Math.min(260, (frameWidth - 80) / VISIBLE_TIMELINE_NODES));
    timelineRail.style.setProperty("--node-width", `${nodeWidth}px`);
  }
});
