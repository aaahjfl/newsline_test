const shell = document.querySelector(".app-shell");
const form = document.querySelector("#topicForm");
const topicInput = document.querySelector("#topicInput");
const modeSelect = document.querySelector("#modeSelect");
const startYearSelect = document.querySelector("#startYearSelect");
const startMonthSelect = document.querySelector("#startMonthSelect");
const startDaySelect = document.querySelector("#startDaySelect");
const endYearSelect = document.querySelector("#endYearSelect");
const endMonthSelect = document.querySelector("#endMonthSelect");
const endDaySelect = document.querySelector("#endDaySelect");
const forceRegenerateToggle = document.querySelector("#forceRegenerateToggle");
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
const timelineInsight = document.querySelector("#timelineInsight");
const monthScrubber = document.querySelector("#monthScrubber");
const timelineFrame = document.querySelector("#timelineFrame");
const timelineScroller = document.querySelector("#timelineScroller");
const timelineRail = document.querySelector("#timelineRail");
const articlePopover = document.querySelector("#articlePopover");
const recentList = document.querySelector("#recentList");
const refreshRecentButton = document.querySelector("#refreshRecentButton");
const nodeDrawer = document.querySelector("#nodeDrawer");
const nodeDrawerBody = document.querySelector("#nodeDrawerBody");
const closeNodeDrawer = document.querySelector("#closeNodeDrawer");
const backButton = document.querySelector("#backButton");
const VISIBLE_TIMELINE_NODES = 6;
const DATASET_START = "2025-06-01";
const DATASET_END = "2026-04-01";
const MIN_TIMELINE_NODE_WIDTH = 188;
const MAX_TIMELINE_NODE_WIDTH = 260;
const MIN_TIMELINE_SIDE_PADDING = 40;

let currentJobId = null;
let pollTimer = null;
let scrollVelocity = 0;
let scrollAnimation = null;
let displayedProgress = 0;
let targetProgress = 0;
let softProgressCap = 0;
let progressLoop = null;
let lastProgressTick = 0;
let currentTimelineResult = null;
let activeNodeIndex = null;
let currentMonthGroups = [];
let hoveredNodeIndex = null;
let hoverClearTimer = null;
let popoverNodeIndex = null;
let popoverNode = null;
let popoverHasPointer = false;
let monthScrubberDrag = null;
let monthScrubberIgnoreClickUntil = 0;

function setView(view) {
  shell.dataset.view = view;
}

function parseDateParts(value) {
  const [year, month, day] = value.split("-").map((part) => Number(part));
  return { year, month, day };
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function formatDateParts(year, month, day) {
  return `${year}-${pad2(month)}-${pad2(day)}`;
}

function daysInMonth(year, month) {
  return new Date(year, month, 0).getDate();
}

function monthBoundsForYear(year) {
  const start = parseDateParts(DATASET_START);
  const end = parseDateParts(DATASET_END);
  if (year === start.year && year === end.year) return { min: start.month, max: end.month };
  if (year === start.year) return { min: start.month, max: 12 };
  if (year === end.year) return { min: 1, max: end.month };
  return { min: 1, max: 12 };
}

function dayBoundsForMonth(year, month) {
  const start = parseDateParts(DATASET_START);
  const end = parseDateParts(DATASET_END);
  let min = 1;
  let max = daysInMonth(year, month);
  if (year === start.year && month === start.month) min = start.day;
  if (year === end.year && month === end.month) max = end.day;
  return { min, max };
}

function fillSelect(select, values, formatter = (value) => value, selectedValue = null) {
  const previous = selectedValue ?? select.value;
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = formatter(value);
    select.append(option);
  });
  if (values.map(String).includes(String(previous))) {
    select.value = String(previous);
  }
}

function fillDateControls(prefix, selectedDate) {
  const yearSelect = prefix === "start" ? startYearSelect : endYearSelect;
  const monthSelect = prefix === "start" ? startMonthSelect : endMonthSelect;
  const daySelect = prefix === "start" ? startDaySelect : endDaySelect;
  const selected = parseDateParts(selectedDate);
  const start = parseDateParts(DATASET_START);
  const end = parseDateParts(DATASET_END);
  const years = [];
  for (let year = start.year; year <= end.year; year += 1) years.push(year);

  fillSelect(yearSelect, years, (year) => `${year}年`, selected.year);
  const { min: minMonth, max: maxMonth } = monthBoundsForYear(Number(yearSelect.value));
  const months = [];
  for (let month = minMonth; month <= maxMonth; month += 1) months.push(month);
  fillSelect(monthSelect, months, (month) => `${month}月`, selected.month);

  const { min: minDay, max: maxDay } = dayBoundsForMonth(Number(yearSelect.value), Number(monthSelect.value));
  const days = [];
  for (let day = minDay; day <= maxDay; day += 1) days.push(day);
  fillSelect(daySelect, days, (day) => `${day}日`, selected.day);
}

function getDateValue(prefix) {
  const yearSelect = prefix === "start" ? startYearSelect : endYearSelect;
  const monthSelect = prefix === "start" ? startMonthSelect : endMonthSelect;
  const daySelect = prefix === "start" ? startDaySelect : endDaySelect;
  return formatDateParts(Number(yearSelect.value), Number(monthSelect.value), Number(daySelect.value));
}

function refreshDateControls(prefix) {
  const current = getDateValue(prefix);
  fillDateControls(prefix, current);
}

function initDateControls() {
  fillDateControls("start", DATASET_START);
  fillDateControls("end", DATASET_END);
  [startYearSelect, startMonthSelect].forEach((select) => {
    select.addEventListener("change", () => refreshDateControls("start"));
  });
  [endYearSelect, endMonthSelect].forEach((select) => {
    select.addEventListener("change", () => refreshDateControls("end"));
  });
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

async function createJob(topic, mode, startDate, endDate, forceRegenerate = false) {
  return apiFetch("/api/timeline/jobs", {
    method: "POST",
    body: JSON.stringify({
      topic,
      mode,
      start_date: startDate || null,
      end_date: endDate || null,
      force_regenerate: forceRegenerate,
    }),
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

async function getTimelineResultByRun(reasoningRunId) {
  return apiFetch(`/api/timeline/results/${encodeURIComponent(reasoningRunId)}`);
}

async function getRecentTimelines() {
  return apiFetch("/api/timeline/recent?limit=6");
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
        result.elapsed_seconds = result.elapsed_seconds ?? status.elapsed_seconds;
        renderTimeline(result);
        loadRecentTimelines();
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
  const selectedStartDate = getDateValue("start");
  const selectedEndDate = getDateValue("end");
  const startDate = selectedStartDate;
  const endDate = selectedEndDate;
  const forceRegenerate = forceRegenerateToggle.checked;
  if (!topic) return;
  if (selectedStartDate && selectedEndDate && selectedStartDate > selectedEndDate) {
    setView("running");
    setProgress(0, "日期范围无效", "开始日期不能晚于结束日期。", { status: "failed" });
    showNotice("请调整时间范围", "当前数据集范围为 2025-06-01 至 2026-04-01，可以只填写开始或结束日期。");
    return;
  }

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
    const job = await createJob(topic, mode, startDate, endDate, forceRegenerate);
    currentJobId = job.job_id;
    submitButton.textContent = "停止生成";
    submitButton.disabled = false;
    setProgress(job.progress, job.stage, job.message, job);
    if (job.cache_hit) {
      showNotice("已复用历史结果", "MySQL 中存在同 topic、mode 和日期范围的已完成时间线，本次没有重复运行 SBERT 和 LLM。");
      const result = await getJobResult(job.job_id);
      renderTimeline(result);
      loadRecentTimelines();
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

initDateControls();
loadRecentTimelines();

refreshRecentButton.addEventListener("click", () => {
  loadRecentTimelines();
});

backButton.addEventListener("click", () => {
  hidePopover();
  hideNodeDrawer();
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
  currentTimelineResult = result;
  activeNodeIndex = null;
  hoveredNodeIndex = null;
  hideNodeDrawer();
  const nodes = Array.isArray(result.timeline) ? result.timeline : [];
  resultTitle.textContent = `${result.topic || "Topic"} 时间线`;
  const rangeText = formatDateRange(result.start_date, result.end_date);
  const runtimeText = result.cache_hit
    ? "复用历史结果"
    : result.elapsed_seconds != null
      ? `总用时 ${formatDuration(result.elapsed_seconds)}`
      : null;
  timelineStats.textContent = [
    `${nodes.length} 个时间线节点`,
    `${result.mode || "fast"} 模式`,
    rangeText,
    runtimeText,
    result.reasoning_run_id || "",
  ].filter(Boolean).join(" · ");
  const timelineAnalysis = buildTimelineAnalysis(nodes);
  timelineInsight.textContent = timelineAnalysis.summary;
  renderMonthScrubber(timelineAnalysis.months);
  timelineRail.innerHTML = "";
  timelineRail.style.justifyContent = nodes.length <= VISIBLE_TIMELINE_NODES ? "center" : "flex-start";
  updateTimelineLayout(nodes.length);

  nodes.forEach((node, index) => {
    const item = document.createElement("article");
    item.className = "timeline-node";
    item.dataset.index = String(index);
    item.addEventListener("mouseenter", () => setHoveredNode(index));
    item.addEventListener("mouseleave", () => scheduleClearHoveredNode(index));

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
    title.addEventListener("mouseenter", () => showPopover(title, node, index));
    title.addEventListener("focus", () => showPopover(title, node, index));
    title.addEventListener("mouseleave", scheduleHidePopover);
    title.addEventListener("blur", scheduleHidePopover);
    title.addEventListener("click", () => openNodeDrawer(node, index));

    inner.append(date, dot, title);
    item.append(inner);
    timelineRail.append(item);
  });

  timelineScroller.scrollLeft = 0;
  window.requestAnimationFrame(() => {
    updateTimelineEdgeFades();
    updateCenteredNode();
    updateActiveMonth();
  });
}

function updateTimelineLayout(nodeCount = timelineRail.children.length) {
  const frameWidth = Math.max(320, timelineFrame.clientWidth || window.innerWidth);
  const nodeWidth = Math.max(
    MIN_TIMELINE_NODE_WIDTH,
    Math.min(MAX_TIMELINE_NODE_WIDTH, (frameWidth - MIN_TIMELINE_SIDE_PADDING * 2) / VISIBLE_TIMELINE_NODES),
  );
  const canScroll = nodeCount > VISIBLE_TIMELINE_NODES;
  const sidePadding = canScroll
    ? Math.max(MIN_TIMELINE_SIDE_PADDING, timelineScroller.clientWidth / 2 - nodeWidth / 2)
    : MIN_TIMELINE_SIDE_PADDING;
  timelineRail.style.setProperty("--node-width", `${nodeWidth}px`);
  timelineRail.style.setProperty("--rail-side-padding", `${sidePadding}px`);
}

function formatDateRange(startDate, endDate) {
  if (startDate && endDate) return `${startDate} 至 ${endDate}`;
  if (startDate) return `${startDate} 之后`;
  if (endDate) return `${endDate} 之前`;
  return "全时段";
}

function getNodeDateValue(node) {
  const candidates = [
    node.display_date,
    node.resolved_time_anchor,
    node.resolved_time_start,
    node.event_time_anchor,
    node.event_time_start,
  ];
  for (const candidate of candidates) {
    const match = String(candidate || "").match(/\d{4}-\d{2}-\d{2}/);
    if (match) return match[0];
  }
  return null;
}

function parseIsoDate(value) {
  if (!value) return null;
  const [year, month, day] = value.split("-").map((part) => Number(part));
  if (!year || !month || !day) return null;
  return new Date(year, month - 1, day);
}

function monthKeyFromDate(value) {
  return value ? value.slice(0, 7) : null;
}

function buildTimelineAnalysis(nodes) {
  const groups = [];
  const groupByMonth = new Map();
  const dates = [];

  nodes.forEach((node, index) => {
    const dateValue = getNodeDateValue(node);
    const parsed = parseIsoDate(dateValue);
    if (parsed) dates.push(parsed);
    const monthKey = monthKeyFromDate(dateValue);
    if (!monthKey) return;
    if (!groupByMonth.has(monthKey)) {
      const group = { key: monthKey, firstIndex: index, count: 0 };
      groupByMonth.set(monthKey, group);
      groups.push(group);
    }
    groupByMonth.get(monthKey).count += 1;
  });

  if (dates.length === 0) {
    return {
      summary: nodes.length > 0 ? `共 ${nodes.length} 个节点 · 暂无可解析日期` : "暂无可分析节点",
      months: [],
    };
  }

  const minTime = Math.min(...dates.map((date) => date.getTime()));
  const maxTime = Math.max(...dates.map((date) => date.getTime()));
  const coverageDays = Math.max(1, Math.round((maxTime - minTime) / 86400000) + 1);
  const peakMonth = groups.reduce((best, group) => (!best || group.count > best.count ? group : best), null);
  const averagePerDay = nodes.length / coverageDays;
  return {
    summary: [
      `覆盖 ${coverageDays} 天`,
      `平均每日 ${formatAverage(averagePerDay)} 节点`,
      peakMonth ? `峰值月份 ${peakMonth.key}` : null,
    ].filter(Boolean).join(" · "),
    months: groups,
  };
}

function formatAverage(value) {
  if (value >= 10) return String(Math.round(value));
  return value.toFixed(1);
}

function renderMonthScrubber(months) {
  currentMonthGroups = Array.isArray(months) ? months : [];
  monthScrubber.innerHTML = "";
  if (currentMonthGroups.length === 0) {
    monthScrubber.dataset.empty = "true";
    return;
  }
  monthScrubber.dataset.empty = "false";
  currentMonthGroups.forEach((month) => {
    const button = document.createElement("button");
    button.className = "month-chip";
    button.type = "button";
    button.dataset.month = month.key;
    button.dataset.index = String(month.firstIndex);
    button.innerHTML = `<span>${month.key}</span><small>${month.count}</small>`;
    button.addEventListener("click", (event) => {
      if (performance.now() < monthScrubberIgnoreClickUntil) {
        event.preventDefault();
        return;
      }
      scrollToTimelineNode(month.firstIndex);
    });
    monthScrubber.append(button);
  });
}

function scrollToTimelineNode(index, behavior = "smooth") {
  const item = timelineRail.children[index];
  if (!item) return;
  const targetLeft = item.offsetLeft + item.offsetWidth / 2 - timelineScroller.clientWidth / 2;
  timelineScroller.scrollTo({
    left: Math.max(0, targetLeft),
    behavior,
  });
}

function getCenteredNodeIndex() {
  if (timelineRail.children.length === 0) return null;
  const center = timelineScroller.scrollLeft + timelineScroller.clientWidth / 2;
  let activeIndex = 0;
  let activeDistance = Number.POSITIVE_INFINITY;
  [...timelineRail.children].forEach((item, index) => {
    const itemCenter = item.offsetLeft + item.offsetWidth / 2;
    const distance = Math.abs(itemCenter - center);
    if (distance < activeDistance) {
      activeDistance = distance;
      activeIndex = index;
    }
  });
  return activeIndex;
}

function updateCenteredNode() {
  const highlightedIndex = activeNodeIndex ?? hoveredNodeIndex ?? getCenteredNodeIndex();
  [...timelineRail.children].forEach((item, index) => {
    item.dataset.center = index === highlightedIndex ? "true" : "false";
  });
}

function setHoveredNode(index) {
  window.clearTimeout(hoverClearTimer);
  hoveredNodeIndex = index;
  updateCenteredNode();
}

function clearHoveredNode(expectedIndex = null) {
  window.clearTimeout(hoverClearTimer);
  if (expectedIndex !== null && hoveredNodeIndex !== expectedIndex) return;
  if (expectedIndex !== null && popoverHasPointer && popoverNodeIndex === expectedIndex) return;
  hoveredNodeIndex = null;
  updateCenteredNode();
}

function scheduleClearHoveredNode(expectedIndex = null) {
  window.clearTimeout(hoverClearTimer);
  hoverClearTimer = window.setTimeout(() => clearHoveredNode(expectedIndex), 140);
}

function updateActiveMonth() {
  if (currentMonthGroups.length === 0 || timelineRail.children.length === 0) return;
  const activeIndex = getCenteredNodeIndex();
  const activeNode = currentTimelineResult?.timeline?.[activeIndex];
  const activeMonth = monthKeyFromDate(getNodeDateValue(activeNode || {}));
  [...monthScrubber.children].forEach((button) => {
    const isActive = button.dataset.month === activeMonth;
    button.dataset.active = isActive ? "true" : "false";
    if (isActive) {
      keepMonthChipInView(button);
    }
  });
}

function keepMonthChipInView(button) {
  if (monthScrubberDrag) return;
  const scrubberLeft = monthScrubber.scrollLeft;
  const scrubberRight = scrubberLeft + monthScrubber.clientWidth;
  const buttonLeft = button.offsetLeft - 8;
  const buttonRight = button.offsetLeft + button.offsetWidth + 8;
  if (buttonLeft < scrubberLeft) {
    monthScrubber.scrollTo({ left: Math.max(0, buttonLeft), behavior: "smooth" });
  } else if (buttonRight > scrubberRight) {
    monthScrubber.scrollTo({ left: buttonRight - monthScrubber.clientWidth, behavior: "smooth" });
  }
}

monthScrubber.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || monthScrubber.dataset.empty === "true") return;
  monthScrubberDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startScrollLeft: monthScrubber.scrollLeft,
    moved: false,
  };
});

function moveMonthScrubberDrag(event) {
  if (!monthScrubberDrag || monthScrubberDrag.pointerId !== event.pointerId) return;
  const deltaX = event.clientX - monthScrubberDrag.startX;
  if (Math.abs(deltaX) > 3) {
    monthScrubberDrag.moved = true;
    monthScrubber.dataset.dragging = "true";
  }
  if (monthScrubberDrag.moved) {
    event.preventDefault();
    monthScrubber.scrollLeft = monthScrubberDrag.startScrollLeft - deltaX;
  }
}

function endMonthScrubberDrag(event) {
  if (!monthScrubberDrag || monthScrubberDrag.pointerId !== event.pointerId) return;
  const dragged = monthScrubberDrag.moved;
  monthScrubberDrag = null;
  monthScrubber.dataset.dragging = "false";
  if (dragged) {
    monthScrubberIgnoreClickUntil = performance.now() + 180;
  }
}

document.addEventListener("pointermove", moveMonthScrubberDrag);
document.addEventListener("pointerup", endMonthScrubberDrag);
document.addEventListener("pointercancel", endMonthScrubberDrag);

async function loadRecentTimelines() {
  recentList.innerHTML = '<p class="recent-empty">正在读取最近生成记录。</p>';
  try {
    const payload = await getRecentTimelines();
    const records = Array.isArray(payload.items) ? payload.items : [];
    if (records.length === 0) {
      recentList.innerHTML = '<p class="recent-empty">还没有已完成的时间线。</p>';
      return;
    }
    recentList.innerHTML = "";
    records.forEach((record) => {
      const item = document.createElement("button");
      item.className = "recent-item";
      item.type = "button";
      item.innerHTML = `
        <span class="recent-topic">${escapeHtml(record.topic || "Topic")}</span>
        <span class="recent-meta">${escapeHtml(formatRecentMeta(record))}</span>
      `;
      item.addEventListener("click", () => loadRecentTimeline(record.reasoning_run_id));
      recentList.append(item);
    });
  } catch (error) {
    recentList.innerHTML = '<p class="recent-empty">最近生成记录暂时不可用。</p>';
  }
}

function formatRecentMeta(record) {
  const parts = [
    `${record.node_count || 0} 节点`,
    `${record.mode || "fast"} 模式`,
    formatDateRange(record.start_date, record.end_date),
  ];
  if (record.generated_at) parts.push(formatGeneratedTime(record.generated_at));
  return parts.filter(Boolean).join(" · ");
}

function formatGeneratedTime(value) {
  const text = String(value || "").replace("T", " ");
  const [datePart, timePart = ""] = text.split(" ");
  const [year, month, day] = datePart.split("-");
  const shortTime = timePart.slice(0, 5);
  if (year && month && day && shortTime) return `生成于 ${year}年${month}月${day}日 ${shortTime}`;
  if (year && month && day) return `生成于 ${year}年${month}月${day}日`;
  return text ? `生成于 ${text}` : "";
}

async function loadRecentTimeline(reasoningRunId) {
  if (!reasoningRunId) return;
  hideNotice();
  hidePopover();
  hideNodeDrawer();
  try {
    const result = await getTimelineResultByRun(reasoningRunId);
    result.cache_hit = true;
    renderTimeline(result);
    setView("result");
  } catch (error) {
    showNotice("无法打开历史记录", error.hint || error.message);
  }
}

let hideTimer = null;

function showPopover(anchor, node, index = null) {
  window.clearTimeout(hideTimer);
  window.clearTimeout(hoverClearTimer);
  popoverNodeIndex = index;
  popoverNode = node;
  if (index !== null) setHoveredNode(index);
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
  articlePopover.dataset.nodeIndex = index === null ? "" : String(index);
  articlePopover.setAttribute("aria-hidden", "false");

  const rect = anchor.getBoundingClientRect();
  const popoverRect = articlePopover.getBoundingClientRect();
  const left = Math.min(
    window.innerWidth - popoverRect.width - 16,
    Math.max(16, rect.left + rect.width / 2 - popoverRect.width / 2),
  );
  const top = Math.min(
    window.innerHeight - popoverRect.height - 16,
    Math.max(16, rect.bottom + 26),
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
  delete articlePopover.dataset.nodeIndex;
  articlePopover.setAttribute("aria-hidden", "true");
  popoverNodeIndex = null;
  popoverNode = null;
  popoverHasPointer = false;
}

articlePopover.addEventListener("mouseenter", () => {
  window.clearTimeout(hideTimer);
  window.clearTimeout(hoverClearTimer);
  popoverHasPointer = true;
  if (popoverNodeIndex !== null) setHoveredNode(popoverNodeIndex);
});

articlePopover.addEventListener("mouseleave", () => {
  popoverHasPointer = false;
  scheduleHidePopover();
  scheduleClearHoveredNode(popoverNodeIndex);
});

function openPopoverNodeDrawer(event) {
  const target = event.target instanceof Element ? event.target : null;
  if (target?.closest("#articlePopover a")) return;
  const datasetNodeIndex =
    articlePopover.dataset.nodeIndex === "" || articlePopover.dataset.nodeIndex == null
      ? null
      : Number(articlePopover.dataset.nodeIndex);
  const nodeIndex = popoverNodeIndex ?? datasetNodeIndex;
  const node = popoverNode ?? currentTimelineResult?.timeline?.[nodeIndex];
  if (!node || !Number.isInteger(nodeIndex)) return;
  event.preventDefault();
  event.stopPropagation();
  openNodeDrawer(node, nodeIndex);
}

articlePopover.addEventListener("pointerup", openPopoverNodeDrawer);
articlePopover.addEventListener("click", openPopoverNodeDrawer);

function openNodeDrawer(node, index) {
  activeNodeIndex = index;
  hidePopover();
  [...timelineRail.children].forEach((item, itemIndex) => {
    item.dataset.selected = itemIndex === index ? "true" : "false";
  });
  const articles = Array.isArray(node.articles) ? node.articles : [];
  const articleLinks = articles
    .map((article) => {
      const title = escapeHtml(article.title || "Untitled article");
      const source = escapeHtml(article.source || "Unknown source");
      const url = article.url || "#";
      return `<a href="${escapeAttribute(url)}" target="_blank" rel="noreferrer">${title}<span>${source}</span></a>`;
    })
    .join("");
  const confidence = node.decision_confidence ?? node.confidence;
  const riskFlags = Array.isArray(node.risk_flags) ? node.risk_flags.filter(Boolean) : [];
  const reasons = [
    node.decision_reason,
    node.split_reason ? `拆分提示：${node.split_reason}` : null,
    node.merge_reason ? `合并提示：${node.merge_reason}` : null,
  ].filter(Boolean);

  nodeDrawerBody.innerHTML = `
    <h3>${escapeHtml(getNodeTitle(node))}</h3>
    <p class="node-subtitle">${escapeHtml(node.canonical_title || "未提供 canonical title")}</p>
    <div class="node-detail-grid">
      <span><strong>${escapeHtml(node.display_date || node.resolved_time_anchor || "未知")}</strong>时间锚点</span>
      <span><strong>${articles.length}</strong>相关新闻</span>
      <span><strong>${node.cluster_size ?? "-"}</strong>聚类规模</span>
      <span><strong>${formatConfidence(confidence)}</strong>置信度</span>
    </div>
    ${riskFlags.length > 0 ? `<div class="risk-row">${riskFlags.map((flag) => `<span>${escapeHtml(flag)}</span>`).join("")}</div>` : ""}
    ${reasons.length > 0 ? `<div class="node-reason">${reasons.map((reason) => `<p>${escapeHtml(reason)}</p>`).join("")}</div>` : ""}
    <div class="drawer-section-title">相关新闻</div>
    <div class="drawer-article-list">${articleLinks || "<p>暂无新闻标题。</p>"}</div>
  `;
  nodeDrawer.dataset.open = "true";
  nodeDrawer.setAttribute("aria-hidden", "false");
  updateCenteredNode();
}

function hideNodeDrawer() {
  nodeDrawer.dataset.open = "false";
  nodeDrawer.setAttribute("aria-hidden", "true");
  activeNodeIndex = null;
  [...timelineRail.children].forEach((item) => {
    item.dataset.selected = "false";
  });
  updateCenteredNode();
}

function formatConfidence(value) {
  if (value == null || value === "") return "-";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return `${Math.round(numeric * 100)}%`;
}

closeNodeDrawer.addEventListener("click", hideNodeDrawer);

document.addEventListener("pointerdown", (event) => {
  if (shell.dataset.view !== "result" || nodeDrawer.dataset.open !== "true") return;
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  if (nodeDrawer.contains(target) || target.closest(".timeline-title") || target.closest("#articlePopover")) return;
  hideNodeDrawer();
});

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

function updateTimelineEdgeFades() {
  const maxScroll = Math.max(0, timelineScroller.scrollWidth - timelineScroller.clientWidth);
  const current = timelineScroller.scrollLeft;
  const threshold = 4;
  timelineFrame.dataset.leftFade = current <= threshold ? "off" : "on";
  timelineFrame.dataset.rightFade = current >= maxScroll - threshold ? "off" : "on";
  if (maxScroll <= threshold) {
    timelineFrame.dataset.leftFade = "off";
    timelineFrame.dataset.rightFade = "off";
  }
}

timelineFrame.addEventListener("mousemove", (event) => {
  if (shell.dataset.view !== "result") return;
  const rect = timelineFrame.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const zone = rect.width * 0.09;
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
      updateTimelineEdgeFades();
      scrollAnimation = window.requestAnimationFrame(step);
      return;
    }
    scrollAnimation = null;
  };
  scrollAnimation = window.requestAnimationFrame(step);
}

timelineScroller.addEventListener("scroll", () => {
  updateTimelineEdgeFades();
  updateCenteredNode();
  updateActiveMonth();
});

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
    const activeIndex = getCenteredNodeIndex();
    updateTimelineLayout();
    if (activeIndex !== null) scrollToTimelineNode(activeIndex, "auto");
    updateTimelineEdgeFades();
    updateCenteredNode();
    updateActiveMonth();
  }
});
