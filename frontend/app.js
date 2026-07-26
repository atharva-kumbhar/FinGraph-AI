const qs = (selector) => document.querySelector(selector);
const API_BASE = (window.FINGRAPH_CONFIG && window.FINGRAPH_CONFIG.apiBase) || "";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

const state = {
  latest: null,
  benchmark: null,
  graph: { nodes: [], edges: [], paths: [] },
};

const pipelineOrder = ["llm_only", "basic_rag", "graphrag"];
const colors = {
  llm_only: "#ef4444",
  basic_rag: "#f59e0b",
  graphrag: "#14b8a6",
};

function pendingPipeline(label, accent, warning = "Waiting to run...") {
  return {
    label,
    accent,
    answer: "",
    warning,
    error: null,
    provider: "nvidia",
    model: "",
    evidence: [],
    retrieved_chunks: [],
    metrics: {
      tokens: 0,
      latency_ms: 0,
      cost_usd: 0,
      accuracy: null,
      bertscore_f1: null,
      judge: "NOT_EVALUATED",
      hallucination_risk: "UNKNOWN",
      token_reduction_vs_basic: null,
      cost_reduction_vs_basic: null,
    },
  };
}

function initialRunState(query) {
  return {
    query,
    expected_answer: "",
    expected_answer_source: "Running pipelines separately",
    pipelines: {
      llm_only: pendingPipeline("LLM Only", "red", "Runs last because it sends the dataset context to NVIDIA."),
      basic_rag: pendingPipeline("Basic RAG", "yellow", "Running FAISS RAG first..."),
      graphrag: pendingPipeline("TigerGraph GraphRAG", "cyan", "Running GraphRAG first..."),
    },
    retrieval: { top_chunks: [], entities: [] },
    graph: { nodes: [], edges: [], paths: [], status: "GraphRAG has not completed yet." },
    stats: {},
  };
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value ?? 0);
}

function formatCost(value) {
  return `$${Number(value ?? 0).toFixed(5)}`;
}

function formatLatency(value) {
  return `${formatNumber(value)} ms`;
}

function formatAccuracy(metrics = {}) {
  if (metrics.accuracy === null || metrics.accuracy === undefined) return "Judge unavailable";
  const judge = metrics.judge && !["NOT_EVALUATED", "NEEDS_REVIEW"].includes(metrics.judge)
    ? `${metrics.judge} `
    : "";
  return `${judge}${Number(metrics.accuracy).toFixed(0)}%`;
}

function formatBertScore(metrics = {}) {
  if (metrics.bertscore_f1 === null || metrics.bertscore_f1 === undefined) return "Unavailable";
  return Number(metrics.bertscore_f1).toFixed(4);
}

function pipelineBadge(key, pipeline) {
  if (pipeline.error) return "Needs config";
  const provider = (pipeline.provider || "nvidia").toUpperCase();
  const model = pipeline.model ? ` · ${pipeline.model}` : "";
  if (key === "llm_only") return `${provider}${model}`;
  if (key === "basic_rag") return `FAISS + ${provider}${model}`;
  const reduction = pipeline.metrics?.token_reduction_vs_basic;
  const suffix =
    reduction === null || reduction === undefined ? "" : ` · ${reduction}% tokens`;
  return `TG + ${provider}${model}${suffix}`;
}

function renderPipelineCards(data) {
  const grid = qs("#pipeline-grid");
  grid.innerHTML = pipelineOrder
    .map((key) => {
      const pipeline = data.pipelines[key];
      const title = key === "llm_only" ? "LLM" : key === "basic_rag" ? "RAG" : "GraphRAG";
      const context = contextHtml(key, pipeline);
      const body = pipeline.error
        ? `<p class="error-text">${escapeHtml(pipeline.error)}</p>`
        : `<p class="answer">${escapeHtml(pipeline.answer || pipeline.warning || "Waiting...")}</p>`;

      return `
        <article class="pipeline-card ${pipeline.accent}">
          <div class="pipeline-head">
            <span>${title}</span>
            <b>${escapeHtml(pipelineBadge(key, pipeline))}</b>
          </div>
          <div class="answer-shell">${body}</div>
          <div class="context-block">
            <h4>${context.title}</h4>
            <div class="evidence-list">${context.html}</div>
          </div>
          <div class="mini-metrics">
            ${miniMetric("Tokens", formatNumber(pipeline.metrics.tokens))}
            ${miniMetric("Latency", formatLatency(pipeline.metrics.latency_ms))}
            ${miniMetric("Cost", formatCost(pipeline.metrics.cost_usd))}
            ${miniMetric("Judge", formatAccuracy(pipeline.metrics))}
            ${miniMetric("BERTScore", formatBertScore(pipeline.metrics))}
          </div>
        </article>
      `;
    })
    .join("");
}

function contextHtml(key, pipeline) {
  if (key === "llm_only") {
    const chunks = pipeline.evidence || [];
    return {
      title: `Dataset chunks (${chunks.length})`,
      html:
        chunks
          .slice(0, 6)
          .map(
            (chunk) =>
              `<div class="evidence-item"><b>${escapeHtml(chunk.chunk_id)}</b> ` +
              `${escapeHtml(chunk.preview)}</div>`
          )
          .join("") || '<div class="evidence-item muted">No dataset chunks passed.</div>',
    };
  }
  if (key === "graphrag") {
    const paths = pipeline.graph?.paths || [];
    return {
      title: "Graph path",
      html:
        paths
          .slice(0, 5)
          .map((path) => `<div class="evidence-item">${escapeHtml(path)}</div>`)
          .join("") ||
        `<div class="evidence-item muted">${escapeHtml(
          pipeline.graph?.status || "No TigerGraph path returned."
        )}</div>`,
    };
  }
  const chunks = pipeline.evidence || [];
  return {
    title: `Retrieved chunks (${chunks.length})`,
    html:
      chunks
        .slice(0, 12)
        .map(
          (chunk) =>
            `<div class="evidence-item"><b>${escapeHtml(chunk.chunk_id)}</b> ` +
            `${escapeHtml(chunk.preview)}</div>`
        )
        .join("") || '<div class="evidence-item muted">No chunks returned.</div>',
  };
}

function miniMetric(label, value) {
  return `<div class="mini-metric"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function setText(selector, value) {
  const el = qs(selector);
  if (el) el.textContent = value;
}

function renderExpectedAnswer(data) {
  const answer = data.expected_answer || "";
  setText(
    "#expected-answer",
    answer || data.expected_answer_source || "Expected answer was not generated."
  );
  setText(
    "#expected-source",
    data.expected_answer_source === "gemini_auto_reference"
      ? "Gemini auto-reference"
      : data.expected_answer_source === "retrieved_context_fallback"
        ? "Retrieved context fallback"
      : data.expected_answer_source || "Unavailable"
  );
  qs(".expected-panel")?.classList.toggle("has-error", !answer);
}

function renderTicker(data) {
  const p = data.pipelines;
  setText("#metric-llm-tokens", formatNumber(p.llm_only.metrics.tokens));
  setText("#metric-rag-tokens", formatNumber(p.basic_rag.metrics.tokens));
  setText("#metric-graph-tokens", formatNumber(p.graphrag.metrics.tokens));
  setText("#metric-llm-cost", formatCost(p.llm_only.metrics.cost_usd));
  setText("#metric-rag-cost", formatCost(p.basic_rag.metrics.cost_usd));
  setText("#metric-graph-cost", formatCost(p.graphrag.metrics.cost_usd));
  setText("#metric-llm-latency", formatLatency(p.llm_only.metrics.latency_ms));
  setText("#metric-rag-latency", formatLatency(p.basic_rag.metrics.latency_ms));
  setText("#metric-graph-latency", formatLatency(p.graphrag.metrics.latency_ms));
  setText("#metric-llm-accuracy", formatAccuracy(p.llm_only.metrics));
  setText("#metric-rag-accuracy", formatAccuracy(p.basic_rag.metrics));
  setText("#metric-graph-accuracy", formatAccuracy(p.graphrag.metrics));
  setText("#metric-llm-bertscore", formatBertScore(p.llm_only.metrics));
  setText("#metric-rag-bertscore", formatBertScore(p.basic_rag.metrics));
  setText("#metric-graph-bertscore", formatBertScore(p.graphrag.metrics));
  setText("#metric-llm-risk", p.llm_only.metrics.hallucination_risk || "UNKNOWN");
  setText("#metric-rag-risk", p.basic_rag.metrics.hallucination_risk || "UNKNOWN");
  setText("#metric-graph-risk", p.graphrag.metrics.hallucination_risk || "UNKNOWN");
}

function renderCharts(data) {
  const p = data.pipelines;
  drawBarChart("tokens-chart", "tokens", [
    ["LLM", p.llm_only.metrics.tokens, colors.llm_only],
    ["RAG", p.basic_rag.metrics.tokens, colors.basic_rag],
    ["Graph", p.graphrag.metrics.tokens, colors.graphrag],
  ]);
  drawBarChart("cost-chart", "usd", [
    ["LLM", p.llm_only.metrics.cost_usd, colors.llm_only],
    ["RAG", p.basic_rag.metrics.cost_usd, colors.basic_rag],
    ["Graph", p.graphrag.metrics.cost_usd, colors.graphrag],
  ]);
  drawBarChart("latency-chart", "ms", [
    ["LLM", p.llm_only.metrics.latency_ms, colors.llm_only],
    ["RAG", p.basic_rag.metrics.latency_ms, colors.basic_rag],
    ["Graph", p.graphrag.metrics.latency_ms, colors.graphrag],
  ]);
  drawBarChart("accuracy-chart", "%", [
    ["LLM", p.llm_only.metrics.accuracy, colors.llm_only],
    ["RAG", p.basic_rag.metrics.accuracy, colors.basic_rag],
    ["Graph", p.graphrag.metrics.accuracy, colors.graphrag],
  ]);
  drawBarChart("bertscore-chart", "", [
    ["LLM", p.llm_only.metrics.bertscore_f1, colors.llm_only],
    ["RAG", p.basic_rag.metrics.bertscore_f1, colors.basic_rag],
    ["Graph", p.graphrag.metrics.bertscore_f1, colors.graphrag],
  ]);
}

function drawBarChart(canvasId, unit, rows) {
  const canvas = qs(`#${canvasId}`);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 34;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255, 255, 255, 0.025)";
  ctx.fillRect(0, 0, width, height);

  const values = rows.map((row) => Number(row[1] ?? 0));
  const max = Math.max(...values, 1);
  const barWidth = 70;
  const gap = (width - padding * 2 - barWidth * rows.length) / Math.max(rows.length - 1, 1);

  rows.forEach(([label, value, color], index) => {
    const numericValue = Number(value ?? 0);
    const barHeight = numericValue <= 0 ? 4 : Math.max(8, ((height - 78) * numericValue) / max);
    const x = padding + index * (barWidth + gap);
    const y = height - padding - barHeight;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barWidth, barHeight);
    ctx.fillStyle = "#f6fbf9";
    ctx.font = "13px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x + barWidth / 2, height - 12);
    ctx.fillStyle = "#9caeb1";
    const valueText =
      value === null || value === undefined
        ? "N/A"
        : unit === "usd"
          ? `$${Number(value).toFixed(4)}`
          : unit === ""
            ? Number(value).toFixed(4)
          : `${formatNumber(value)}${unit}`;
    ctx.fillText(valueText, x + barWidth / 2, Math.max(16, y - 10));
  });
}

function renderGraph(graph) {
  state.graph = graph || { nodes: [], edges: [], paths: [] };
  const svg = qs("#graph-svg");
  const nodes = (state.graph.nodes || []).slice(0, 28);
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = (state.graph.edges || [])
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .slice(0, 42);
  svg.innerHTML = "";

  if (!nodes.length) {
    svg.appendChild(svgText(380, 230, state.graph.status || "Run Analysis to render TigerGraph traversal.", "graph-empty"));
    return;
  }

  const positions = layoutNodes(nodes);
  const status =
    state.graph.backend === "tigergraph"
      ? `${nodes.length} nodes / ${edges.length} edges`
      : "TigerGraph not configured";
  svg.appendChild(svgText(380, 28, status, "edge-label"));

  edges.forEach((edge) => {
    const a = positions[edge.source];
    const b = positions[edge.target];
    if (!a || !b) return;
    svg.appendChild(svgLine(a.x, a.y, b.x, b.y, "graph-edge"));
    svg.appendChild(svgText((a.x + b.x) / 2, (a.y + b.y) / 2 - 6, truncate(edge.relation, 18), "edge-label"));
  });

  nodes.forEach((node) => {
    const pos = positions[node.id];
    if (!pos) return;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", pos.x);
    circle.setAttribute("cy", pos.y);
    circle.setAttribute("r", "24");
    circle.setAttribute("class", "graph-node");
    circle.setAttribute("fill", nodeColor(node.type));
    svg.appendChild(circle);
    svg.appendChild(svgText(pos.x, pos.y + 42, truncate(node.label || node.id, 22), "graph-label"));
  });
}

function layoutNodes(nodes) {
  const centerX = 380;
  const centerY = 235;
  const radiusX = nodes.length > 12 ? 300 : 240;
  const radiusY = nodes.length > 12 ? 165 : 130;
  const positions = {};
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(nodes.length, 1) - Math.PI / 2;
    positions[node.id] = {
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  });
  return positions;
}

function nodeColor(type = "") {
  if (type.includes("Company")) return "#3b82f6";
  if (type.includes("Document")) return "#f59e0b";
  if (type.includes("Executive")) return "#8b5cf6";
  if (type.includes("Risk")) return "#ef4444";
  if (type.includes("Event")) return "#06b6d4";
  if (type.includes("FinancialMetric")) return "#10b981";
  return "#64748b";
}

function svgLine(x1, y1, x2, y2, cls) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("class", cls);
  return line;
}

function svgText(x, y, value, cls) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("class", cls);
  text.textContent = value;
  return text;
}

function renderRetrieval(data) {
  const root = qs("#retrieval-list");
  const chunks = data.retrieval?.top_chunks || [];
  const paths = data.graph?.paths || [];
  const extracted = data.graph?.extracted_entities || [];
  const graphStatus = data.graph?.status;
  const chunkHtml = chunks
    .slice(0, 12)
    .map(
      (chunk) => `
        <div class="retrieval-card">
          <h4>${escapeHtml(chunk.chunk_id)} · ${Math.round(chunk.similarity * 100)}%</h4>
          <p>${escapeHtml(chunk.preview)}</p>
          <span>${escapeHtml(chunk.source)}</span>
        </div>
      `
    )
    .join("");
  const extractedHtml = extracted
    .slice(0, 6)
    .map((entity) => '<div class="path-row">' + escapeHtml(entity) + '</div>')
    .join("");
  const pathHtml = paths
    .slice(0, 8)
    .map((path) => `<div class="path-row">${escapeHtml(path)}</div>`)
    .join("");

  root.innerHTML = `
    <h4>FAISS chunks</h4>
    ${chunkHtml || '<p class="empty-state">No retrieved chunks returned.</p>'}
    <h4>Extracted entities</h4>
    ${extractedHtml || '<p class="empty-state">No entities extracted from the query.</p>'}
    <h4>TigerGraph paths</h4>
    ${pathHtml || `<p class="empty-state">${escapeHtml(graphStatus || "No graph paths returned.")}</p>`}
  `;
}

async function runQuery(query) {
  if (!query.trim()) {
    qs("#query-input").focus();
    return;
  }
  setLoading(true);
  const payload = { query };
  state.latest = initialRunState(query);
  renderExpectedAnswer(state.latest);
  renderPipelineCards(state.latest);
  renderTicker(state.latest);
  renderGraph(state.latest.graph);
  renderRetrieval(state.latest);
  try {
    const statsResponse = await fetch(apiUrl("/api/stats"));
    if (statsResponse.ok) {
      state.latest.stats = await statsResponse.json();
      renderStats(state.latest.stats);
    }

    await Promise.all([
      runPipelineStep("basic_rag", "/api/pipeline/basic-rag", payload),
      runPipelineStep("graphrag", "/api/pipeline/graphrag", payload),
    ]);

    await runPipelineStep("llm_only", "/api/pipeline/llm-only", payload);
  } catch (error) {
    qs("#pipeline-grid").innerHTML = `<article class="pipeline-card red"><div class="pipeline-head"><span>Backend Error</span><b>HTTP</b></div><p class="error-text">${escapeHtml(
      error.message
    )}</p></article>`;
  } finally {
    setLoading(false);
  }
}

async function runPipelineStep(key, path, payload) {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`${key} failed with ${response.status}`);
  const pipeline = await response.json();
  state.latest.pipelines[key] = pipeline;

  if (key === "basic_rag") {
    state.latest.retrieval = {
      top_chunks: pipeline.retrieved_chunks || pipeline.evidence || [],
      entities: sortedEntities(pipeline.retrieved_chunks || pipeline.evidence || []),
    };
  }
  if (key === "graphrag") {
    state.latest.graph = pipeline.graph || {
      nodes: [],
      edges: [],
      paths: [],
      status: pipeline.graph_context || pipeline.warning,
    };
  }

  renderPipelineCards(state.latest);
  renderTicker(state.latest);
  renderCharts(state.latest);
  renderGraph(state.latest.graph);
  renderRetrieval(state.latest);
}

function sortedEntities(chunks) {
  return Array.from(
    new Set((chunks || []).flatMap((chunk) => chunk.entities || []))
  ).sort();
}

function batchQuestions() {
  const input = qs("#batch-input");
  return (input?.value || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

async function runBatch() {
  const questions = batchQuestions();
  if (!questions.length) {
    qs("#batch-input").focus();
    setText("#batch-status", "Paste at least 1 question");
    return;
  }

  setBatchLoading(true, `Running 0/${questions.length} questions...`);
  try {
    const response = await fetch(apiUrl("/api/benchmark"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        items: questions.map((question) => ({ question })),
      }),
    });
    if (!response.ok) throw new Error(`Benchmark failed with ${response.status}`);
    const data = await response.json();
    state.benchmark = data;
    renderBenchmark(data);
    setText("#batch-status", `${data.items_evaluated || 0} questions evaluated`);
    const exportButton = qs("#export-csv");
    if (exportButton) exportButton.disabled = !data.rows?.length;
  } catch (error) {
    qs("#batch-table").innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
    setText("#batch-status", "Batch failed");
  } finally {
    setBatchLoading(false);
  }
}

function setBatchLoading(isLoading, label = "") {
  document.body.classList.toggle("is-batching", isLoading);
  const runButton = qs("#batch-form button[type='submit']");
  if (runButton) {
    runButton.disabled = isLoading;
    runButton.textContent = isLoading ? "Running..." : "Run Batch Evaluation";
  }
  const exportButton = qs("#export-csv");
  if (exportButton) exportButton.disabled = isLoading || !state.benchmark?.rows?.length;
  if (label) setText("#batch-status", label);
}

function renderBenchmark(data) {
  const aggregate = data.aggregate || {};
  const improvements = data.improvements || {};
  setText("#improve-token", formatPercent(improvements.graph_token_reduction_vs_rag_percent));
  setText("#improve-cost", formatPercent(improvements.graph_cost_reduction_vs_rag_percent));
  setText("#improve-latency", formatPercent(improvements.graph_latency_improvement_vs_rag_percent));
  setText(
    "#improve-accuracy",
    improvements.graph_accuracy_improvement_vs_rag_percent === null ||
      improvements.graph_accuracy_improvement_vs_rag_percent === undefined
      ? formatPointDelta(improvements.graph_accuracy_point_delta_vs_rag)
      : formatPercent(improvements.graph_accuracy_improvement_vs_rag_percent)
  );
  setText(
    "#improve-bert",
    improvements.graph_bertscore_improvement_vs_rag_percent === null ||
      improvements.graph_bertscore_improvement_vs_rag_percent === undefined
      ? formatPointDelta(improvements.graph_bertscore_delta_vs_rag)
      : formatPercent(improvements.graph_bertscore_improvement_vs_rag_percent)
  );

  const rows = [
    ["LLM", aggregate.llm_only],
    ["RAG", aggregate.basic_rag],
    ["GraphRAG", aggregate.graphrag],
  ];
  qs("#batch-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Pipeline</th>
          <th>Avg Tokens</th>
          <th>Avg Latency</th>
          <th>Avg Cost</th>
          <th>Judge Pass %</th>
          <th>Avg BERTScore</th>
          <th>Coverage</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(([label, bucket]) => renderBenchmarkRow(label, bucket || {}))
          .join("")}
      </tbody>
    </table>
  `;
}

function renderBenchmarkRow(label, bucket) {
  return `
    <tr>
      <td><strong>${escapeHtml(label)}</strong></td>
      <td>${formatNumber(bucket.avg_tokens ?? bucket.tokens)}</td>
      <td>${formatLatency(bucket.avg_latency_ms ?? bucket.latency_ms)}</td>
      <td>${formatCost(bucket.avg_cost_usd ?? bucket.cost_usd)}</td>
      <td>${bucket.judge_pass_percent === null || bucket.judge_pass_percent === undefined ? "Judge unavailable" : `${Number(bucket.judge_pass_percent).toFixed(1)}%`}</td>
      <td>${bucket.avg_bertscore_f1 === null || bucket.avg_bertscore_f1 === undefined ? "Unavailable" : Number(bucket.avg_bertscore_f1).toFixed(4)}</td>
      <td>${formatNumber(bucket.judge_evaluated)} judge / ${formatNumber(bucket.bertscore_evaluated)} BERT</td>
    </tr>
  `;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(1)}%`;
}

function formatPointDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const prefix = Number(value) > 0 ? "+" : "";
  return `${prefix}${Number(value).toFixed(1)} pts`;
}

function exportBenchmarkCsv() {
  if (!state.benchmark?.rows?.length) return;
  const headers = [
    "question",
    "expected_answer",
    "pipeline",
    "answer",
    "judge",
    "accuracy",
    "bertscore_f1",
    "tokens",
    "latency_ms",
    "cost_usd",
    "provider",
    "model",
    "error",
  ];
  const csvRows = [headers];
  state.benchmark.rows.forEach((row) => {
    pipelineOrder.forEach((key) => {
      const pipeline = row.pipelines[key] || {};
      const metrics = pipeline.metrics || {};
      csvRows.push([
        row.query,
        row.expected_answer || "",
        key,
        pipeline.answer || "",
        metrics.judge || "",
        metrics.accuracy ?? "",
        metrics.bertscore_f1 ?? "",
        metrics.tokens ?? "",
        metrics.latency_ms ?? "",
        metrics.cost_usd ?? "",
        pipeline.provider || "",
        pipeline.model || "",
        pipeline.error || "",
      ]);
    });
  });
  const csv = csvRows.map((row) => row.map(csvCell).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `fingraph-benchmark-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function setLoading(isLoading) {
  const button = qs("#query-form button[type='submit']");
  const stopButton = qs("#stop-button");
  document.body.classList.toggle("is-running", isLoading);
  if (button) {
    button.disabled = isLoading;
    button.textContent = isLoading ? "Running..." : "Run Analysis";
  }
  if (stopButton) {
    stopButton.disabled = !isLoading;
    if (isLoading) stopButton.textContent = "Stop";
  }
  if (isLoading) {
    setText("#expected-source", "Gemini auto-reference");
    setText("#expected-answer", "Generating automatically from the query, retrieved context, financial filings, and company data...");
    qs(".expected-panel")?.classList.remove("has-error");
  }
}

async function loadStats() {
  try {
    const response = await fetch(apiUrl("/api/stats"));
    renderStats(await response.json());
  } catch {
    setText("#stat-mode", "offline");
    setText("#stat-graph-source", "start backend");
  }
}

function renderStats(stats) {
  setText("#stat-chunks", formatNumber(stats.loaded_chunks || stats.full_dataset_chunks));
  setText("#stat-entities", formatNumber(stats.entities));
  setText("#stat-relationships", formatNumber(stats.relationships));
  setText("#stat-mode", stats.graph_configured ? "live" : "configured");
  setText("#stat-mode-detail", stats.loaded_from || `${formatNumber(stats.loaded_chunks)} chunks loaded`);
  setText("#stat-graph-source", stats.graph_source || "uploaded counts");
  const pills = qs("#hero-pills");
  if (pills) {
    pills.innerHTML = `
      <span>${formatNumber(stats.loaded_chunks || stats.full_dataset_chunks)} chunks</span>
      <span>${formatNumber(stats.entities)} vertices</span>
      <span>${formatNumber(stats.relationships)} edges</span>
    `;
  }
}

function truncate(value, limit) {
  const text = String(value ?? "");
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}...`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function boot() {
  loadStats();
  renderGraph(state.graph);
  qs("#query-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runQuery(qs("#query-input").value);
  });
  const stopBtn = qs("#stop-button");
  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      stopBtn.textContent = "Stopping...";
      stopBtn.disabled = true;
      try {
        await fetch(apiUrl("/api/stop"), { method: "POST" });
      } catch (err) {
        console.error("Failed to stop:", err);
      }
    });
  }
  qs("#batch-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    runBatch();
  });
  qs("#batch-input")?.addEventListener("input", () => {
    const count = batchQuestions().length;
    setText("#batch-status", `${count} question${count === 1 ? "" : "s"} ready`);
  });
  qs("#export-csv")?.addEventListener("click", exportBenchmarkCsv);
}

boot();
