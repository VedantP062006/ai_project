/**
 * Frontend logic for CSP Research Dashboard
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global state
  let currentTab = "tab-overview";
  let presetsData = {};
  let benchmarkResults = [];
  let benchmarkSummary = [];
  let traceEvents = [];
  let currentTraceStep = 0;
  let isPlayingTrace = false;
  let traceTimer = null;
  let activeProblemFamily = "sudoku";
  let currentInitialBoard = [];
  let currentSolution = null;

  // DOM Elements
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");
  const familySelect = document.getElementById("select-family");
  const presetSelect = document.getElementById("select-preset");
  const heuristicSelect = document.getElementById("select-heuristic");
  const propSelect = document.getElementById("select-propagation");
  
  const btnRunSolver = document.getElementById("btn-run-solver");
  const btnTraceSolver = document.getElementById("btn-trace-solver");
  const btnPlayPause = document.getElementById("btn-play-pause");
  const btnStep = document.getElementById("btn-step");
  const btnResetTrace = document.getElementById("btn-reset-trace");
  const speedSlider = document.getElementById("playback-speed");
  const speedLabel = document.getElementById("speed-label");

  const sudokuContainer = document.getElementById("sudoku-container");
  const sudokuBoard = document.getElementById("sudoku-board");
  const graphContainer = document.getElementById("graph-container");
  const graphCanvas = document.getElementById("graph-canvas");
  const cryptoContainer = document.getElementById("crypto-container");
  const cryptoDisplay = document.getElementById("crypto-display");

  // Chart instances
  let backtracksChart = null;
  let par2Chart = null;
  let dolanMoreChart = null;

  // Initialize
  initTabs();
  loadHypotheses();
  loadResults();
  loadPresets();
  initSudokuBoard();

  // Tab switching
  function initTabs() {
    navTabs.forEach(tab => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        navTabs.forEach(t => t.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));

        tab.classList.add("active");
        const targetEl = document.getElementById(target);
        if (targetEl) targetEl.classList.add("active");
        currentTab = target;

        if (target === "tab-analytics") {
          setTimeout(renderAnalyticsCharts, 50);
        } else if (target === "tab-explorer") {
          renderExplorerTable();
        }
      });
    });
  }

  // Load Hypotheses
  async function loadHypotheses() {
    try {
      const res = await fetch("/api/hypotheses");
      const list = await res.json();
      const container = document.getElementById("hypotheses-container");
      if (!container) return;
      container.innerHTML = "";

      list.forEach(h => {
        const card = document.createElement("div");
        card.className = "hyp-card";
        card.innerHTML = `
          <div class="hyp-header">
            <span class="hyp-id">${h.id}</span>
            <span class="hyp-status-pill">${h.status}</span>
          </div>
          <h3 class="hyp-title">${h.title}</h3>
          <p class="hyp-statement">${h.statement}</p>
          <div class="hyp-evidence-box">
            <div class="hyp-evidence-label">Benchmark Evidence</div>
            <div class="hyp-evidence-text">${h.baseline_benchmark}</div>
          </div>
        `;
        container.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load hypotheses:", err);
    }
  }

  // Experiment Selector
  const expSelect = document.getElementById("select-global-exp");
  if (expSelect) {
    expSelect.addEventListener("change", () => {
      loadResults(expSelect.value);
    });
  }

  // Load Results
  async function loadResults(exp = "e5") {
    try {
      const res = await fetch(`/api/results?exp=${exp}`);
      const data = await res.json();
      benchmarkResults = data.raw || [];
      benchmarkSummary = data.summary || [];

      if (benchmarkSummary.length > 0) {
        const bestBt = benchmarkSummary.reduce((min, cur) => (Number(cur.mean_backtracks) || 0) < (Number(min.mean_backtracks) || 0) ? cur : min, benchmarkSummary[0]);
        const bestPar2 = benchmarkSummary.reduce((min, cur) => (Number(cur.par2_sec) || 0) < (Number(min.par2_sec) || 0) ? cur : min, benchmarkSummary[0]);

        const btEl = document.getElementById("card-best-bt");
        if (btEl) btEl.innerText = bestBt.mean_backtracks ?? "-";
        const par2El = document.getElementById("card-best-par2");
        if (par2El) par2El.innerText = bestPar2.par2_sec !== undefined ? `${Number(bestPar2.par2_sec).toFixed(4)} s` : "-";
      }

      updateFilterHeuristicOptions();
      renderExplorerTable();
      renderAnalyticsCharts();
    } catch (err) {
      console.error("Failed to load results:", err);
    }
  }

  // Load Presets
  async function loadPresets() {
    try {
      const res = await fetch("/api/instances/list");
      presetsData = await res.json();
      updatePresetOptions();
    } catch (err) {
      console.error("Failed to load presets:", err);
    }
  }

  function updatePresetOptions() {
    const family = familySelect.value;
    activeProblemFamily = family;
    const presets = presetsData[family] || [];
    presetSelect.innerHTML = "";

    presets.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.innerText = p.name;
      opt.dataset.item = JSON.stringify(p);
      presetSelect.appendChild(opt);
    });

    toggleVisualizerViewport(family);
  }

  familySelect.addEventListener("change", () => {
    updatePresetOptions();
  });

  function toggleVisualizerViewport(family) {
    sudokuContainer.classList.add("hidden");
    graphContainer.classList.add("hidden");
    cryptoContainer.classList.add("hidden");

    if (family === "sudoku") {
      sudokuContainer.classList.remove("hidden");
      document.getElementById("studio-inst-badge").innerText = "SUDOKU";
    } else if (family === "graph_coloring") {
      graphContainer.classList.remove("hidden");
      document.getElementById("studio-inst-badge").innerText = "GRAPH COLORING";
      drawInitialGraph(25);
    } else if (family === "cryptarithmetic") {
      cryptoContainer.classList.remove("hidden");
      document.getElementById("studio-inst-badge").innerText = "CRYPTARITHMETIC";
      cryptoDisplay.innerText = "  SEND\n+ MORE\n------\n MONEY";
    }
  }

  // Sudoku Board Builder
  function initSudokuBoard() {
    sudokuBoard.innerHTML = "";
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const cell = document.createElement("div");
        cell.className = "sudoku-cell";
        cell.id = `cell-${r * 9 + c}`;
        if (c === 2 || c === 5) cell.classList.add("border-right-thick");
        if (r === 2 || r === 5) cell.classList.add("border-bottom-thick");
        cell.innerText = "";
        sudokuBoard.appendChild(cell);
      }
    }
  }

  function renderSudokuBoard(cells) {
    initSudokuBoard();
    cells.forEach(c => {
      const el = document.getElementById(`cell-${c.var}`);
      if (!el) return;
      if (c.assigned_val !== null) {
        el.innerText = c.assigned_val + 1;
        el.classList.add("clue");
      } else {
        el.innerHTML = `<span class="domain-tag">${c.domain_size}</span>`;
      }
    });
  }

  // Draw Graph Canvas
  function drawInitialGraph(numNodes = 25, assignments = {}) {
    const ctx = graphCanvas.getContext("2d");
    ctx.clearRect(0, 0, graphCanvas.width, graphCanvas.height);

    const centerX = graphCanvas.width / 2;
    const centerY = graphCanvas.height / 2;
    const radius = 170;

    const colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#a855f7"];
    const positions = [];

    // Arrange in a circle
    for (let i = 0; i < numNodes; i++) {
      const angle = (i / numNodes) * Math.PI * 2;
      positions.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      });
    }

    // Draw some edges
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i < numNodes; i++) {
      for (let j = i + 1; j < numNodes; j++) {
        if ((i * 3 + j) % 5 === 0) {
          ctx.beginPath();
          ctx.moveTo(positions[i].x, positions[i].y);
          ctx.lineTo(positions[j].x, positions[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw vertices
    positions.forEach((pos, idx) => {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 11, 0, Math.PI * 2);
      const assigned = assignments[idx];
      ctx.fillStyle = assigned !== undefined ? colors[assigned % colors.length] : "#1e293b";
      ctx.fill();
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(idx, pos.x, pos.y);
    });
  }

  // Solve button
  btnRunSolver.addEventListener("click", async () => {
    const payload = getSolverPayload();
    logEvent("RUN", `Solving ${payload.family} using ${payload.heuristic}...`);

    try {
      const res = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      document.getElementById("live-stat-status").innerText = data.stats.status;
      document.getElementById("live-stat-nodes").innerText = data.stats.nodes;
      document.getElementById("live-stat-backtracks").innerText = data.stats.backtracks;
      document.getElementById("live-stat-depth").innerText = data.stats.max_depth;
      document.getElementById("live-stat-time").innerText = `${data.stats.wall_time_sec.toFixed(4)} s`;
      document.getElementById("studio-inst-name").innerText = data.instance_name;

      if (data.family === "sudoku" && data.solution) {
        Object.entries(data.solution).forEach(([varIdx, val]) => {
          const el = document.getElementById(`cell-${varIdx}`);
          if (el) {
            el.innerText = val + 1;
            if (!el.classList.contains("clue")) {
              el.classList.add("assigned");
            }
          }
        });
      } else if (data.family === "graph_coloring" && data.solution) {
        drawInitialGraph(data.num_variables, data.solution);
      }

      logEvent("SOLVED", `Successfully verified solution! Nodes: ${data.stats.nodes}, Backtracks: ${data.stats.backtracks}`);
    } catch (err) {
      logEvent("ERROR", err.message);
    }
  });

  // Step trace button
  btnTraceSolver.addEventListener("click", async () => {
    const payload = getSolverPayload();
    logEvent("TRACE", `Loading trace for ${payload.family}...`);

    try {
      const res = await fetch("/api/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      traceEvents = data.events || [];
      currentTraceStep = 0;
      document.getElementById("trace-step-total").innerText = traceEvents.length;
      document.getElementById("trace-step-curr").innerText = 0;
      document.getElementById("studio-inst-name").innerText = data.instance_name;

      initSudokuBoard();
      logEvent("READY", `Trace loaded with ${traceEvents.length} search steps. Click Play (▶) or Step (⏭).`);
    } catch (err) {
      logEvent("ERROR", err.message);
    }
  });

  function getSolverPayload() {
    const selectedOpt = presetSelect.options[presetSelect.selectedIndex];
    const preset = selectedOpt ? JSON.parse(selectedOpt.dataset.item) : {};

    return {
      family: familySelect.value,
      heuristic: heuristicSelect.value,
      propagation_mode: propSelect.value,
      puzzle_key: preset.puzzle_key,
      difficulty: preset.difficulty || "medium",
      seed: preset.seed || 100,
      num_vertices: preset.num_vertices || 25,
      num_colors: preset.num_colors || 3,
      avg_degree: preset.avg_degree || 4.2,
    };
  }

  // Playback Controls
  btnPlayPause.addEventListener("click", () => {
    if (isPlayingTrace) {
      pauseTrace();
    } else {
      playTrace();
    }
  });

  btnStep.addEventListener("click", () => {
    pauseTrace();
    stepTraceForward();
  });

  btnResetTrace.addEventListener("click", () => {
    pauseTrace();
    currentTraceStep = 0;
    document.getElementById("trace-step-curr").innerText = 0;
    initSudokuBoard();
    logEvent("RESET", "Trace rewinded to start.");
  });

  speedSlider.addEventListener("input", (e) => {
    speedLabel.innerText = `${e.target.value} ms`;
    if (isPlayingTrace) {
      pauseTrace();
      playTrace();
    }
  });

  function playTrace() {
    isPlayingTrace = true;
    btnPlayPause.innerText = "⏸";
    const speed = parseInt(speedSlider.value);

    traceTimer = setInterval(() => {
      if (currentTraceStep >= traceEvents.length) {
        pauseTrace();
        return;
      }
      stepTraceForward();
    }, speed);
  }

  function pauseTrace() {
    isPlayingTrace = false;
    btnPlayPause.innerText = "▶";
    if (traceTimer) clearInterval(traceTimer);
  }

  function stepTraceForward() {
    if (currentTraceStep >= traceEvents.length) return;
    const evt = traceEvents[currentTraceStep];
    currentTraceStep++;
    document.getElementById("trace-step-curr").innerText = currentTraceStep;

    applyTraceEvent(evt);
  }

  function applyTraceEvent(evt) {
    if (evt.nodes !== undefined) document.getElementById("live-stat-nodes").innerText = evt.nodes;
    if (evt.backtracks !== undefined) document.getElementById("live-stat-backtracks").innerText = evt.backtracks;
    if (evt.depth !== undefined) document.getElementById("live-stat-depth").innerText = evt.depth;

    if (evt.type === "SELECT_VAR") {
      logEvent("SELECT", `Selected ${evt.var_name} (Domain Size: ${evt.domain_size})`);
      highlightSudokuCell(evt.var, "active-choice");
    } else if (evt.type === "ASSIGN") {
      logEvent("ASSIGN", `Assigning ${evt.var_name} ← ${evt.val + 1}`);
      setSudokuCellVal(evt.var, evt.val + 1, "assigned");
    } else if (evt.type === "BACKTRACK") {
      logEvent("BACKTRACK", `Conflict! Backtracking on ${evt.var}`);
      highlightSudokuCell(evt.var, "backtracked");
      setTimeout(() => clearSudokuCellVal(evt.var), 150);
    } else if (evt.type === "SOLVED") {
      logEvent("SOLVED", "Complete consistent solution verified!");
      document.getElementById("live-stat-status").innerText = "SOLVED";
    }
  }

  function highlightSudokuCell(varIdx, className) {
    const el = document.getElementById(`cell-${varIdx}`);
    if (el) {
      el.classList.add(className);
      setTimeout(() => el.classList.remove(className), 300);
    }
  }

  function setSudokuCellVal(varIdx, val, className) {
    const el = document.getElementById(`cell-${varIdx}`);
    if (el) {
      el.innerText = val;
      el.className = `sudoku-cell ${className}`;
    }
  }

  function clearSudokuCellVal(varIdx) {
    const el = document.getElementById(`cell-${varIdx}`);
    if (el && !el.classList.contains("clue")) {
      el.innerText = "";
      el.className = "sudoku-cell";
    }
  }

  function logEvent(type, msg) {
    const stream = document.getElementById("log-stream");
    const item = document.createElement("div");
    item.className = `log-item ${type.toLowerCase()}`;
    item.innerText = `[${type}] ${msg}`;
    stream.prepend(item);
    document.getElementById("log-step-desc").innerText = msg;
  }

  // Render Charts
  function renderAnalyticsCharts() {
    if (typeof Chart === "undefined") {
      console.warn("Chart.js is not loaded yet.");
      return;
    }

    if (!benchmarkSummary || benchmarkSummary.length === 0) return;

    // Aggregate stats from benchmarkResults for any missing summary metrics
    const nodesByHeur = {};
    const btByHeur = {};
    (benchmarkResults || []).forEach(r => {
      if (!r || !r.heuristic) return;
      if (!nodesByHeur[r.heuristic]) nodesByHeur[r.heuristic] = [];
      if (!btByHeur[r.heuristic]) btByHeur[r.heuristic] = [];
      nodesByHeur[r.heuristic].push(Number(r.nodes) || 0);
      btByHeur[r.heuristic].push(Number(r.backtracks) || 0);
    });

    const labels = benchmarkSummary.map(s => s.heuristic);
    const meanBacktracks = benchmarkSummary.map(s => {
      if (s.mean_backtracks !== undefined) return Number(s.mean_backtracks);
      const arr = btByHeur[s.heuristic] || [];
      return arr.length > 0 ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : 0;
    });

    const medianBacktracks = benchmarkSummary.map(s => {
      if (s.median_backtracks !== undefined) return Number(s.median_backtracks);
      const arr = (btByHeur[s.heuristic] || []).slice().sort((a, b) => a - b);
      if (arr.length === 0) return 0;
      const mid = Math.floor(arr.length / 2);
      return arr.length % 2 !== 0 ? arr[mid] : +((arr[mid - 1] + arr[mid]) / 2).toFixed(1);
    });

    const par2Runtimes = benchmarkSummary.map(s => Number(s.par2_sec) || 0);

    const meanNodes = benchmarkSummary.map(s => {
      if (s.mean_nodes !== undefined) return Number(s.mean_nodes);
      const arr = nodesByHeur[s.heuristic] || [];
      return arr.length > 0 ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : 0;
    });

    // 1. Backtracks Chart
    const ctxBt = document.getElementById("backtracks-chart");
    if (ctxBt) {
      if (backtracksChart) {
        try { backtracksChart.destroy(); } catch (e) {}
      }
      backtracksChart = new Chart(ctxBt, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Mean Backtracks",
              data: meanBacktracks,
              backgroundColor: "rgba(0, 242, 254, 0.75)",
              borderColor: "#00f2fe",
              borderWidth: 1,
            },
            {
              label: "Median Backtracks",
              data: medianBacktracks,
              backgroundColor: "rgba(168, 85, 247, 0.75)",
              borderColor: "#a855f7",
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              title: { display: true, text: "Mean / Median Backtracks", color: "#94a3b8" },
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: { color: "#94a3b8" },
            },
            x: {
              grid: { display: false },
              ticks: { color: "#cbd5e1" },
            },
          },
          plugins: {
            legend: { labels: { color: "#f8fafc" } },
          },
        },
      });
    }

    // 2. PAR2 Runtime Chart
    const ctxPar2 = document.getElementById("par2-chart");
    if (ctxPar2) {
      if (par2Chart) {
        try { par2Chart.destroy(); } catch (e) {}
      }
      par2Chart = new Chart(ctxPar2, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "PAR2 Runtime (s)",
              data: par2Runtimes,
              backgroundColor: "rgba(16, 185, 129, 0.75)",
              borderColor: "#10b981",
              borderWidth: 1,
            },
            {
              label: "Mean Search Nodes",
              data: meanNodes,
              backgroundColor: "rgba(245, 158, 11, 0.75)",
              borderColor: "#f59e0b",
              borderWidth: 1,
              yAxisID: "y1",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              title: { display: true, text: "PAR2 (sec)", color: "#10b981" },
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: { color: "#94a3b8" },
            },
            y1: {
              beginAtZero: true,
              position: "right",
              title: { display: true, text: "Search Nodes", color: "#f59e0b" },
              grid: { display: false },
              ticks: { color: "#94a3b8" },
            },
            x: {
              grid: { display: false },
              ticks: { color: "#cbd5e1" },
            },
          },
          plugins: {
            legend: { labels: { color: "#f8fafc" } },
          },
        },
      });
    }

    // 3. Dolan-Moré Curves
    renderDolanMoreCurve();
  }

  function renderDolanMoreCurve() {
    const ctx = document.getElementById("dolan-more-chart");
    if (!ctx) return;
    if (dolanMoreChart) {
      try { dolanMoreChart.destroy(); } catch (e) {}
    }

    if (!benchmarkResults || benchmarkResults.length === 0) return;

    // Group runs by instance
    const instances = {};
    benchmarkResults.forEach(r => {
      if (!r || !r.instance) return;
      if (!instances[r.instance]) instances[r.instance] = {};
      const t = Number(r.wall_time_sec);
      instances[r.instance][r.heuristic] = isNaN(t) ? 999 : t;
    });

    const foundHeuristics = Array.from(new Set(benchmarkResults.map(r => r.heuristic))).filter(Boolean);
    const heuristics = foundHeuristics.length > 0 ? foundHeuristics : ["dom_wdeg", "mrv_degree", "mrv"];
    const colors = ["#00f2fe", "#10b981", "#a855f7", "#ec4899", "#3b82f6", "#f59e0b", "#94a3b8", "#f43f5e", "#14b8a6", "#8b5cf6", "#fbbf24"];

    const ratios = {};
    heuristics.forEach(h => ratios[h] = []);

    const validInstances = Object.values(instances).filter(hMap => Object.keys(hMap).length > 0);
    if (validInstances.length === 0) return;

    validInstances.forEach(instHeurs => {
      const times = Object.values(instHeurs);
      const minVal = Math.min(...times);
      heuristics.forEach(h => {
        const val = instHeurs[h] !== undefined ? instHeurs[h] : 999;
        const ratio = (val <= 0 || minVal <= 0) ? 1.0 : val / minVal;
        ratios[h].push(isFinite(ratio) ? ratio : 999);
      });
    });

    const taus = [];
    for (let t = 1.0; t <= 5.0; t += 0.2) taus.push(parseFloat(t.toFixed(1)));

    const datasets = heuristics.map((h, i) => {
      const data = taus.map(t => {
        if (!ratios[h] || ratios[h].length === 0) return 0;
        const count = ratios[h].filter(r => r <= t).length;
        return +(count / ratios[h].length).toFixed(3);
      });
      return {
        label: h,
        data: data,
        borderColor: colors[i % colors.length],
        borderWidth: 2,
        tension: 0.2,
        fill: false,
      };
    });

    dolanMoreChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: taus,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Performance Ratio Tau (Wall Time)", color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" },
          },
          y: {
            title: { display: true, text: "P(r_{p, s} <= Tau)", color: "#94a3b8" },
            min: 0,
            max: 1.05,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" },
          },
        },
        plugins: {
          legend: { labels: { color: "#f8fafc" } },
        },
      },
    });
  }

  // Dynamic filter options update
  function updateFilterHeuristicOptions() {
    const filterHeur = document.getElementById("filter-heuristic");
    if (!filterHeur) return;
    const currentVal = filterHeur.value;
    const uniqueHeurs = Array.from(new Set(benchmarkResults.map(r => r.heuristic))).filter(Boolean);

    filterHeur.innerHTML = '<option value="all">All Heuristics</option>';
    uniqueHeurs.forEach(h => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.innerText = h;
      if (h === currentVal) opt.selected = true;
      filterHeur.appendChild(opt);
    });
  }

  // Raw Results Explorer
  const filterSearch = document.getElementById("filter-search");
  const filterHeuristic = document.getElementById("filter-heuristic");
  const filterFamily = document.getElementById("filter-family");

  if (filterSearch) filterSearch.addEventListener("input", renderExplorerTable);
  if (filterHeuristic) filterHeuristic.addEventListener("change", renderExplorerTable);
  if (filterFamily) filterFamily.addEventListener("change", renderExplorerTable);

  function renderExplorerTable() {
    const tbody = document.getElementById("runs-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const searchInput = document.getElementById("filter-search");
    const filterHeur = document.getElementById("filter-heuristic");
    const filterFam = document.getElementById("filter-family");

    const search = searchInput ? searchInput.value.trim().toLowerCase() : "";
    const heurFilter = filterHeur ? filterHeur.value : "all";
    const famFilter = filterFam ? filterFam.value : "all";

    const filtered = (benchmarkResults || []).filter(r => {
      if (!r) return false;
      const instName = String(r.instance || "").toLowerCase();
      if (search && !instName.includes(search)) return false;
      if (heurFilter !== "all" && r.heuristic !== heurFilter) return false;
      if (famFilter !== "all" && r.family !== famFilter) return false;
      return true;
    });

    const countLabel = document.getElementById("table-counts-label");
    if (countLabel) {
      countLabel.innerText = `Showing ${filtered.length} / ${(benchmarkResults || []).length} runs`;
    }

    if (filtered.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="9" style="text-align: center; color: var(--text-dim); padding: 2.5rem; font-family: var(--font-mono);">No matching runs found for active filters.</td>`;
      tbody.appendChild(tr);
      return;
    }

    const fragment = document.createDocumentFragment();
    const toRender = filtered.slice(0, 300);

    toRender.forEach(r => {
      const tr = document.createElement("tr");
      const timeNum = Number(r.wall_time_sec);
      const timeStr = isNaN(timeNum) ? (r.wall_time_sec || "0.0000") : timeNum.toFixed(4);
      const statusClass = r.status === "SOLVED" ? "badge-status-solved" : "badge-status-timeout";

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 600;">${r.instance || "-"}</td>
        <td>${r.family || "-"}</td>
        <td>${r.difficulty || "-"}</td>
        <td><span class="badge-heuristic">${r.heuristic || "-"}</span></td>
        <td><span class="${statusClass}">${r.status || "UNKNOWN"}</span></td>
        <td>${r.nodes ?? 0}</td>
        <td>${r.backtracks ?? 0}</td>
        <td>${r.propagation_calls ?? 0}</td>
        <td>${timeStr}</td>
      `;
      fragment.appendChild(tr);
    });

    tbody.appendChild(fragment);

    if (filtered.length > 300) {
      const noteTr = document.createElement("tr");
      noteTr.innerHTML = `<td colspan="9" style="text-align: center; color: var(--text-dim); font-size: 0.8rem; padding: 0.75rem;">Displaying first 300 of ${filtered.length} runs.</td>`;
      tbody.appendChild(noteTr);
    }
  }
});
