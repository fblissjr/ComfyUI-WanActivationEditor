// Global variables
let currentTab = "overview";
let charts = {};

// VAE constants from WanVideoWrapper
const VAE_STRIDE = [4, 8, 8];
const PATCH_SIZE = [1, 2, 2];

// Initialize on page load
document.addEventListener("DOMContentLoaded", function () {
  updateMetrics();
  initializeCharts();
  initializeBlockGrid();
  updateBlockInfo();
  drawPipelineFlow();
});

// Tab switching
function showTab(tabName) {
  // Hide all tab contents
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.add("hidden");
  });

  // Remove active class from all tabs
  document.querySelectorAll("nav button").forEach((tab) => {
    tab.classList.remove("active-tab");
  });

  // Show selected tab content
  document.getElementById(`content-${tabName}`).classList.remove("hidden");
  document.getElementById(`tab-${tabName}`).classList.add("active-tab");

  currentTab = tabName;

  // Update charts when switching tabs
  if (tabName === "text-encoding") {
    updateTokenChart();
  } else if (tabName === "latent-space") {
    updateLatentCharts();
  } else if (tabName === "memory") {
    updateMemoryCharts();
  } else if (tabName === "activation") {
    updateBlendingChart();
  }
}

// Update computed metrics
function updateMetrics() {
  const width = parseInt(document.getElementById("width").value);
  const height = parseInt(document.getElementById("height").value);
  const frames = parseInt(document.getElementById("frames").value);

  // Calculate latent dimensions
  const latentT = Math.floor((frames - 1) / VAE_STRIDE[0]) + 1;
  const latentH = Math.floor(height / VAE_STRIDE[1]);
  const latentW = Math.floor(width / VAE_STRIDE[2]);

  // Calculate patches
  const numPatches =
    Math.floor(latentT / PATCH_SIZE[0]) *
    Math.floor(latentH / PATCH_SIZE[1]) *
    Math.floor(latentW / PATCH_SIZE[2]);

  // Calculate compression
  const pixelSize = width * height * frames * 3;
  const latentSize = 16 * latentT * latentH * latentW;
  const compression = pixelSize / latentSize;

  // Update display
  document.getElementById("latent-shape").textContent =
    `16×${latentT}×${latentH}×${latentW}`;
  document.getElementById("seq-length").textContent =
    numPatches.toLocaleString();
  document.getElementById("compression").textContent =
    `${compression.toFixed(1)}x`;

  // Update memory calculations
  updateMemoryCalculations();
}

// Event listeners for configuration changes
document.getElementById("width").addEventListener("change", updateMetrics);
document.getElementById("height").addEventListener("change", updateMetrics);
document.getElementById("frames").addEventListener("change", updateMetrics);
document.getElementById("batch-size").addEventListener("change", updateMetrics);
document
  .getElementById("precision")
  .addEventListener("change", updateMemoryCalculations);

// Initialize all charts
function initializeCharts() {
  // Token count chart
  const tokenCtx = document.getElementById("token-chart").getContext("2d");
  charts.tokenChart = new Chart(tokenCtx, {
    type: "bar",
    data: {
      labels: ["Positive Prompt", "Negative Prompt"],
      datasets: [
        {
          label: "Token Count",
          data: [0, 0],
          backgroundColor: ["rgba(34, 197, 94, 0.8)", "rgba(239, 68, 68, 0.8)"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 256,
          title: { display: true, text: "Number of Tokens" },
        },
      },
    },
  });

  // Memory comparison chart
  const memoryCtx = document.getElementById("memory-chart").getContext("2d");
  charts.memoryChart = new Chart(memoryCtx, {
    type: "bar",
    data: {
      labels: ["Pixel Space", "Latent Space"],
      datasets: [
        {
          label: "Memory (MB)",
          data: [0, 0],
          backgroundColor: ["rgba(239, 68, 68, 0.8)", "rgba(34, 197, 94, 0.8)"],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Memory (MB)" },
        },
      },
    },
  });

  // Block specialization chart
  const blockCtx = document
    .getElementById("block-specialization-chart")
    .getContext("2d");
  const blocks = Array.from({ length: 40 }, (_, i) => i);

  charts.blockSpecChart = new Chart(blockCtx, {
    type: "line",
    data: {
      labels: blocks,
      datasets: [
        {
          label: "Structure/Motion",
          data: blocks.map((i) => Math.exp(-i / 8)),
          borderColor: "rgb(59, 130, 246)",
          tension: 0.4,
        },
        {
          label: "Semantic Content",
          data: blocks.map((i) => Math.exp(-Math.pow(i - 20, 2) / 100)),
          borderColor: "rgb(34, 197, 94)",
          tension: 0.4,
        },
        {
          label: "Fine Details",
          data: blocks.map((i) => 1 / (1 + Math.exp(-(i - 30) / 3))),
          borderColor: "rgb(239, 68, 68)",
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: { title: { display: true, text: "Block Index" } },
        y: { title: { display: true, text: "Influence Level" } },
      },
    },
  });
}

// Update token chart based on prompts
function updateTokenChart() {
  const mainPrompt = document.getElementById("main-prompt").value;
  const negativePrompt = document.getElementById("negative-prompt").value;

  // Approximate token count (4 chars per token)
  const mainTokens = Math.min(Math.floor(mainPrompt.length / 4), 256);
  const negTokens = Math.min(Math.floor(negativePrompt.length / 4), 256);

  charts.tokenChart.data.datasets[0].data = [mainTokens, negTokens];
  charts.tokenChart.update();
}

// Update latent space charts
function updateLatentCharts() {
  const width = parseInt(document.getElementById("width").value);
  const height = parseInt(document.getElementById("height").value);
  const frames = parseInt(document.getElementById("frames").value);

  const pixelSize = width * height * frames * 3;
  const latentSize =
    16 *
    Math.floor((frames - 1) / 4 + 1) *
    Math.floor(height / 8) *
    Math.floor(width / 8);

  // Update memory chart
  charts.memoryChart.data.datasets[0].data = [
    (pixelSize * 4) / 1024 / 1024, // FP32 in MB
    (latentSize * 4) / 1024 / 1024,
  ];
  charts.memoryChart.update();

  // Update dimension chart if it exists
  if (!charts.dimensionChart) {
    const dimCtx = document.getElementById("dimension-chart").getContext("2d");
    charts.dimensionChart = new Chart(dimCtx, {
      type: "bar",
      data: {
        labels: ["Frames", "Height", "Width", "Channels"],
        datasets: [
          {
            label: "Pixel Space",
            data: [frames, height, width, 3],
            backgroundColor: "rgba(239, 68, 68, 0.8)",
          },
          {
            label: "Latent Space",
            data: [
              Math.floor((frames - 1) / 4 + 1),
              Math.floor(height / 8),
              Math.floor(width / 8),
              16,
            ],
            backgroundColor: "rgba(34, 197, 94, 0.8)",
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: {
            type: "logarithmic",
            title: { display: true, text: "Size" },
          },
        },
      },
    });
  } else {
    charts.dimensionChart.data.datasets[0].data = [frames, height, width, 3];
    charts.dimensionChart.data.datasets[1].data = [
      Math.floor((frames - 1) / 4 + 1),
      Math.floor(height / 8),
      Math.floor(width / 8),
      16,
    ];
    charts.dimensionChart.update();
  }
}

// Update memory charts and calculations
function updateMemoryCalculations() {
  const precision = document.getElementById("precision").value;
  const batchSize = parseInt(document.getElementById("batch-size").value);
  const width = parseInt(document.getElementById("width").value);
  const height = parseInt(document.getElementById("height").value);
  const frames = parseInt(document.getElementById("frames").value);

  const bytesPerParam = {
    fp32: 4,
    fp16: 2,
    fp8: 1,
  }[precision];

  // Model memory (in GB)
  const modelMemory = {
    "WanVideo (14B)": (14e9 * bytesPerParam) / 1e9,
    "UMT5-XXL (11B)": (11e9 * bytesPerParam) / 1e9,
    "VAE (500M)": (0.5e9 * bytesPerParam) / 1e9,
  };

  // Calculate activation memory
  const latentT = Math.floor((frames - 1) / 4 + 1);
  const latentH = Math.floor(height / 8);
  const latentW = Math.floor(width / 8);
  const numPatches =
    Math.floor(latentT / 1) * Math.floor(latentH / 2) * Math.floor(latentW / 2);

  const activationMemory = {
    "Text Embeddings": (batchSize * 256 * 4096 * 4) / 1e9,
    "Latent Tensors": (batchSize * 16 * latentT * latentH * latentW * 4) / 1e9,
    "Attention Maps": (batchSize * 40 * numPatches * numPatches * 4) / 1e9,
    "FFN Activations": (batchSize * 40 * numPatches * 13824 * 4) / 1e9,
  };

  const totalModel = Object.values(modelMemory).reduce((a, b) => a + b, 0);
  const totalActivation = Object.values(activationMemory).reduce(
    (a, b) => a + b,
    0,
  );
  const totalMemory = totalModel + totalActivation + 2; // +2GB buffer

  // Update display
  document.getElementById("model-memory-total").textContent =
    `${totalModel.toFixed(1)} GB`;
  document.getElementById("activation-memory-total").textContent =
    `${totalActivation.toFixed(1)} GB`;
  document.getElementById("total-memory").textContent =
    `${totalMemory.toFixed(1)} GB`;

  // Update charts if on memory tab
  if (currentTab === "memory") {
    updateMemoryCharts();
  }
}

// Update memory visualization charts
function updateMemoryCharts() {
  const precision = document.getElementById("precision").value;
  const batchSize = parseInt(document.getElementById("batch-size").value);

  const bytesPerParam = {
    fp32: 4,
    fp16: 2,
    fp8: 1,
  }[precision];

  // Model memory chart
  if (!charts.modelMemoryChart) {
    const modelCtx = document
      .getElementById("model-memory-chart")
      .getContext("2d");
    charts.modelMemoryChart = new Chart(modelCtx, {
      type: "pie",
      data: {
        labels: ["WanVideo (14B)", "T5-XXL (11B)", "VAE (500M)"],
        datasets: [
          {
            data: [
              (14e9 * bytesPerParam) / 1e9,
              (11e9 * bytesPerParam) / 1e9,
              (0.5e9 * bytesPerParam) / 1e9,
            ],
            backgroundColor: [
              "rgba(59, 130, 246, 0.8)",
              "rgba(34, 197, 94, 0.8)",
              "rgba(251, 191, 36, 0.8)",
            ],
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom" },
          title: {
            display: true,
            text: `Model Memory (${precision.toUpperCase()})`,
          },
        },
      },
    });
  } else {
    charts.modelMemoryChart.data.datasets[0].data = [
      (14e9 * bytesPerParam) / 1e9,
      (11e9 * bytesPerParam) / 1e9,
      (0.5e9 * bytesPerParam) / 1e9,
    ];
    charts.modelMemoryChart.options.plugins.title.text = `Model Memory (${precision.toUpperCase()})`;
    charts.modelMemoryChart.update();
  }

  // Activation memory chart
  const width = parseInt(document.getElementById("width").value);
  const height = parseInt(document.getElementById("height").value);
  const frames = parseInt(document.getElementById("frames").value);

  const latentT = Math.floor((frames - 1) / 4 + 1);
  const latentH = Math.floor(height / 8);
  const latentW = Math.floor(width / 8);
  const numPatches =
    Math.floor(latentT / 1) * Math.floor(latentH / 2) * Math.floor(latentW / 2);

  if (!charts.activationMemoryChart) {
    const actCtx = document
      .getElementById("activation-memory-chart")
      .getContext("2d");
    charts.activationMemoryChart = new Chart(actCtx, {
      type: "bar",
      data: {
        labels: [
          "Text Embeddings",
          "Latent Tensors",
          "Attention Maps",
          "FFN Activations",
        ],
        datasets: [
          {
            label: "Memory (GB)",
            data: [
              (batchSize * 256 * 4096 * 4) / 1e9,
              (batchSize * 16 * latentT * latentH * latentW * 4) / 1e9,
              (batchSize * 40 * numPatches * numPatches * 4) / 1e9,
              (batchSize * 40 * numPatches * 13824 * 4) / 1e9,
            ],
            backgroundColor: "rgba(59, 130, 246, 0.8)",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: "Activation Memory" },
        },
        scales: {
          y: {
            type: "logarithmic",
            title: { display: true, text: "Memory (GB)" },
          },
        },
      },
    });
  } else {
    charts.activationMemoryChart.data.datasets[0].data = [
      (batchSize * 256 * 4096 * 4) / 1e9,
      (batchSize * 16 * latentT * latentH * latentW * 4) / 1e9,
      (batchSize * 40 * numPatches * numPatches * 4) / 1e9,
      (batchSize * 40 * numPatches * 13824 * 4) / 1e9,
    ];
    charts.activationMemoryChart.update();
  }
}

// Block info update
function updateBlockInfo() {
  const blockNum = parseInt(document.getElementById("block-selector").value);
  document.getElementById("block-number").textContent = blockNum;

  let category, description, color;
  if (blockNum < 10) {
    category = "Early Block";
    description =
      "Controls structure and motion. These blocks establish the fundamental spatial and temporal relationships in the video.";
    color = "bg-blue-100";
  } else if (blockNum < 30) {
    category = "Middle Block";
    description =
      "Handles semantic content. These blocks process object relationships and scene semantics.";
    color = "bg-green-100";
  } else {
    category = "Late Block";
    description =
      "Refines fine details. These blocks add textures, enhance sharpness, and polish the final output.";
    color = "bg-red-100";
  }

  const width = parseInt(document.getElementById("width").value);
  const height = parseInt(document.getElementById("height").value);
  const frames = parseInt(document.getElementById("frames").value);

  const latentT = Math.floor((frames - 1) / 4 + 1);
  const latentH = Math.floor(height / 8);
  const latentW = Math.floor(width / 8);
  const numPatches =
    Math.floor(latentT / 1) * Math.floor(latentH / 2) * Math.floor(latentW / 2);

  const selfAttnOps = numPatches * numPatches;
  const ffnOps = numPatches * 13824;
  const blockMemory = ((selfAttnOps + ffnOps) * 4) / 1024 / 1024;

  document.getElementById("block-info").innerHTML = `
        <div class="${color} rounded p-3 mb-3">
            <p class="font-semibold">${category}</p>
            <p class="text-sm text-gray-700 mt-1">${description}</p>
        </div>
        <div class="grid grid-cols-3 gap-4 text-sm">
            <div>
                <p class="text-gray-600">Self-Attention Ops</p>
                <p class="font-mono font-bold">${selfAttnOps.toLocaleString()}</p>
            </div>
            <div>
                <p class="text-gray-600">FFN Ops</p>
                <p class="font-mono font-bold">${ffnOps.toLocaleString()}</p>
            </div>
            <div>
                <p class="text-gray-600">Block Memory</p>
                <p class="font-mono font-bold">${blockMemory.toFixed(1)} MB</p>
            </div>
        </div>
    `;
}

// Initialize block grid for activation editor
function initializeBlockGrid() {
  const grid = document.getElementById("block-grid");
  grid.innerHTML = "";

  for (let i = 0; i < 40; i++) {
    const blockDiv = document.createElement("div");
    blockDiv.className =
      "block-item flex flex-col items-center p-2 border rounded cursor-pointer hover:bg-gray-100";
    blockDiv.innerHTML = `
            <input type="checkbox" id="block-${i}" class="block-checkbox" onchange="updateActiveBlockCount()">
            <label for="block-${i}" class="text-xs mt-1">${i}</label>
        `;
    grid.appendChild(blockDiv);
  }
}

// Update block pattern based on preset
function updateBlockPattern() {
  const preset = document.getElementById("pattern-preset").value;
  const checkboxes = document.querySelectorAll(".block-checkbox");

  checkboxes.forEach((cb, i) => {
    switch (preset) {
      case "early":
        cb.checked = i < 10;
        break;
      case "middle":
        cb.checked = i >= 10 && i < 30;
        break;
      case "late":
        cb.checked = i >= 30;
        break;
      case "alternating":
        cb.checked = i % 2 === 0;
        break;
      case "sparse":
        cb.checked = i % 4 === 0;
        break;
      case "first_half":
        cb.checked = i < 20;
        break;
      case "second_half":
        cb.checked = i >= 20;
        break;
      case "custom":
        // Keep current state
        break;
    }
  });

  updateActiveBlockCount();
}

// Update active block count
function updateActiveBlockCount() {
  const activeCount = document.querySelectorAll(
    ".block-checkbox:checked",
  ).length;
  document.getElementById("active-block-count").textContent = activeCount;
}

// Update strength value display
function updateStrengthValue() {
  const strength = document.getElementById("injection-strength").value;
  document.getElementById("strength-value").textContent = strength;
  updateBlendingChart();
}

// Update blending chart
function updateBlendingChart() {
  const strength = parseFloat(
    document.getElementById("injection-strength").value,
  );

  if (!charts.blendingChart) {
    const ctx = document.getElementById("blending-chart").getContext("2d");
    const strengths = Array.from({ length: 101 }, (_, i) => i / 100);

    charts.blendingChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: strengths,
        datasets: [
          {
            label: "Main Prompt",
            data: strengths.map((s) => 1 - s),
            borderColor: "rgb(59, 130, 246)",
            borderWidth: 3,
            tension: 0,
          },
          {
            label: "Injected Prompt",
            data: strengths,
            borderColor: "rgb(239, 68, 68)",
            borderWidth: 3,
            tension: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          annotation: {
            annotations: {
              line1: {
                type: "line",
                xMin: strength,
                xMax: strength,
                borderColor: "rgb(34, 197, 94)",
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  display: true,
                  content: `Current: ${strength}`,
                  position: "start",
                },
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Injection Strength" },
            min: 0,
            max: 1,
          },
          y: {
            title: { display: true, text: "Influence Level" },
            min: 0,
            max: 1,
          },
        },
      },
    });
  } else {
    // Update annotation
    charts.blendingChart.options.plugins.annotation.annotations.line1.xMin =
      strength;
    charts.blendingChart.options.plugins.annotation.annotations.line1.xMax =
      strength;
    charts.blendingChart.options.plugins.annotation.annotations.line1.label.content = `Current: ${strength}`;
    charts.blendingChart.update();
  }
}

// Draw pipeline flow using D3
function drawPipelineFlow() {
  const container = d3.select("#pipeline-flow");
  const width = container.node().offsetWidth;
  const height = 500;

  // Clear existing content
  container.selectAll("*").remove();

  const svg = container
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  // Define gradient
  const defs = svg.append("defs");

  const gradient = defs
    .append("linearGradient")
    .attr("id", "flow-gradient")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  gradient
    .append("stop")
    .attr("offset", "0%")
    .style("stop-color", "#3b82f6")
    .style("stop-opacity", 0.8);

  gradient
    .append("stop")
    .attr("offset", "100%")
    .style("stop-color", "#1e40af")
    .style("stop-opacity", 0.8);

  // Pipeline stages based on actual WanVideoWrapper flow
  const stages = [
    {
      name: "Text Input",
      shape: "[prompts]",
      desc: "User prompts",
      color: "#FF6B6B",
      y: 50,
    },
    {
      name: "T5-XXL Encoder",
      shape: "[B, L, 4096]",
      desc: "Text embeddings",
      color: "#4ECDC4",
      y: 130,
    },
    {
      name: "Latent Noise",
      shape: "[B, 16, T, H, W]",
      desc: "Initial latents",
      color: "#45B7D1",
      y: 210,
    },
    {
      name: "WanVideo Transformer",
      shape: "[B, seq_len, 5120]",
      desc: "40 blocks",
      color: "#FECA57",
      y: 290,
    },
    {
      name: "VAE Decoder",
      shape: "[B, 3, F, H, W]",
      desc: "RGB frames",
      color: "#6C5CE7",
      y: 370,
    },
  ];

  // Draw connections
  stages.forEach((stage, i) => {
    if (i < stages.length - 1) {
      svg
        .append("line")
        .attr("x1", width / 2)
        .attr("y1", stage.y + 40)
        .attr("x2", width / 2)
        .attr("y2", stages[i + 1].y)
        .attr("stroke", "#94a3b8")
        .attr("stroke-width", 2)
        .attr("marker-end", "url(#arrowhead)");
    }
  });

  // Arrow marker
  svg
    .append("defs")
    .append("marker")
    .attr("id", "arrowhead")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 5)
    .attr("refY", 5)
    .attr("markerWidth", 5)
    .attr("markerHeight", 5)
    .append("path")
    .attr("d", "M 0 0 L 10 5 L 0 10 z")
    .attr("fill", "#94a3b8");

  // Draw stages
  const stageGroups = svg
    .selectAll(".stage")
    .data(stages)
    .enter()
    .append("g")
    .attr("class", "stage flow-node")
    .attr("transform", (d) => `translate(${width / 2}, ${d.y})`);

  // Stage rectangles
  stageGroups
    .append("rect")
    .attr("x", -150)
    .attr("y", -20)
    .attr("width", 300)
    .attr("height", 40)
    .attr("rx", 5)
    .attr("fill", (d) => d.color)
    .attr("opacity", 0.8)
    .attr("stroke", "#333")
    .attr("stroke-width", 2);

  // Stage names
  stageGroups
    .append("text")
    .attr("text-anchor", "middle")
    .attr("y", -5)
    .attr("font-weight", "bold")
    .attr("font-size", "14px")
    .text((d) => d.name);

  // Stage shapes
  stageGroups
    .append("text")
    .attr("text-anchor", "middle")
    .attr("y", 10)
    .attr("font-size", "11px")
    .attr("font-family", "monospace")
    .text((d) => d.shape);

  // Stage descriptions
  stageGroups
    .append("text")
    .attr("text-anchor", "middle")
    .attr("y", 35)
    .attr("font-size", "10px")
    .attr("font-style", "italic")
    .attr("fill", "#666")
    .text((d) => d.desc);

  // Add key technical details
  const details = svg
    .append("g")
    .attr("transform", `translate(${width - 200}, 20)`);

  const detailsData = [
    "VAE Stride: (4, 8, 8)",
    "Patch Size: (1, 2, 2)",
    "Hidden Dim: 5,120",
    "FFN Dim: 13,824",
    "Attention Heads: 40",
  ];

  details
    .selectAll("text")
    .data(detailsData)
    .enter()
    .append("text")
    .attr("y", (d, i) => i * 20)
    .attr("font-size", "12px")
    .attr("font-family", "monospace")
    .attr("fill", "#475569")
    .text((d) => d);
}

// Event listeners for prompt changes
document
  .getElementById("main-prompt")
  .addEventListener("input", updateTokenChart);
document
  .getElementById("negative-prompt")
  .addEventListener("input", updateTokenChart);

// Calculate prompt difference (simplified estimation)
function calculatePromptDifference() {
  const mainPrompt = document.getElementById("activation-main-prompt").value;
  const injectionPrompt = document.getElementById(
    "activation-injection-prompt",
  ).value;

  // Simplified semantic difference calculation
  // In reality, this would use T5 embeddings and cosine similarity

  // Extract key concepts (simplified)
  const mainWords = new Set(
    mainPrompt
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 3),
  );
  const injectionWords = new Set(
    injectionPrompt
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 3),
  );

  // Calculate overlap
  const intersection = new Set(
    [...mainWords].filter((x) => injectionWords.has(x)),
  );
  const union = new Set([...mainWords, ...injectionWords]);

  // Jaccard distance as a proxy for semantic difference
  const similarity = intersection.size / union.size;
  const difference = (1 - similarity) * 100;

  // Check for domain shifts (simplified heuristic)
  const domainShifts = [
    ["indoor", "outdoor"],
    ["day", "night"],
    ["modern", "ancient"],
    ["natural", "synthetic"],
    ["calm", "chaotic"],
    ["realistic", "abstract"],
    ["urban", "nature"],
    ["underwater", "aerial"],
    ["mechanical", "organic"],
  ];

  let domainBonus = 0;
  for (const [term1, term2] of domainShifts) {
    if (
      (mainPrompt.includes(term1) && injectionPrompt.includes(term2)) ||
      (mainPrompt.includes(term2) && injectionPrompt.includes(term1))
    ) {
      domainBonus += 15;
    }
  }

  const totalDifference = Math.min(Math.round(difference + domainBonus), 95);

  const resultDiv = document.getElementById("prompt-difference-result");
  if (totalDifference < 30) {
    resultDiv.innerHTML = `<span class="text-red-600">~${totalDifference}% - Too Similar! Prompts need to be more different.</span>`;
  } else if (totalDifference < 50) {
    resultDiv.innerHTML = `<span class="text-yellow-600">~${totalDifference}% - Borderline. Consider making prompts more distinct.</span>`;
  } else {
    resultDiv.innerHTML = `<span class="text-green-600">~${totalDifference}% - Good! Prompts are sufficiently different.</span>`;
  }

  // Also show prompt length warning if needed
  if (mainPrompt.length < 100 || injectionPrompt.length < 100) {
    resultDiv.innerHTML +=
      '<br><span class="text-xs text-gray-600">Note: WAN-style prompts should be detailed paragraphs (100+ chars)</span>';
  }
}
