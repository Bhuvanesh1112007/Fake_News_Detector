const API_BASE = "http://localhost:8000/api";

// Elements
const serverStatus = document.getElementById("serverStatus");
const tabTextBtn = document.getElementById("tabTextBtn");
const tabUrlBtn = document.getElementById("tabUrlBtn");
const tabPageBtn = document.getElementById("tabPageBtn");
const paneText = document.getElementById("paneText");
const paneUrl = document.getElementById("paneUrl");
const panePage = document.getElementById("panePage");
const newsTitle = document.getElementById("newsTitle");
const newsBody = document.getElementById("newsBody");
const newsUrl = document.getElementById("newsUrl");
const pageInfo = document.getElementById("pageInfo");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const btnSpinner = document.getElementById("btnSpinner");
const resultsCard = document.getElementById("resultsCard");
const verdictBanner = document.getElementById("verdictBanner");
const verdictText = document.getElementById("verdictText");
const confidenceText = document.getElementById("confidenceText");
const probBarFill = document.getElementById("probBarFill");
const realProbText = document.getElementById("realProbText");
const fakeProbText = document.getElementById("fakeProbText");
const riskBadge = document.getElementById("riskBadge");
const heatmapBox = document.getElementById("heatmapBox");
const reasonsList = document.getElementById("reasonsList");
const presetFake = document.getElementById("presetFake");
const presetReal = document.getElementById("presetReal");

let currentMode = "text"; // "text" | "url" | "page"
let activeTabUrl = "";
let activeTabTitle = "";

// 1. Check Server Health
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      serverStatus.textContent = "AI Online (79-dim)";
      serverStatus.className = "badge-status online";
    } else {
      serverStatus.textContent = "Server Error";
      serverStatus.className = "badge-status offline";
    }
  } catch (err) {
    serverStatus.textContent = "Backend Offline";
    serverStatus.className = "badge-status offline";
  }
}

// 2. Tab Navigation
function setTab(mode) {
  currentMode = mode;
  [tabTextBtn, tabUrlBtn, tabPageBtn].forEach(b => b.classList.remove("active"));
  [paneText, paneUrl, panePage].forEach(p => p.classList.remove("active"));

  if (mode === "text") {
    tabTextBtn.classList.add("active");
    paneText.classList.add("active");
  } else if (mode === "url") {
    tabUrlBtn.classList.add("active");
    paneUrl.classList.add("active");
  } else if (mode === "page") {
    tabPageBtn.classList.add("active");
    panePage.classList.add("active");
    fetchCurrentTabInfo();
  }
}

tabTextBtn.addEventListener("click", () => setTab("text"));
tabUrlBtn.addEventListener("click", () => setTab("url"));
tabPageBtn.addEventListener("click", () => setTab("page"));

// 3. Fetch Active Tab Info
async function fetchCurrentTabInfo() {
  if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0]) {
        activeTabUrl = tabs[0].url || "";
        activeTabTitle = tabs[0].title || "";
        pageInfo.textContent = `Active Page: "${activeTabTitle.substring(0, 50)}..." (${activeTabUrl.substring(0, 40)}...)`;
      }
    });
  } else {
    pageInfo.textContent = "Active tab extraction is available when loaded in Chrome.";
  }
}

// 4. Presets
presetFake.addEventListener("click", () => {
  setTab("text");
  newsTitle.value = "Government Distributing Free Laptops to WhatsApp Users";
  newsBody.value = "Every student who forwards this WhatsApp link to 5 groups will get a free high-end laptop delivered to their address!";
});

presetReal.addEventListener("click", () => {
  setTab("text");
  newsTitle.value = "Clinical Trial Demonstrates Efficacy of New Immunotherapy Drug";
  newsBody.value = "Researchers published findings in The Lancet showing a 40 percent improvement in five-year survival rates for oncology patients.";
});

// 5. Run Prediction
analyzeBtn.addEventListener("click", async () => {
  let endpoint = `${API_BASE}/predict`;
  let payload = {};

  if (currentMode === "text") {
    const title = newsTitle.value.trim();
    const text = newsBody.value.trim();
    if (!title || !text) {
      alert("Please enter both title and body text.");
      return;
    }
    payload = { title, text, source: "Web", author: "Unknown", shares: 1000.0, sentiment: 0.0 };
  } else if (currentMode === "url") {
    const url = newsUrl.value.trim();
    if (!url) {
      alert("Please enter a valid news URL.");
      return;
    }
    endpoint = `${API_BASE}/predict-url`;
    payload = { url };
  } else if (currentMode === "page") {
    if (!activeTabUrl) {
      alert("Could not detect active tab URL.");
      return;
    }
    endpoint = `${API_BASE}/predict-url`;
    payload = { url: activeTabUrl };
  }

  // Loading UI
  btnText.textContent = "Analyzing...";
  btnSpinner.classList.remove("hidden");
  analyzeBtn.disabled = true;
  resultsCard.classList.add("hidden");

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis request failed.");
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btnText.textContent = "Analyze Article";
    btnSpinner.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
});

// 6. Render Results
function renderResults(data) {
  resultsCard.classList.remove("hidden");

  const isFake = data.prediction === "FAKE";
  verdictBanner.className = `verdict-banner ${isFake ? "fake" : "real"}`;
  verdictText.textContent = data.prediction === "FAKE" ? "⚠️ FAKE NEWS DETECTED" : "✅ VERIFIED AUTHENTIC";
  confidenceText.textContent = isFake ? `${data.confidence_fake}% Fake` : `${data.confidence_real}% Real`;

  probBarFill.style.width = `${data.confidence_fake}%`;
  realProbText.textContent = `Real: ${data.confidence_real}%`;
  fakeProbText.textContent = `Fake: ${data.confidence_fake}%`;
  riskBadge.textContent = data.risk_level || "Risk Assessed";

  // Render Token Heatmap
  heatmapBox.innerHTML = "";
  if (data.token_attributions && data.token_attributions.length > 0) {
    data.token_attributions.forEach(item => {
      const span = document.createElement("span");
      span.textContent = item.token + " ";
      if (item.type === "fake_signal") {
        span.className = "tok-fake";
      } else if (item.type === "real_signal") {
        span.className = "tok-real";
      }
      heatmapBox.appendChild(span);
    });
  } else {
    heatmapBox.textContent = "No word token attributions returned.";
  }

  // Render Key Reasons
  reasonsList.innerHTML = "";
  if (data.key_reasons && data.key_reasons.length > 0) {
    data.key_reasons.slice(0, 3).forEach(reason => {
      const li = document.createElement("li");
      li.textContent = reason;
      reasonsList.appendChild(li);
    });
  }
}

// Init
checkHealth();
