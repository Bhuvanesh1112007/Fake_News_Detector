/**
 * Frontend Controller for VeriShield AI News Verification System
 * Handcrafted, Minimalist & Professional Architecture
 */

const API_BASE = window.location.origin;

// Benchmark Test Presets
const PRESET_SAMPLES = {
    real_nasa: {
        title: "NASA James Webb Space Telescope Captures Deepest Image of Early Universe",
        text: "NASA's James Webb Space Telescope has produced the deepest and sharpest infrared image of the distant universe to date. Known as Webb's First Deep Field, this image of galaxy cluster SMACS 0723 is overflowing with detail. Thousands of galaxies – including the faintest objects ever observed in the infrared – have appeared in Webb's view for the first time. Scientific teams across the globe are analyzing the spectroscopic data to understand the formation of early stars.",
        source: "NASA Press Release",
        author: "Dr. Sarah Jenkins",
        shares: 14200
    },
    real_fed: {
        title: "Federal Reserve Announces Interest Rate Decision Following Policy Meeting",
        text: "The Federal Reserve concluded its two-day monetary policy meeting today, announcing an adjustment in the benchmark federal funds rate. Federal Reserve officials indicated that the decision aims to balance ongoing inflation management with sustainable economic growth and labor market stability. Analysts from major financial institutions noted that the central bank's forward guidance remains contingent on incoming macroeconomic data.",
        source: "Reuters Financial",
        author: "Marcus Vance",
        shares: 8900
    },
    fake_conspiracy: {
        title: "SHOCKING PROOF: Secret Satellite Emits Mind-Control Frequencies to Control Elections",
        text: "BREAKING NEWS THAT THE MAINSTREAM MEDIA REFUSES TO SHOW YOU! Leaked classified files from an anonymous insider prove that secret military satellites are beaming dangerous 5G radiation frequencies directly into people's brains to manipulate how they vote! Millions of citizens are being hypnotized without knowing it! Share this viral post before it gets banned everywhere forever! Wake up sheeple!",
        source: "ConspiracyTruthWire247.net",
        author: "PatriotX_99",
        shares: 85000
    },
    fake_cure: {
        title: "Drinking Boiled Lemon Peel Juice Cures All Cancers in 48 Hours, Doctors BANNED from Revealing",
        text: "Big Pharma is furious! A miraculous ancient home remedy made of boiled lemon peels and baking soda dissolves 100% of cancer cells in just two days! Corrupt medical authorities have silenced honest doctors who tried to tell the public the truth about this free miracle cure. You will never need chemotherapy or doctors again! Click here now to see the secret recipe they are trying to delete from the internet!",
        source: "MiracleHealthSecretNews.com",
        author: "HealerGuru7",
        shares: 120000
    },
    ta_fake_cure: {
        title: "சுடுதண்ணீரில் எலுமிச்சை சாறு குடித்தால் 100% புற்றுநோய் குணமாகும் - மறைக்கப்பட்ட ரகசியம்!",
        text: "மருத்துவர்கள் மறைத்த உண்மை! எலுமிச்சை தோல் சாறு குடித்தால் ஒரே நாளில் புற்றுநோய் செல்கள் அழியும் என்று வெளிவந்துள்ளது. இதை உடனே பகிருங்கள்!",
        source: "TamilHealthSecrets.net",
        author: "நாட்டு வைத்தியர்",
        shares: 75000
    },
    hi_fake_money: {
        title: "प्रधानमंत्री डिजिटल राहत योजना: व्हाट्सएप पर शेयर करने वाले सभी नागरिकों को ₹50,000 मुफ्त!",
        text: "सरकार ने आपातकालीन वित्तीय सहायता शुरू की है। नीचे दिए गए लिंक पर क्लिक करके बैंक विवरण दर्ज करें और तुरंत खाते में राशि प्राप्त करें!",
        source: "FreeWelfareYojana2026.org",
        author: "वायरल समाचार",
        shares: 95000
    },
    es_fake_5g: {
        title: "¡ALERTA! Filtran frecuencias secretas de antenas 5G que alteran el ADN humano",
        text: "Documentos confidenciales confirman que la radiación de las torres móviles manipula la mente de los ciudadanos. ¡Comparte antes de que lo censuren!",
        source: "VerdadOcultaGlobal.es",
        author: "InvestigadorX",
        shares: 45000
    }
};

let modelsChart = null;
let importanceChart = null;
let radarChart = null;
let currentPredictionData = null;

// Theme Toggle (Light / Dark)
function initTheme() {
    const savedTheme = localStorage.getItem('verishield-theme') || 'light';
    setTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(nextTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('verishield-theme', theme);
    
    const icon = document.getElementById('theme-icon');
    if (icon) {
        if (theme === 'dark') {
            icon.className = 'fa-regular fa-sun';
        } else {
            icon.className = 'fa-regular fa-moon';
        }
    }

    // Refresh charts with theme-appropriate colors
    if (currentPredictionData) {
        renderMultimodalRadar(currentPredictionData);
    }
    if (document.getElementById('tab-analytics')?.classList.contains('active')) {
        fetchMetrics();
    }
    if (document.getElementById('tab-simulator')?.classList.contains('active')) {
        runPropagationSimulation();
    }
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.style.display = 'none';
        p.classList.remove('active');
    });
    document.querySelectorAll('.nav-tab').forEach(b => {
        b.classList.remove('active');
    });

    const targetPane = document.getElementById(`tab-${tabId}`);
    const targetBtn = document.getElementById(`tab-btn-${tabId}`);
    if (targetPane && targetBtn) {
        targetPane.style.display = 'flex';
        targetPane.classList.add('active');
        targetBtn.classList.add('active');
    }

    if (tabId === 'analytics') {
        fetchMetrics();
    } else if (tabId === 'history') {
        fetchHistory();
    } else if (tabId === 'threat') {
        fetchThreatStream();
    } else if (tabId === 'drift') {
        fetchActiveLearningStats();
    } else if (tabId === 'simulator') {
        runPropagationSimulation();
    } else if (tabId === 'adversarial') {
        runAdversarialStressTest();
    }
}

// Preset Loader
function loadSample(key) {
    const sample = PRESET_SAMPLES[key];
    if (!sample) return;
    document.getElementById('input-title').value = sample.title;
    document.getElementById('input-text').value = sample.text;
    document.getElementById('input-source').value = sample.source;
    document.getElementById('input-author').value = sample.author;
    document.getElementById('input-shares').value = sample.shares;
}

function clearInputs() {
    document.getElementById('predict-form').reset();
    document.getElementById('results-card').style.display = 'none';
    document.getElementById('results-placeholder').style.display = 'flex';
    currentPredictionData = null;
}

// Handle Form Submit
async function handlePrediction(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-submit');
    const originalText = btn.innerHTML;
    
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin" style="font-size: 0.75rem;"></i> Analyzing article...`;

    const title = document.getElementById('input-title').value.trim();
    const text = document.getElementById('input-text').value.trim();
    const source = document.getElementById('input-source').value.trim() || "General Web";
    const author = document.getElementById('input-author').value.trim() || "Unknown";
    const shares = parseInt(document.getElementById('input-shares').value) || 1000;

    const isMultilingual = /[\u0B80-\u0BFF\u0900-\u097F\u0C00-\u0C7F\u0600-\u06FF]/.test(title + " " + text);
    const endpoint = isMultilingual ? `${API_BASE}/api/predict-multilingual` : `${API_BASE}/api/predict`;

    const payload = isMultilingual 
        ? { title, text, source, author }
        : { title, text, source, author, shares };

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to analyze article");
        }

        const resData = await response.json();
        const finalData = isMultilingual ? resData.prediction_result : resData;
        
        if (isMultilingual) {
            finalData.language_meta = {
                code: resData.detected_language,
                name: resData.language_name,
                flag: resData.language_flag
            };
        }

        renderResults(finalData);
    } catch (err) {
        alert(`Prediction Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// Handle URL Prediction
async function handleURLPrediction() {
    const urlInput = document.getElementById('input-url');
    const url = urlInput.value.trim();
    if (!url) {
        alert("Please enter a valid news article URL");
        return;
    }

    const btn = document.getElementById('btn-fetch-url');
    const origBtnHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin" style="font-size: 0.75rem;"></i> Fetching...`;

    try {
        const response = await fetch(`${API_BASE}/api/predict-url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to scrape and verify article URL");
        }

        const data = await response.json();
        if (data.title) document.getElementById('input-title').value = data.title;
        if (data.source) document.getElementById('input-source').value = data.source;
        renderResults(data);
    } catch (err) {
        alert(`URL Verification Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = origBtnHtml;
    }
}

// Render Results & Explanations
function renderResults(data) {
    currentPredictionData = data;
    document.getElementById('results-placeholder').style.display = 'none';
    const resultsCard = document.getElementById('results-card');
    resultsCard.style.display = 'flex';

    const isFake = data.prediction === 'FAKE';
    const banner = document.getElementById('verdict-banner');
    const icon = document.getElementById('verdict-icon');
    const badge = document.getElementById('verdict-badge');
    const verdictTitle = document.getElementById('verdict-title');
    const riskCat = document.getElementById('risk-category');
    const confPct = document.getElementById('confidence-percentage');
    const confSub = document.getElementById('confidence-sub');
    const fbBadge = document.getElementById('firebase-sync-badge');

    banner.className = `verdict-banner ${isFake ? 'verdict-fake' : 'verdict-real'}`;
    
    if (isFake) {
        icon.className = 'fa-solid fa-triangle-exclamation verdict-icon';
        badge.textContent = 'Misinformation Risk';
        verdictTitle.textContent = 'Flagged as FAKE News';
        riskCat.textContent = `· ${data.risk_level}`;
        confPct.textContent = `${data.confidence_fake}%`;
        confPct.style.color = 'var(--status-fake-text)';
        confSub.textContent = `Fake Probability (${data.confidence_real}% Real)`;
    } else {
        icon.className = 'fa-solid fa-check verdict-icon';
        badge.textContent = 'Verified Authentic';
        verdictTitle.textContent = 'Classified as REAL News';
        riskCat.textContent = `· ${data.risk_level}`;
        confPct.textContent = `${data.confidence_real}%`;
        confPct.style.color = 'var(--status-real-text)';
        confSub.textContent = `Credibility Score (${data.confidence_fake}% Fake)`;
    }

    if (fbBadge) {
        let badgeText = `Persisted: ${data.database_provider || 'SQLite'}`;
        if (data.language_meta) {
            badgeText += ` · ${data.language_meta.flag} ${data.language_meta.name}`;
        }
        fbBadge.textContent = badgeText;
    }

    // Radar Chart
    renderMultimodalRadar(data);

    // Token Attribution Heatmap
    const tokenContainer = document.getElementById('token-heatmap-container');
    tokenContainer.innerHTML = '';
    
    data.token_attributions.forEach(item => {
        const span = document.createElement('span');
        const isHighAttn = item.bert_attention_weight && item.bert_attention_weight > 0.4;
        
        if (item.type === 'fake_signal') {
            span.className = 'token-fake';
            span.title = `Misinformation Signal (+${item.score})`;
        } else if (item.type === 'real_signal') {
            span.className = 'token-real';
            span.title = `Credibility Signal (${item.score})`;
        } else if (isHighAttn) {
            span.className = 'token-attn';
            span.title = `High Attention Weight (${item.bert_attention_weight})`;
        } else {
            span.className = 'token-neutral';
        }
        span.textContent = item.token + ' ';
        tokenContainer.appendChild(span);
    });

    // SHAP Feature Impact List
    const impactList = document.getElementById('feature-impact-list');
    impactList.innerHTML = '';
    data.top_features.slice(0, 6).forEach(f => {
        const isPushFake = f.direction === 'INCREASES_FAKE';
        const div = document.createElement('div');
        div.style.cssText = 'display: flex; align-items: center; justify-content: space-between; font-size: 0.75rem; padding: 0.5rem 0.75rem; background: var(--bg-surface-raised); border: 1px solid var(--border); border-radius: var(--radius-sm);';
        div.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: ${isPushFake ? 'var(--status-fake-text)' : 'var(--status-real-text)'};"></span>
                <span style="font-weight: 500; color: var(--text-primary);">${f.feature}</span>
                <span style="font-size: 0.6875rem; color: var(--text-muted); background: var(--bg-surface); padding: 0.1rem 0.35rem; border-radius: 3px; border: 1px solid var(--border);">${f.category}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem; font-family: var(--font-mono);">
                <span style="color: var(--text-muted); font-size: 0.6875rem;">${f.value}</span>
                <span style="font-weight: 600; font-size: 0.75rem; color: ${isPushFake ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">
                    ${isPushFake ? '+' : '-'}${f.shap_value}
                </span>
            </div>
        `;
        impactList.appendChild(div);
    });

    // Reasons List
    const reasonsList = document.getElementById('reasons-list');
    reasonsList.innerHTML = '';
    data.key_reasons.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        reasonsList.appendChild(li);
    });
}

// Multimodal Radar Chart
function renderMultimodalRadar(data) {
    const ctx = document.getElementById('multimodalRadarChart');
    if (!ctx) return;
    if (radarChart) radarChart.destroy();

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const isFake = data.prediction === 'FAKE';
    const conf = isFake ? data.confidence_fake : data.confidence_real;

    const linguisticScore = Math.min(100, Math.max(10, conf * 0.95));
    const userScore = isFake ? 25 : 85;
    const networkScore = isFake ? 88 : 35;
    const stylometricScore = isFake ? 82 : 20;

    const strokeColor = isFake ? '#b91c1c' : '#15803d';
    const fillColor = isFake 
        ? (isDark ? 'rgba(239, 68, 68, 0.2)' : 'rgba(185, 28, 28, 0.12)')
        : (isDark ? 'rgba(34, 197, 94, 0.2)' : 'rgba(21, 128, 61, 0.12)');

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                'BERT Linguistic (64-d)',
                'Source Credibility (6-d)',
                'Graph Diffusion (6-d)',
                'Stylometrics (3-d)'
            ],
            datasets: [{
                data: [linguisticScore, userScore, networkScore, stylometricScore],
                backgroundColor: fillColor,
                borderColor: strokeColor,
                borderWidth: 1.5,
                pointBackgroundColor: strokeColor,
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    angleLines: { color: isDark ? '#23252b' : '#e2e8f0' },
                    grid: { color: isDark ? '#1a1b20' : '#edf2f7' },
                    pointLabels: { 
                        color: isDark ? '#cbd5e1' : '#475569', 
                        font: { family: 'Inter', size: 10, weight: '500' } 
                    },
                    ticks: { display: false, min: 0, max: 100 }
                }
            }
        }
    });
}

// Export Forensic Audit Report
async function downloadCurrentAuditReport() {
    if (!currentPredictionData) {
        alert("Please analyze an article first before exporting an audit report.");
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/export-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prediction_data: currentPredictionData })
        });

        if (!res.ok) throw new Error("Failed to generate audit report.");

        const htmlStr = await res.text();
        const printWindow = window.open('', '_blank');
        if (printWindow) {
            printWindow.document.open();
            printWindow.document.write(htmlStr);
            printWindow.document.close();
        } else {
            alert("Popup blocked. Please allow popups to view the printable certificate.");
        }
    } catch (e) {
        alert(`Audit Report Error: ${e.message}`);
    }
}

// Action Launchers
function launchPropagationForCurrent() {
    if (!currentPredictionData) return;
    switchTab('simulator');
    document.getElementById('sim-input-headline').value = currentPredictionData.title || "";
    runPropagationSimulation();
}

function launchAdversarialAuditForCurrent() {
    if (!currentPredictionData) return;
    switchTab('adversarial');
    document.getElementById('adv-input-headline').value = currentPredictionData.title || "";
    runAdversarialStressTest();
}

// Social Spread Simulator
async function runPropagationSimulation() {
    const headline = document.getElementById('sim-input-headline')?.value || "Viral Claim Simulation";
    const nodes = parseInt(document.getElementById('sim-nodes-slider')?.value || 35);
    const pred = currentPredictionData ? currentPredictionData.prediction : "FAKE";
    const fakeConf = currentPredictionData ? currentPredictionData.confidence_fake : 85.0;

    try {
        const res = await fetch(`${API_BASE}/api/simulate/propagation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                headline: headline,
                prediction: pred,
                fake_confidence: fakeConf,
                network_size: nodes,
                viral_speed: 1.5
            })
        });

        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('sim-r0-score').textContent = data.cascade_metrics.basic_reproduction_number_R0;
        document.getElementById('sim-reach-pct').textContent = `Total Reach: ${data.cascade_metrics.total_population_reach_percent}%`;
        document.getElementById('sim-param-beta').textContent = data.cascade_metrics.transmission_rate_beta;
        document.getElementById('sim-param-gamma').textContent = data.cascade_metrics.containment_rate_gamma;
        document.getElementById('sim-param-infected').textContent = `${data.cascade_metrics.infected_spreaders_count} nodes`;
        document.getElementById('sim-param-factchecked').textContent = `${data.cascade_metrics.fact_checked_count} nodes`;
        
        const badge = document.getElementById('sim-cascade-badge');
        if (badge) badge.textContent = data.cascade_metrics.cascade_risk;

        drawPropagationGraph(data.graph_topology);
    } catch (e) {
        console.error("Propagation simulation error", e);
    }
}

function drawPropagationGraph(topology) {
    const canvas = document.getElementById('propagationCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Edges
    ctx.strokeStyle = isDark ? '#2e313a' : '#cbd5e1';
    ctx.lineWidth = 1;
    topology.edges.forEach(edge => {
        const src = topology.nodes[edge.source];
        const tgt = topology.nodes[edge.target];
        if (src && tgt) {
            ctx.beginPath();
            ctx.moveTo(src.x, src.y);
            ctx.lineTo(tgt.x, tgt.y);
            ctx.stroke();
        }
    });

    // Draw Nodes
    topology.nodes.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
        ctx.fillStyle = node.color;
        ctx.fill();
        ctx.strokeStyle = isDark ? '#131417' : '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    });
}

// Adversarial Stress Testing
async function runAdversarialStressTest() {
    const headline = document.getElementById('adv-input-headline')?.value || "Sample Headline For Stress Test";
    const btn = document.getElementById('btn-run-adv');
    if (btn) btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/adversarial/audit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: headline,
                text: headline,
                source: "Stress Lab"
            })
        });

        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('adv-ars-score').textContent = `${data.adversarial_robustness_score}%`;
        document.getElementById('adv-grade-badge').textContent = data.robustness_grade;
        document.getElementById('adv-evasion-stat').textContent = `${data.evasions_prevented}/${data.total_attacks_tested} Defended`;

        const tbody = document.getElementById('adv-attacks-tbody');
        tbody.innerHTML = '';
        data.attack_breakdown.forEach(atk => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: 500; color: var(--text-primary);">${atk.attack_type}</td>
                <td style="font-family: var(--font-mono); font-size: 0.6875rem; color: var(--text-secondary); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${atk.perturbed_sample_headline}</td>
                <td style="font-weight: 600; color: ${atk.attack_prediction === 'FAKE' ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">${atk.attack_prediction}</td>
                <td style="font-family: var(--font-mono); color: var(--text-muted);">Δ ${atk.confidence_shift_delta}%</td>
                <td>
                    <span class="brand-badge" style="color: ${atk.evasion_successful ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">
                        ${atk.evasion_successful ? 'Evasion' : 'Defended'}
                    </span>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Adversarial audit error", e);
    } finally {
        if (btn) btn.disabled = false;
    }
}

// Feedback Modal Handlers
function openFeedbackModal() {
    if (!currentPredictionData) {
        alert("Please analyze an article before submitting feedback.");
        return;
    }
    document.getElementById('feedback-title').value = currentPredictionData.title || "";
    document.getElementById('feedback-model-pred').value = `${currentPredictionData.prediction} (${currentPredictionData.prediction === 'FAKE' ? currentPredictionData.confidence_fake : currentPredictionData.confidence_real}%)`;
    document.getElementById('feedback-modal').style.display = 'flex';
}

function closeFeedbackModal() {
    document.getElementById('feedback-modal').style.display = 'none';
}

async function handleFeedbackSubmit(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-submit-feedback');
    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `Saving...`;

    const payload = {
        article_id: String(currentPredictionData.id || "1"),
        article_title: document.getElementById('feedback-title').value,
        model_prediction: currentPredictionData.prediction,
        corrected_label: document.getElementById('feedback-correct-label').value,
        notes: document.getElementById('feedback-notes').value,
        analyst_name: document.getElementById('feedback-analyst').value
    };

    try {
        const res = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Failed to submit feedback.");
        alert("Annotation recorded for active learning dataset.");
        closeFeedbackModal();
    } catch (e) {
        alert(`Feedback Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = origHtml;
    }
}

// Live Threat Stream Feed
async function fetchThreatStream(forceRefresh = false) {
    const container = document.getElementById('threat-feed-container');
    if (!container) return;

    if (forceRefresh) {
        container.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--text-muted); font-size: 0.8125rem;">Updating public news feed...</div>`;
    }

    try {
        const res = await fetch(`${API_BASE}/api/threat-stream`);
        if (!res.ok) return;
        const data = await res.json();

        const mtiEl = document.getElementById('mti-score');
        const badgeEl = document.getElementById('threat-status-badge');
        const updatedEl = document.getElementById('mti-updated');
        const statMon = document.getElementById('threat-stat-monitored');
        const statFake = document.getElementById('threat-stat-fake');
        const statReal = document.getElementById('threat-stat-real');

        if (mtiEl) mtiEl.textContent = `${data.threat_index} / 100`;
        if (statMon) statMon.textContent = data.total_monitored;
        if (statFake) statFake.textContent = data.flagged_threats;
        if (statReal) statReal.textContent = data.verified_authentic;
        if (updatedEl) updatedEl.textContent = `Last sync: ${data.last_updated}`;

        if (badgeEl) {
            badgeEl.textContent = `${data.threat_status} Status`;
            badgeEl.style.color = data.threat_status === 'HIGH' ? 'var(--status-fake-text)' : 'var(--status-real-text)';
        }

        container.innerHTML = '';
        data.feed_items.forEach(item => {
            const isFake = item.prediction === 'FAKE';
            const card = document.createElement('div');
            card.style.cssText = `padding: 0.875rem 1rem; border-radius: var(--radius-md); border: 1px solid var(--border); background: var(--bg-surface); display: flex; flex-direction: column; gap: 0.5rem; justify-content: space-between;`;
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                            <span class="brand-badge" style="color: ${isFake ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">
                                ${isFake ? 'Flagged Misinformation' : 'Authentic'}
                            </span>
                            <span style="font-size: 0.6875rem; color: var(--text-dim);">${item.source} · ${item.timestamp}</span>
                        </div>
                        <h4 style="font-size: 0.875rem; font-weight: 600; color: var(--text-primary); margin-top: 0.125rem;">${item.title}</h4>
                        <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem; line-height: 1.45;">${item.summary}</p>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="text-align: right; font-family: var(--font-mono);">
                            <div style="font-size: 0.875rem; font-weight: 600; color: ${isFake ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">
                                ${isFake ? item.confidence_fake : item.confidence_real}%
                            </div>
                            <div style="font-size: 0.625rem; color: var(--text-dim);">${isFake ? 'Risk' : 'Confidence'}</div>
                        </div>
                        <button onclick="inspectThreatItem('${encodeURIComponent(item.title)}', '${encodeURIComponent(item.summary)}', '${encodeURIComponent(item.source)}')" class="btn btn-secondary btn-sm">
                            Inspect
                        </button>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.error("Failed to fetch threat stream", e);
    }
}

function inspectThreatItem(encodedTitle, encodedSummary, encodedSource) {
    const title = decodeURIComponent(encodedTitle);
    const summary = decodeURIComponent(encodedSummary);
    const source = decodeURIComponent(encodedSource);

    switchTab('detector');
    document.getElementById('input-title').value = title;
    document.getElementById('input-text').value = summary || title;
    document.getElementById('input-source').value = source || "Wire News";
    document.getElementById('predict-form').dispatchEvent(new Event('submit'));
}

// Active Learning & Concept Drift Stats
async function fetchActiveLearningStats() {
    try {
        const res = await fetch(`${API_BASE}/api/active-learning/stats`);
        if (!res.ok) return;
        const data = await res.json();

        const psiScoreEl = document.getElementById('drift-psi-score');
        const psiStatusEl = document.getElementById('drift-psi-status');
        const feedbackCountEl = document.getElementById('drift-feedback-count');
        const retrainPctEl = document.getElementById('drift-retrain-pct');
        const retrainStatusEl = document.getElementById('drift-retrain-status');
        const tableBody = document.getElementById('feedback-table-body');

        if (psiScoreEl) psiScoreEl.textContent = data.concept_drift.population_stability_index;
        if (psiStatusEl) {
            psiStatusEl.textContent = `${data.concept_drift.drift_status} (PSI: ${data.concept_drift.population_stability_index})`;
        }
        if (feedbackCountEl) feedbackCountEl.textContent = data.total_human_feedback_samples;
        if (retrainPctEl) retrainPctEl.textContent = `${data.active_learning_pipeline.retrain_readiness_pct}%`;
        if (retrainStatusEl) retrainStatusEl.textContent = `Status: ${data.active_learning_pipeline.status} (${data.total_human_feedback_samples}/${data.active_learning_pipeline.feedback_threshold})`;

        if (tableBody) {
            tableBody.innerHTML = '';
            if (!data.recent_corrections || data.recent_corrections.length === 0) {
                tableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 2rem; color: var(--text-dim);">No human feedback entries logged yet.</td></tr>`;
                return;
            }

            data.recent_corrections.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: 500; color: var(--text-primary); max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${row.title}</td>
                    <td>${row.model_pred}</td>
                    <td style="font-weight: 600; color: ${row.corrected === 'FAKE' ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">${row.corrected}</td>
                    <td style="color: var(--text-muted); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${row.notes || '—'}</td>
                    <td style="color: var(--text-dim);">${row.timestamp}</td>
                `;
                tableBody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Failed to load active learning stats", e);
    }
}

// Metrics & Benchmarks
async function fetchMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        if (!res.ok) return;
        const metrics = await res.json();
        
        renderMetricsTable(metrics);
        renderCharts(metrics);
    } catch (e) {
        console.error("Failed to load metrics", e);
    }
}

function renderMetricsTable(metrics) {
    const tbody = document.getElementById('metrics-table-body');
    if (!tbody || !metrics.model_comparison) return;
    tbody.innerHTML = '';

    metrics.model_comparison.forEach(m => {
        const isProposed = m.is_proposed || m.model.includes('Proposed') || m.model.includes('BERT Contextual');
        const tr = document.createElement('tr');
        if (isProposed) {
            tr.style.background = 'var(--bg-surface-raised)';
        }
        tr.innerHTML = `
            <td style="font-weight: ${isProposed ? '600' : '500'}; color: ${isProposed ? 'var(--text-primary)' : 'var(--text-secondary)'};">
                ${m.model} ${isProposed ? '<span class="brand-badge" style="margin-left: 0.35rem;">Proposed</span>' : ''}
            </td>
            <td style="font-family: var(--font-mono); font-weight: ${isProposed ? '600' : '400'}; color: ${isProposed ? 'var(--text-primary)' : 'var(--text-secondary)'};">${(m.accuracy * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono);">${(m.precision * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono);">${(m.recall * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono);">${(m.f1 * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono);">${(m.roc_auc * 100).toFixed(1)}%</td>
            <td><span class="brand-badge">${m.category || 'Baseline'}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function renderCharts(metrics) {
    if (!metrics.model_comparison) return;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    // Accuracy Comparison Bar Chart
    const ctx1 = document.getElementById('chart-models-comparison');
    if (ctx1) {
        if (modelsChart) modelsChart.destroy();
        const labels = metrics.model_comparison.map(m => m.model.replace(' Classifier', '').replace(' Network', ''));
        const accs = metrics.model_comparison.map(m => (m.accuracy * 100).toFixed(1));
        const f1s = metrics.model_comparison.map(m => (m.f1 * 100).toFixed(1));

        const barColor1 = isDark ? '#f8fafc' : '#0f172a';
        const barColor2 = isDark ? '#64748b' : '#94a3b8';

        modelsChart = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: accs,
                        backgroundColor: barColor1,
                        borderRadius: 3
                    },
                    {
                        label: 'F1-Score (%)',
                        data: f1s,
                        backgroundColor: barColor2,
                        borderRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        labels: { 
                            color: isDark ? '#cbd5e1' : '#334155', 
                            font: { family: 'Inter', size: 11, weight: '500' } 
                        } 
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: isDark ? '#8b949e' : '#64748b', font: { family: 'Inter', size: 9 } }, 
                        grid: { display: false } 
                    },
                    y: { 
                        min: 70, 
                        max: 100, 
                        ticks: { color: isDark ? '#8b949e' : '#64748b', font: { family: 'Inter', size: 10 } }, 
                        grid: { color: isDark ? '#1a1b20' : '#f1f5f9' } 
                    }
                }
            }
        });
    }

    // Global SHAP Feature Importance Chart
    fetchGlobalSHAP();
}

async function fetchGlobalSHAP() {
    try {
        const res = await fetch(`${API_BASE}/api/shap/global`);
        if (!res.ok) return;
        const shapData = await res.json();
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        
        const ctx2 = document.getElementById('chart-feature-importance');
        if (!ctx2) return;
        if (importanceChart) importanceChart.destroy();

        const top8 = shapData.features.slice(0, 8);
        const labels = top8.map(f => f.feature);
        const values = top8.map(f => f.importance_percent);

        importanceChart = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Global Influence (%)',
                    data: values,
                    backgroundColor: isDark ? '#3b82f6' : '#0f172a',
                    borderRadius: 3
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { 
                        ticks: { color: isDark ? '#8b949e' : '#64748b', font: { family: 'Inter', size: 10 } }, 
                        grid: { color: isDark ? '#1a1b20' : '#f1f5f9' } 
                    },
                    y: { 
                        ticks: { color: isDark ? '#cbd5e1' : '#334155', font: { family: 'Inter', size: 10 } }, 
                        grid: { display: false } 
                    }
                }
            }
        });
    } catch (e) {
        console.error("Failed to load global SHAP chart", e);
    }
}

// Prediction Audit History
async function fetchHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        if (!res.ok) return;
        const history = await res.json();
        
        const tableBody = document.getElementById('history-table-body');
        if (!tableBody) return;
        tableBody.innerHTML = '';

        if (history.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 2rem; color: var(--text-dim);">No prediction history recorded yet.</td></tr>`;
            return;
        }

        history.forEach((row) => {
            const isFake = row.prediction === 'FAKE';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-family: var(--font-mono); font-size: 0.6875rem; color: var(--text-muted);">${row.firebase_id || 'rec_' + row.id}</td>
                <td style="font-weight: 500; color: var(--text-primary); max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${row.title}</td>
                <td style="color: var(--text-secondary);">${row.source || 'Web'}</td>
                <td style="color: var(--text-dim); font-size: 0.6875rem;">${row.model_architecture || 'BERT Multimodal'}</td>
                <td>
                    <span class="brand-badge" style="color: ${isFake ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">
                        ${row.prediction}
                    </span>
                </td>
                <td style="font-family: var(--font-mono); font-weight: 600; color: ${isFake ? 'var(--status-fake-text)' : 'var(--status-real-text)'};">
                    ${isFake ? row.confidence_fake : row.confidence_real}%
                </td>
                <td style="color: var(--text-dim);">${row.timestamp}</td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load history", e);
    }
}

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    fetchMetrics();
});
