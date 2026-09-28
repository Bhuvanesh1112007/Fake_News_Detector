/**
 * VeriShield AI - Content Script (Manifest V3)
 * Renders an unobtrusive floating glassmorphism verdict badge on the active page.
 */

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "VERISHIELD_RESULT") {
    showVerdictBadge(message.data);
  } else if (message.action === "VERISHIELD_ERROR") {
    showErrorBadge(message.message);
  }
});

function showVerdictBadge(data) {
  const existing = document.getElementById("verishield-floating-badge");
  if (existing) existing.remove();

  const isFake = data.prediction === "FAKE";
  const badge = document.createElement("div");
  badge.id = "verishield-floating-badge";

  badge.style.position = "fixed";
  badge.style.bottom = "20px";
  badge.style.right = "20px";
  badge.style.zIndex = "9999999";
  badge.style.background = "#0b0f19";
  badge.style.color = "#ffffff";
  badge.style.border = `2px solid ${isFake ? "#f43f5e" : "#10b981"}`;
  badge.style.borderRadius = "10px";
  badge.style.padding = "14px 18px";
  badge.style.boxShadow = "0 10px 25px rgba(0,0,0,0.5)";
  badge.style.fontFamily = "sans-serif";
  badge.style.maxWidth = "320px";
  badge.style.fontSize = "13px";
  badge.style.lineHeight = "1.5";

  badge.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
      <strong style="color:${isFake ? '#f43f5e' : '#10b981'}; font-size:14px;">
        ${isFake ? '⚠️ FAKE NEWS DETECTED' : '✅ VERIFIED AUTHENTIC'}
      </strong>
      <button id="verishield-close-btn" style="background:none; border:none; color:#94a3b8; font-size:16px; cursor:pointer;">&times;</button>
    </div>
    <div style="font-size:12px; color:#cbd5e1; margin-bottom:6px;">
      Confidence: <strong>${isFake ? data.confidence_fake : data.confidence_real}%</strong> (${data.risk_level})
    </div>
    <div style="font-size:11px; color:#94a3b8;">
      ${data.key_reasons && data.key_reasons[0] ? data.key_reasons[0] : 'Verified via VeriShield AI 79-Dim Engine.'}
    </div>
  `;

  document.body.appendChild(badge);

  document.getElementById("verishield-close-btn").addEventListener("click", () => {
    badge.remove();
  });

  // Auto remove after 10 seconds
  setTimeout(() => {
    if (document.getElementById("verishield-floating-badge")) {
      badge.remove();
    }
  }, 10000);
}

function showErrorBadge(msg) {
  const existing = document.getElementById("verishield-floating-badge");
  if (existing) existing.remove();

  const badge = document.createElement("div");
  badge.id = "verishield-floating-badge";
  badge.style.position = "fixed";
  badge.style.bottom = "20px";
  badge.style.right = "20px";
  badge.style.zIndex = "9999999";
  badge.style.background = "#1e293b";
  badge.style.color = "#f8fafc";
  badge.style.border = "1px solid #ef4444";
  badge.style.borderRadius = "8px";
  badge.style.padding = "10px 14px";
  badge.style.fontSize = "12px";
  badge.style.boxShadow = "0 8px 20px rgba(0,0,0,0.4)";

  badge.innerHTML = `🛡️ VeriShield AI: ${msg}`;
  document.body.appendChild(badge);

  setTimeout(() => badge.remove(), 5000);
}
