/**
 * VeriShield AI - Background Service Worker (Manifest V3)
 * Handles context menu registration and background inference requests.
 */

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verishield_verify_selection",
    title: "🛡️ Verify with VeriShield AI",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "verishield_verify_selection" && info.selectionText) {
    const selectedText = info.selectionText.trim();
    if (!selectedText) return;

    try {
      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: selectedText.substring(0, 80),
          text: selectedText,
          source: "Web Selection",
          author: "Unknown",
          shares: 1000.0,
          sentiment: 0.0
        })
      });

      if (!response.ok) throw new Error("API request failed.");
      const result = await response.json();

      // Send result to content script for in-page notification
      if (tab && tab.id) {
        chrome.tabs.sendMessage(tab.id, {
          action: "VERISHIELD_RESULT",
          data: result
        });
      }
    } catch (err) {
      if (tab && tab.id) {
        chrome.tabs.sendMessage(tab.id, {
          action: "VERISHIELD_ERROR",
          message: "Could not connect to VeriShield AI backend (http://localhost:8000)."
        });
      }
    }
  }
});
