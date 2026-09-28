"""
Extended Multi-Domain Generalization Test Suite.
Tests diverse real-world fake and real news across healthcare, politics, scams, astronomy, and science.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import backend.app as app

app.load_ai_engine()

extended_cases = [
    # Health Misinformation
    ("Drinking Kerosene Cures Diabetes Overnight", "Doctors in secret clinic revealed that drinking a tablespoon of kerosene eliminates diabetes in 48 hours without insulin.", "FAKE"),
    ("Raw Garlic and Vinegar Eliminates All Heart Blockages", "Banned cardiologists leak shocking secret recipe of garlic and vinegar that dissolves all arterial plaque instantly!", "FAKE"),
    # Scams & Crypto Giveaway
    ("Government Distributing Free Laptops to WhatsApp Users", "Every student who forwards this WhatsApp link to 5 groups will get a free high-end laptop delivered to their address!", "FAKE"),
    ("Secret Telegram Channel Leaks Free Bitcoin Codes", "Claim free 1.5 BTC by clicking this link and verifying your digital wallet before midnight!", "FAKE"),
    # Conspiracies & Fabricated Science
    ("Chemtrails Confirmed to Contain Mind-Altering Parasites", "Whistleblower pilots exposed that commercial airlines are secretly spraying subterranean mind-control chemicals on cities.", "FAKE"),
    ("Time Traveler from 2085 Predicts Worldwide Blackout Next Month", "Viral video shows man claiming to be from 2085 warning that global electrical grid will shut down permanently next Friday.", "FAKE"),
    # Real News - Healthcare & Clinical Trials
    ("Clinical Trial Demonstrates Efficacy of New Immunotherapy Drug", "Researchers published findings in The Lancet showing a 40 percent improvement in five-year survival rates for oncology patients.", "REAL"),
    # Real News - Astronomy & Science
    ("Astronomers Detect Gravitational Waves from Binary Neutron Star Merger", "The international astrophysics collaboration announced the observation of gravitational wave signals confirmed by optical observatories.", "REAL"),
    # Real News - Public Governance & Infrastructure
    ("Ministry of Finance Extends Income Tax Filing Deadline by Two Weeks", "The department of revenue notified taxpayers that the annual filing deadline has been extended to ensure smooth portal operations.", "REAL"),
    ("Public Works Department Inaugurates New Flyover to Ease Urban Traffic", "City municipal authorities formally opened the four-lane elevated corridor connecting the tech park with the highway.", "REAL")
]

print(f"{'Title':<45} | {'Expected':<8} | {'Pred':<8} | {'FakeProb':<10} | {'Status':<6}")
print("-" * 90)
passed_count = 0
for title, text, exp in extended_cases:
    req = app.PredictRequest(title=title, text=text, source="Web")
    res = app.predict_news(req)
    status = "PASS" if res.prediction == exp else "FAIL"
    if status == "PASS":
        passed_count += 1
    print(f"{title[:43]:<45} | {exp:<8} | {res.prediction:<8} | {res.confidence_fake:<9}% | {status:<6}")

print("-" * 90)
print(f"Extended Accuracy: {passed_count}/{len(extended_cases)} ({passed_count/len(extended_cases)*100:.1f}%)")
