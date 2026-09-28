import os
import sys
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np

extractor = joblib.load("models/feature_extractor.joblib")
hybrid_model = joblib.load("models/hybrid_model.joblib")

test_cases = [
    ("Alien invasion in New York", "Aliens have invaded New York City and government is hiding it inside bunkers.", "FAKE"),
    ("Miracle Cancer Cure", "Drinking hot lemon water mixed with baking soda cures 100% of all cancers overnight doctors hate this secret.", "FAKE"),
    ("Free iPhone Giveaway", "Forward this WhatsApp message to 10 friends to claim your free iPhone 15 Pro Max and 50000 rupees immediately.", "FAKE"),
    ("Secret 5G Mind Control", "Leaked documents prove 5G towers emit radiation frequencies designed by global elites to mind control the population.", "FAKE"),
    ("Celebrity Secretly Cloned", "Famous actor was cloned in secret laboratory and the person on TV is actually an android robot.", "FAKE"),
    ("Bank Giving Free Money", "Government announced all citizens will get free 500000 deposited in bank account tomorrow without application.", "FAKE"),
    ("Breaking: World Leader Assassinated", "Unconfirmed leaked video shows president was poisoned by deep state agents yesterday.", "FAKE"),
    ("University Exam Schedule", "The university administration published the final exam timetable on the official student portal.", "REAL"),
    ("Heavy Rain Warning", "Meteorological department issued yellow alert for coastal districts predicting heavy thunderstorms.", "REAL"),
    ("Economic Growth Report", "Central statistical organization released quarterly GDP data showing 6.8 percent annualized growth.", "REAL")
]

print("=== PREDICTIONS WITH HYBRID_MODEL (src.features.HybridFeatureExtractor) ===")
print(f"{'Title':<35} | {'Expected':<8} | {'Pred':<8} | {'FakeProb':<10} | {'Status':<6}")
print("-" * 80)
for title, text, exp in test_cases:
    vec = extractor.transform([title], [text], [{"source": "Social Media"}])
    probs = hybrid_model.predict_proba(vec)[0]
    fake_p = probs[1]
    pred = "FAKE" if fake_p >= 0.50 else "REAL"
    status = "PASS" if pred == exp else "FAIL"
    print(f"{title[:33]:<35} | {exp:<8} | {pred:<8} | {fake_p*100:<9.1f}% | {status:<6}")
