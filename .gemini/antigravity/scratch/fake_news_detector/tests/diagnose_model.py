import os
import sys
sys.path.insert(0, os.path.abspath("."))

import backend.app as app

app.load_ai_engine()

test_cases = [
    # 1. Obvious Conspiracy & Aliens
    ("Alien invasion in New York", "Aliens have invaded New York City and government is hiding it inside bunkers.", "FAKE"),
    # 2. Medical miracle hoax
    ("Miracle Cancer Cure", "Drinking hot lemon water mixed with baking soda cures 100% of all cancers overnight doctors hate this secret.", "FAKE"),
    # 3. Financial scam / WhatsApp viral
    ("Free iPhone Giveaway", "Forward this WhatsApp message to 10 friends to claim your free iPhone 15 Pro Max and 50000 rupees immediately.", "FAKE"),
    # 4. 5G conspiracy
    ("Secret 5G Mind Control", "Leaked documents prove 5G towers emit radiation frequencies designed by global elites to mind control the population.", "FAKE"),
    # 5. Celebrity clone hoax
    ("Celebrity Secretly Cloned", "Famous actor was cloned in secret laboratory and the person on TV is actually an android robot.", "FAKE"),
    # 6. Unbelievable government cash
    ("Bank Giving Free Money", "Government announced all citizens will get free 500000 deposited in bank account tomorrow without application.", "FAKE"),
    # 7. Disinformation / Fake death
    ("Breaking: World Leader Assassinated", "Unconfirmed leaked video shows president was poisoned by deep state agents yesterday.", "FAKE"),
    # 8. Real news - university
    ("University Exam Schedule", "The university administration published the final exam timetable on the official student portal.", "REAL"),
    # 9. Real news - weather
    ("Heavy Rain Warning", "Meteorological department issued yellow alert for coastal districts predicting heavy thunderstorms.", "REAL"),
    # 10. Real news - economy
    ("Economic Growth Report", "Central statistical organization released quarterly GDP data showing 6.8 percent annualized growth.", "REAL")
]

print(f"{'Title':<35} | {'Expected':<8} | {'Pred':<8} | {'FakeProb':<10} | {'Status':<6}")
print("-" * 80)
for title, text, exp in test_cases:
    req = app.PredictRequest(title=title, text=text, source="Social Media")
    res = app.predict_news(req)
    status = "PASS" if res.prediction == exp else "FAIL"
    print(f"{title[:33]:<35} | {exp:<8} | {res.prediction:<8} | {res.confidence_fake:<9}% | {status:<6}")
