import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib

model = joblib.load('models/hybrid_model.joblib')
ext = joblib.load('models/feature_extractor.joblib')

user_test_cases = [
    ("NASA has confirmed that the Moon will disappear from Earth's sky for 24 hours next week.", "FAKE"),
    ("Scientists have discovered a pill that allows humans to live without sleeping.", "FAKE"),
    ("The government announced that every citizen will receive ₹1 lakh for forwarding a message on WhatsApp.", "FAKE"),
    ("Researchers developed a smartphone battery that can run continuously for 10 years without charging.", "FAKE"),
    ("Drinking water after midnight has been officially proven to double human intelligence.", "FAKE"),
    ("The university announced the timetable for the upcoming semester examinations through its official website.", "REAL"),
    ("The weather department issued an advisory asking residents to take precautions during heavy rainfall.", "REAL"),
    ("Researchers published a study evaluating a new machine-learning method for improving classification performance.", "REAL"),
    ("The city administration announced temporary traffic restrictions because of scheduled road maintenance.", "REAL"),
    ("The technology company released a software update containing several security and performance improvements.", "REAL")
]

header = f"{'#':<3} | {'Title':<65} | {'Expected':<8} | {'Fake Prob':<10} | {'Verdict':<8} | {'Status':<6}"
print(header)
print("-" * len(header))

all_passed = True
for i, (title, expected) in enumerate(user_test_cases, 1):
    feat, tokens, attn = ext.encode_single_with_attention(title, "")
    prob = model.predict_proba(feat)[0]
    verdict = "FAKE" if prob[1] >= 0.5 else "REAL"
    status = "PASS" if verdict == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    clean_title = title.replace("₹", "Rs. ")
    print(f"{i:2d}  | {clean_title[:63]:<65} | {expected:<8} | {prob[1]*100:6.1f}%   | {verdict:<8} | {status:<6}")

print("-" * len(header))
if all_passed:
    print("ALL 10 BENCHMARK TESTS PASSED (10/10)!")
else:
    print("SOME TESTS FAILED - REVIEW RESULTS.")
