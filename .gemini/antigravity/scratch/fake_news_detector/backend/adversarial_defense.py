"""
Phase 15: Adversarial Perturbation Defense & Robustness Evaluation Engine
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Simulates sophisticated evasion attacks against fake news classifiers:
1. Cyrillic/Greek Homoglyph Substitution
2. Zero-Width Character / Unicode Obfuscation
3. Adversarial Synonym Replacement
4. Emoji Flood & Symbol Injection

Provides defensive Unicode canonicalization & Adversarial Robustness Scoring (ARS).
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Any

# Common Cyrillic and Greek homoglyphs used to evade NLP filters
HOMOGLYPH_MAP = {
    'a': 'а', 'c': 'с', 'e': 'е', 'i': 'і', 'j': 'ј',
    'o': 'о', 'p': 'р', 's': 'ѕ', 'x': 'х', 'y': 'у',
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'I': 'І', 'J': 'Ј', 'K': 'К', 'M': 'М', 'O': 'О',
    'P': 'Р', 'S': 'Ѕ', 'T': 'Т', 'X': 'Х', 'Y': 'Ү'
}

REVERSE_HOMOGLYPH_MAP = {v: k for k, v in HOMOGLYPH_MAP.items()}


def inject_homoglyphs(text: str, substitution_rate: float = 0.25) -> str:
    """Substitutes ASCII characters with visually indistinguishable Cyrillic/Greek homoglyphs."""
    result = []
    for char in text:
        if char in HOMOGLYPH_MAP and (hash(char) % 100) < (substitution_rate * 100):
            result.append(HOMOGLYPH_MAP[char])
        else:
            result.append(char)
    return "".join(result)


def inject_zero_width_spaces(text: str, frequency: int = 3) -> str:
    """Injects zero-width space characters (\u200B, \u200C) between words and letters."""
    words = text.split()
    obfuscated_words = []
    for idx, word in enumerate(words):
        if idx % frequency == 0 and len(word) > 2:
            mid = len(word) // 2
            obfuscated_words.append(word[:mid] + "\u200b" + word[mid:])
        else:
            obfuscated_words.append(word)
    return " ".join(obfuscated_words)


def inject_emoji_noise(text: str) -> str:
    """Injects high-frequency emojis to bypass text-only density heuristic thresholds."""
    emojis = ["🚨", "⚠️", "🔥", "💥", "😱", "🛑", "⚡", "📢"]
    words = text.split()
    res = []
    for idx, word in enumerate(words):
        res.append(word)
        if idx % 5 == 0:
            res.append(emojis[idx % len(emojis)])
    return " ".join(res)


def inject_synonym_obfuscation(text: str) -> str:
    """Replaces common trigger words with unusual synonyms to test semantic embedding stability."""
    synonym_dict = {
        "cancer": "neoplastic malignancy",
        "cure": "biochemical resolution",
        "secret": "undisclosed confidential memorandum",
        "government": "central administrative regime",
        "doctors": "medical practitioners",
        "banned": "censored from dissemination",
        "fake": "unverified synthetically generated"
    }
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in synonym_dict.keys()) + r')\b', re.IGNORECASE)
    return pattern.sub(lambda m: synonym_dict.get(m.group(0).lower(), m.group(0)), text)


def sanitize_text(text: str) -> str:
    """
    Defensive Preprocessing: Strips adversarial unicode artifacts and canonicalizes homoglyphs.
    """
    if not text:
        return ""
    
    # 1. Normalize Unicode NFKD
    normalized = unicodedata.normalize('NFKD', text)
    
    # 2. Reverse homoglyphs
    cleaned_chars = []
    for char in normalized:
        if char in REVERSE_HOMOGLYPH_MAP:
            cleaned_chars.append(REVERSE_HOMOGLYPH_MAP[char])
        elif char in ['\u200b', '\u200c', '\u200d', '\ufeff']:
            continue  # Strip zero-width characters
        else:
            cleaned_chars.append(char)
            
    cleaned_text = "".join(cleaned_chars)
    # 3. Collapse multiple whitespace
    return re.sub(r'\s+', ' ', cleaned_text).strip()


def evaluate_adversarial_robustness(title: str, text: str, source: str, predict_fn) -> Dict[str, Any]:
    """
    Executes a multi-vector adversarial stress test against the 79-dimensional model:
    1. Baseline clean text
    2. Homoglyph perturbation
    3. Zero-width unicode perturbation
    4. Emoji noise perturbation
    5. Synonym obfuscation
    6. Sanitized defense restoration
    """
    # 1. Baseline Prediction
    base_res = predict_fn(title, text, source)
    base_pred = base_res.get("prediction", "REAL")
    base_fake_conf = float(base_res.get("confidence_fake", 50.0))

    # Generate Attacks
    attacks = [
        {
            "attack_type": "Homoglyph Substitution",
            "perturbed_title": inject_homoglyphs(title, 0.4),
            "perturbed_text": inject_homoglyphs(text, 0.3),
            "description": "Cyrillic/Greek Unicode visual spoofing"
        },
        {
            "attack_type": "Zero-Width Obfuscation",
            "perturbed_title": inject_zero_width_spaces(title, 2),
            "perturbed_text": inject_zero_width_spaces(text, 3),
            "description": "Invisible zero-width space boundary disruption"
        },
        {
            "attack_type": "Emoji Noise Injection",
            "perturbed_title": inject_emoji_noise(title),
            "perturbed_text": inject_emoji_noise(text),
            "description": "Heuristic dilution via synthetic emoji flooding"
        },
        {
            "attack_type": "Synonym Obfuscation",
            "perturbed_title": inject_synonym_obfuscation(title),
            "perturbed_text": inject_synonym_obfuscation(text),
            "description": "Keyword evasion via high-order semantic synonym replacement"
        }
    ]

    attack_results = []
    consistent_count = 0
    max_conf_delta = 0.0

    for atk in attacks:
        res = predict_fn(atk["perturbed_title"], atk["perturbed_text"], source)
        pred = res.get("prediction", "REAL")
        fake_conf = float(res.get("confidence_fake", 50.0))
        delta = abs(fake_conf - base_fake_conf)
        
        is_consistent = (pred == base_pred)
        if is_consistent:
            consistent_count += 1
        if delta > max_conf_delta:
            max_conf_delta = delta

        attack_results.append({
            "attack_type": atk["attack_type"],
            "description": atk["description"],
            "perturbed_sample_headline": atk["perturbed_title"][:65] + "...",
            "attack_prediction": pred,
            "attack_confidence_fake": fake_conf,
            "confidence_shift_delta": round(delta, 1),
            "evasion_successful": not is_consistent
        })

    # Test Defensive Sanitizer
    sanitized_title = sanitize_text(attacks[0]["perturbed_title"])
    sanitized_text = sanitize_text(attacks[0]["perturbed_text"])
    sanitized_res = predict_fn(sanitized_title, sanitized_text, source)
    sanitization_restored = (sanitized_res.get("prediction", "REAL") == base_pred)

    # Adversarial Robustness Score (ARS: 0 - 100%)
    # Evaluates end-to-end defended pipeline resilience (Consistency + Sanitization Recovery)
    effective_defended_count = consistent_count + (1 if sanitization_restored and consistent_count < len(attacks) else 0)
    consistency_score = (effective_defended_count / len(attacks)) * 75.0
    stability_score = max(0.0, 25.0 - (max_conf_delta * 0.18))
    ars_score = round(consistency_score + stability_score, 1)

    robustness_grade = "TIER 1 (HIGHLY ROBUST)" if ars_score >= 80 else ("TIER 2 (MODERATE)" if ars_score >= 65 else "VULNERABLE")

    return {
        "baseline_prediction": base_pred,
        "baseline_confidence_fake": base_fake_conf,
        "adversarial_robustness_score": ars_score,
        "robustness_grade": robustness_grade,
        "total_attacks_tested": len(attacks),
        "evasions_prevented": consistent_count,
        "max_confidence_deviation": round(max_conf_delta, 1),
        "attack_breakdown": attack_results,
        "sanitizer_verification": {
            "sanitized_headline": sanitized_title[:65] + "...",
            "sanitized_prediction": sanitized_res.get("prediction", "REAL"),
            "sanitization_restored_baseline": sanitization_restored
        }
    }
