import re
import os
import numpy as np
import pandas as pd

PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{6,}\d)")
PRICE_PATTERNS = [
    re.compile(r"(\d+)\s?k", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*million", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*mil", re.IGNORECASE),
    re.compile(r"(\d{4,9})")
]

SUSPICIOUS_WORDS = set([
    "uhakika",
    "urgent",
    "haraka",
    "cheap",
    "offer",
    "deal",
    "promo",
    "limited",
    "guarantee",
    "100%",
    "call now",
    "piga sasa",
    "special",
])

CATEGORY_KEYWORDS = {
    # truncated mapping copied/adapted from your notebook for runtime checks
    "vehicles":["toyota","nissan","honda","mazda","subaru","car","motorcycle","truck","lorry","prado","landcruiser","hilux"],
    "smartphones":["iphone","samsung","galaxy","tecno","infinix","xiaomi","phone","mobile"],
    "computers":["laptop","computer","desktop","printer","monitor","keyboard","mouse","hp","dell","lenovo","macbook"],
    "real-estate":["house","home","plot","land","nyumba","apartment","room","villa"],
}


def mask_pii(text: str) -> str:
    if text is None:
        return text
    text = str(text)
    # phone numbers (simple international and local patterns)
    text = re.sub(r"\b0[67]\d{8,9}\b", "<PHONE>", text)
    text = re.sub(r"\+\d{7,15}", "<PHONE>", text)
    # emails
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    return text


def extract_price(text: str) -> float:
    if not text:
        return np.nan
    t = str(text).lower()
    for p in PRICE_PATTERNS:
        m = p.search(t)
        if m:
            try:
                value = float(m.group(1))
            except Exception:
                continue
            s = m.group(0)
            if "k" in s:
                value *= 1000
            if "mil" in s or "million" in s:
                value *= 1000000
            if 1 <= value <= 5e8:
                return value
    return np.nan


def suspicious_word_count(text: str) -> int:
    if not text:
        return 0
    t = str(text).lower()
    return sum(1 for w in SUSPICIOUS_WORDS if w in t)


def category_match_strength(category: str, text: str) -> int:
    category = (category or "").lower()
    text = (text or "").lower()
    if category not in CATEGORY_KEYWORDS:
        return 1
    matches = sum(k in text for k in CATEGORY_KEYWORDS[category])
    if matches >= 3:
        return 2
    if matches >= 1:
        return 1
    return 0


def detect_category_conflict(category: str, text: str) -> int:
    # Lightweight conflict detection: if text contains common terms from other categories
    category = (category or "").lower()
    text = (text or "").lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if cat == category:
            continue
        if any(k in text for k in keywords):
            return 1
    return 0


def build_features(title: str = "", description: str = "", category: str = None, image_count: int = None, contact_type: str = None):
    """Build a feature dict for a single ad using the logic in your notebooks.
    Returns a dict with derived features and the combined text under 'text_combined'.
    """
    title = (title or "").strip()
    description = (description or "").strip()
    # mask PII first
    title_masked = mask_pii(title)
    desc_masked = mask_pii(description)

    text_combined = (title_masked + " " + desc_masked).strip().lower()

    title_length = len(title_masked)
    description_length = len(desc_masked)
    title_word_count = len(title_masked.split())
    description_word_count = len(desc_masked.split())
    missing_description_flag = int(description_length == 0)

    # contact related
    has_phone_in_desc = int(bool(PHONE_RE.search(desc_masked)))
    contact_available = 0
    if contact_type is not None:
        ct = str(contact_type).lower()
        contact_available = 0 if ct in ("", "none", "nan") else 1

    contact_leakage = int(bool(re.search(r"whatsapp|whatsap|call|contact|piga|simu", desc_masked, flags=re.IGNORECASE)))

    # price extraction
    price_from_description = extract_price(desc_masked)

    # price_final: we don't have price_clean at inference time unless user provides; use price_from_description
    price_final = price_from_description if not np.isnan(price_from_description) else np.nan
    price_missing_flag = int(np.isnan(price_final))
    log_price = float(np.log1p(price_final)) if not np.isnan(price_final) else 0.0

    # category features (we can't compute category median from single row; set defaults)
    category_match_score = category_match_strength(category or "", text_combined)
    category_mismatch_flag = detect_category_conflict(category or "", text_combined)
    category_reliability = category_match_score - category_mismatch_flag
    category_frequency = 0  # unknown at prediction time unless you provide a lookup table
    price_zscore_category = 0.0

    # condition / delivery / negotiable heuristics
    condition_available = int(bool(re.search(r"new|used|brand new|refurbished|locally", desc_masked, flags=re.IGNORECASE)))
    delivery_flag = int(bool(re.search(r"delivery|shipping|tunatuma|mikoani|hadi nyumbani|tunakufuata", desc_masked, flags=re.IGNORECASE)))
    negotiable_flag = int(bool(re.search(r"negotiable|bei mwisho|bargain|last price|leta ofa yako|bei maelewano", desc_masked, flags=re.IGNORECASE)))
    has_price_in_desc = int(bool(re.search(r"\d+", desc_masked)))

    suspicious_wc = suspicious_word_count(text_combined)

    fraud_indicator_count = (
        price_missing_flag
        + missing_description_flag
        + category_mismatch_flag
        + contact_leakage
        + int(suspicious_wc > 0)
    )

    features = {
        "title": title_masked,
        "description": desc_masked,
        "text_combined": text_combined,
        "title_length": title_length,
        "description_length": description_length,
        "title_word_count": title_word_count,
        "description_word_count": description_word_count,
        "missing_description_flag": missing_description_flag,
        "has_phone_in_desc": has_phone_in_desc,
        "contact_available": contact_available,
        "contact_leakage": contact_leakage,
        "price_from_description": float(price_from_description) if not np.isnan(price_from_description) else None,
        "price_final": float(price_final) if not np.isnan(price_final) else None,
        "price_missing_flag": price_missing_flag,
        "log_price": log_price,
        "category_match_score": category_match_score,
        "category_mismatch_flag": category_mismatch_flag,
        "category_reliability": category_reliability,
        "category_frequency": category_frequency,
        "price_zscore_category": price_zscore_category,
        "condition_available": condition_available,
        "delivery_flag": delivery_flag,
        "negotiable_flag": negotiable_flag,
        "has_price_in_desc": has_price_in_desc,
        "suspicious_word_count": suspicious_wc,
        "fraud_indicator_count": fraud_indicator_count,
        "image_count": int(image_count) if image_count is not None else 0,
        "category": category or "",
    }

    return features
