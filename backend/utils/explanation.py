def generate_explanations(features: dict, probability: float) -> list:
    reasons = []
    # Simple heuristics for explanations. Replace with your explanation engine.
    if features.get("has_price") and features.get("price"):
        try:
            price = float(features.get("price"))
            if price > 0 and price < 10:
                reasons.append("Extremely low price detected")
        except Exception:
            pass
    if features.get("has_phone"):
        reasons.append("Contact information found in the description")
    if features.get("word_count", 0) < 10:
        reasons.append("Insufficient description length")

    # Model-driven reason
    if probability >= 0.8:
        reasons.append("High model confidence for suspicious label")
    elif probability >= 0.5:
        reasons.append("Moderate model confidence for suspicious label")

    if not reasons:
        reasons.append("No obvious heuristic reasons found; model scored this ad low risk.")

    return reasons
