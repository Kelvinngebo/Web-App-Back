import re

PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{6,}\d)")
PRICE_RE = re.compile(r"\b(\$|USD\s?)?([0-9]+(?:\.[0-9]{1,2})?)\b")


def parse_basic(text: str) -> dict:
    """Extract a few basic signals for debugging: phone present, price found, text lengths."""
    if not text:
        return {
            "has_phone": False,
            "phone": None,
            "has_price": False,
            "price": None,
            "text_length": 0,
            "word_count": 0,
        }

    phone_match = PHONE_RE.search(text)
    price_match = PRICE_RE.search(text)

    phone = phone_match.group(1) if phone_match else None
    price = float(price_match.group(2)) if price_match else None

    words = text.split()

    return {
        "has_phone": bool(phone),
        "phone": phone,
        "has_price": bool(price_match),
        "price": price,
        "text_length": len(text),
        "word_count": len(words),
    }
