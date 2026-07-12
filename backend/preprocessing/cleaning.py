import re


def clean_text(text: str) -> str:
    if text is None:
        return ""
    # simple cleaning: normalize whitespace, remove non-printable
    text = text.replace("\r\n", " \n ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
