# core – shared modules used by Dashboard, Evaluation and Training

import re


_RE_COLON_DIGITS = re.compile(r"(:\d+)+$")
_RE_SPACE_HASH = re.compile(r"\s+\d{5,}$")
_RE_DASH_SEQ = re.compile(r"-\d+$")
_RE_DASH_INFIX_NUM = re.compile(r"-\d+(?=-[A-Za-z])")


def strip_numeric_ids(value: str) -> str:
    """Strip trailing instance/hash-like numeric suffixes from IFC text values."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = _RE_COLON_DIGITS.sub("", text).strip()
    text = _RE_SPACE_HASH.sub("", text).strip()
    text = _RE_DASH_INFIX_NUM.sub("", text).strip()
    text = _RE_DASH_SEQ.sub("", text).strip()
    return text
