"""Text processing utilities."""

import math
import re
from .constants import WORD_RE, TR_CHARS_RE, TR_STOPWORDS, EN_STOPWORDS


def tokenize_words(text: str) -> list[str]:
    """Extract words from text."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in raw if re.search(r"[^\W\d_]", s, re.UNICODE)]


def normalize_ascii(word: str) -> str:
    """Transliterate Turkish characters for readability heuristics."""
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return word.translate(tr_map)


def syllable_count(word: str) -> int:
    """Count syllables in an English word."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    
    if w.endswith("e") and count > 1:
        count -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in vowels:
        count += 1
    
    return max(1, count)


def detect_language(content: str, categories: str = "") -> str:
    """Detect if content is Turkish or English."""
    words = tokenize_words(content)
    text = content or ""
    
    tr_char_hits = len(TR_CHARS_RE.findall(text))
    tr_stop_hits = sum(1 for w in words if w in TR_STOPWORDS)
    en_stop_hits = sum(1 for w in words if w in EN_STOPWORDS)
    
    category_lc = (categories or "").lower()
    if "türkçe" in category_lc:
        return "tr"
    if "in english" in category_lc:
        return "en"
    if tr_char_hits >= 4 or tr_stop_hits > en_stop_hits * 1.2:
        return "tr"
    
    return "en"


def is_heading_like_line(line: str) -> bool:
    """Check if a line looks like a heading."""
    stripped = line.strip()
    if not stripped:
        return False
    
    words = tokenize_words(stripped)
    if not words or len(words) > 10:
        return False
    if stripped.endswith((".", "!", "?", ";", ":")):
        return False
    
    initials = sum(
        1 for token in re.findall(r"\b[^\W\d_][^\s]*", stripped, re.UNICODE)
        if token[:1].isupper()
    )
    return initials >= max(1, math.ceil(len(words) * 0.6))


def is_code_like_sentence(sentence: str) -> bool:
    """Detect if a sentence looks like code."""
    s = sentence.strip()
    if not s:
        return True
    
    # Common code patterns
    code_patterns = [
        r"(#include|System\.out|console\.log|printf|cout|std::|public static|int main)",
        r"::|->|==|!=|\+\+|--|&&|\|\|",
        r"\b(public|private|protected|class|static|void|return|const|var|let|function)\b.*[{};<>#]",
        r"^\s*//",
        r"\b\w+\.(h|hpp|c|cpp|java|py|js|ts|cs)\b",
        r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*[;{]?.*[;{}]",
        r"^\s*[A-Za-z_][A-Za-z0-9_<>]*\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^.]*;?\s*$",
        r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=[^=].*;?\s*$",
        r"^\s*(if|for|while|switch)\s*\(",
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, s, re.I):
            return True
    
    # High symbol density
    symbol_ratio = len(re.findall(r"[^0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\s]", s)) / max(len(s), 1)
    return symbol_ratio > 0.18
