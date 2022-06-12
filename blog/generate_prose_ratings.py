#!/usr/bin/env python3
import csv
import math
import re
import sys
from pathlib import Path

INPUT_FILE = Path("posts.txt")
RANKED_OUTPUT_FILE = Path("post_rankings.csv")


def _enable_local_venv_packages():
    root = Path(__file__).resolve().parent
    lib_dir = root / ".venv" / "lib"
    if not lib_dir.exists():
        return
    for site_packages in sorted(lib_dir.glob("python*/site-packages")):
        path_str = str(site_packages)
        if path_str not in sys.path:
            sys.path.append(path_str)


_enable_local_venv_packages()

try:
    import nltk
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lsa import LsaSummarizer

    HAS_SUMY = True
except Exception:
    HAS_SUMY = False

try:
    from scipy.stats import rankdata

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}

TRANSITIONS = {
    "however", "therefore", "thus", "moreover", "meanwhile", "instead", "otherwise",
    "furthermore", "consequently", "similarly", "finally", "first", "second", "third",
    "in contrast", "for example", "for instance", "in addition", "on the other hand",
    "in consequence", "by contrast", "as a result",
}

HEDGES = {
    "maybe", "perhaps", "might", "could", "possibly", "seems", "appears", "likely",
    "generally", "usually", "often", "sometimes", "arguably", "roughly", "mostly",
}

EXAMPLE_MARKERS = {
    "for example", "for instance", "e.g.", "such as", "including", "like",
}

BE_VERBS = {"is", "are", "was", "were", "be", "been", "being", "am"}
WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)?", re.UNICODE)
ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
TR_CHARS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def score_range(value, ideal_low, ideal_high, hard_low, hard_high):
    if value < hard_low or value > hard_high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return 100.0 * (value - hard_low) / (ideal_low - hard_low)
    return 100.0 * (hard_high - value) / (hard_high - ideal_high)


def syllable_count(word):
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


def normalize_ascii(word):
    # Basic transliteration for mixed-language readability heuristics.
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return word.translate(tr_map)


def split_sentences(text):
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in raw if re.search(r"[^\W\d_]", s, re.UNICODE)]


def tokenize_words(text):
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def detect_language(content, categories=""):
    words = tokenize_words(content)
    text = content or ""
    tr_char_hits = len(TR_CHARS_RE.findall(text))
    tr_stop = {
        "ve", "bir", "bu", "için", "ile", "ama", "gibi", "çok", "daha", "olarak", "mi",
        "de", "da", "şu", "ise", "ya", "hem", "kadar", "çünkü", "ancak", "sonra",
    }
    en_stop = {
        "the", "and", "or", "with", "for", "is", "are", "that", "this", "from", "have",
        "was", "were", "not", "you", "your", "we", "they", "their", "it", "as",
    }
    tr_stop_hits = sum(1 for w in words if w in tr_stop)
    en_stop_hits = sum(1 for w in words if w in en_stop)
    category_lc = (categories or "").lower()
    if "türkçe" in category_lc:
        return "tr"
    if "in english" in category_lc:
        return "en"
    if tr_char_hits >= 4 or tr_stop_hits > en_stop_hits * 1.2:
        return "tr"
    return "en"


def term_vector(tokens, idf):
    if not tokens:
        return {}
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens)
    vec = {}
    for t, c in tf.items():
        vec[t] = (c / total) * idf.get(t, 0.0)
    return vec


def cosine_sim(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = 0.0
    for k, v in vec_a.items():
        dot += v * vec_b.get(k, 0.0)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def is_code_like_sentence(sentence):
    s = sentence.strip()
    if not s:
        return True
    if re.search(r"(#include|System\.out|console\.log|printf|cout|std::|public static|int main)", s):
        return True
    if re.search(r"::|->|==|!=|\+\+|--|&&|\|\|", s):
        return True
    if re.search(r"\b(public|private|protected|class|static|void|return|const|var|let|function)\b", s, re.I) and re.search(r"[{};<>#]", s):
        return True
    if re.search(r"^\s*//", s):
        return True
    if re.search(r"\b\w+\.(h|hpp|c|cpp|java|py|js|ts|cs)\b", s, re.I):
        return True
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*[;{]?", s) and re.search(r"[;{}]", s):
        return True
    if re.search(r"^\s*[A-Za-z_][A-Za-z0-9_<>]*\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^.]*;?\s*$", s):
        return True
    if re.search(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=[^=].*;?\s*$", s):
        return True
    if re.search(r"^\s*(if|for|while|switch)\s*\(", s):
        return True
    symbol_ratio = len(re.findall(r"[^0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\s]", s)) / max(len(s), 1)
    return symbol_ratio > 0.18


def parse_posts(text):
    blocks = re.findall(r"<<<POST_START_(\d{4})>>>\n(.*?)\n<<<POST_END_\1>>>", text, re.S)
    posts = []
    for post_id, block in blocks:
        def field(name):
            m = re.search(rf"^{re.escape(name)}:\s*(.*)$", block, re.M)
            return m.group(1).strip() if m else ""

        cm = re.search(r"^CONTENT:\n(.*)$", block, re.S | re.M)
        content = cm.group(1).strip() if cm else ""

        posts.append(
            {
                "post_id": int(post_id),
                "title": field("TITLE"),
                "date": field("DATE"),
                "url": field("URL"),
                "slug": field("SLUG"),
                "categories": field("CATEGORIES"),
                "tags": field("TAGS"),
                "content": content,
            }
        )
    return posts


def prose_metrics(content, categories=""):
    language = detect_language(content, categories=categories)
    lines = [ln.strip() for ln in content.splitlines()]
    nonempty_lines = [ln for ln in lines if ln]
    prose_lines = [ln for ln in nonempty_lines if not is_code_like_sentence(ln)]
    prose_text = "\n".join(prose_lines)

    words = tokenize_words(prose_text)
    lower_words = [w.lower() for w in words]
    alpha_words = [w for w in lower_words if re.search(r"[^\W\d_]", w, re.UNICODE)]
    sentences = [s for s in split_sentences(prose_text) if not is_code_like_sentence(s)]
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", prose_text) if p.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)

    sent_lengths = [len(re.findall(r"[A-Za-z][A-Za-z'-]*", s)) for s in sentences] or [0]
    avg_sentence_len = (sum(sent_lengths) / len(sent_lengths)) if sent_lengths else 0.0
    sent_std = (
        (sum((x - avg_sentence_len) ** 2 for x in sent_lengths) / len(sent_lengths)) ** 0.5
        if sent_lengths and avg_sentence_len > 0
        else 0.0
    )
    sentence_cv = (sent_std / avg_sentence_len) if avg_sentence_len else 0.0

    unique_words = len(set(alpha_words))
    ttr = (unique_words / len(alpha_words)) if alpha_words else 0.0
    root_ttr = (unique_words / math.sqrt(len(alpha_words))) if alpha_words else 0.0
    long_word_ratio = (
        sum(1 for w in alpha_words if len(w) >= 7) / len(alpha_words) if alpha_words else 0.0
    )
    content_ratio = (
        sum(1 for w in alpha_words if w not in STOPWORDS) / len(alpha_words) if alpha_words else 0.0
    )

    ascii_words = [normalize_ascii(w) for w in alpha_words]
    total_syllables = sum(syllable_count(w) for w in ascii_words)
    syllables_per_word = (total_syllables / len(ascii_words)) if ascii_words else 0.0
    if language == "en" and ascii_words:
        flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * syllables_per_word)
    else:
        # Flesch is English-specific; keep a neutral baseline for non-English prose.
        flesch = 62.0

    passive_hits = 0
    for i in range(len(lower_words) - 1):
        if lower_words[i] in BE_VERBS and re.search(r"(ed|en)$", lower_words[i + 1]):
            passive_hits += 1
    passive_ratio = passive_hits / max(sentence_count, 1)

    text_lc = prose_text.lower()
    transition_hits = sum(text_lc.count(t) for t in TRANSITIONS)
    transition_density = transition_hits / max(word_count, 1) * 1000.0

    hedge_hits = sum(lower_words.count(h) for h in HEDGES)
    hedge_ratio = hedge_hits / max(word_count, 1)

    example_hits = sum(text_lc.count(m) for m in EXAMPLE_MARKERS)
    number_density = len(re.findall(r"\b\d+(\.\d+)?\b", prose_text)) / max(word_count, 1) * 100.0

    caps_tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", prose_text)
    proper_noun_density = len(caps_tokens) / max(word_count, 1) * 100.0

    para_word_counts = [len(re.findall(r"[A-Za-z][A-Za-z'-]*", p)) for p in paragraphs] or [0]
    avg_para_words = sum(para_word_counts) / len(para_word_counts) if para_word_counts else 0.0

    non_stop = [w for w in alpha_words if w not in STOPWORDS]
    freq = {}
    for w in non_stop:
        freq[w] = freq.get(w, 0) + 1
    top_ratio = (max(freq.values()) / max(len(non_stop), 1)) if freq else 0.0

    sentence_norm = [" ".join(tokenize_words(s)) for s in sentences if s.strip()]
    duplicate_sentence_ratio = 0.0
    if sentence_norm:
        duplicate_sentence_ratio = 1.0 - (len(set(sentence_norm)) / len(sentence_norm))

    code_line_ratio = (len(nonempty_lines) - len(prose_lines)) / max(len(nonempty_lines), 1)
    quote_like_lines = 0
    for ln in prose_lines:
        wc = len(tokenize_words(ln))
        if wc == 0:
            continue
        starts_quote = ln.startswith(("\"", "“", "‘", "-", "—", "•"))
        if starts_quote and wc <= 18:
            quote_like_lines += 1
    quote_line_ratio = quote_like_lines / max(len(prose_lines), 1)

    metrics = {
        "language": language,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_len": avg_sentence_len,
        "sentence_cv": sentence_cv,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "long_word_ratio": long_word_ratio,
        "content_ratio": content_ratio,
        "flesch_reading_ease": flesch,
        "passive_ratio": passive_ratio,
        "transition_density_per_1k": transition_density,
        "hedge_ratio": hedge_ratio,
        "example_hits": example_hits,
        "number_density_per_100": number_density,
        "proper_noun_density_per_100": proper_noun_density,
        "avg_paragraph_words": avg_para_words,
        "top_term_ratio": top_ratio,
        "duplicate_sentence_ratio": duplicate_sentence_ratio,
        "code_line_ratio": code_line_ratio,
        "quote_line_ratio": quote_line_ratio,
    }
    return metrics


def prose_scores(m):
    readability = score_range(m["flesch_reading_ease"], 50, 75, 10, 100)
    sentence_len_score = score_range(m["avg_sentence_len"], 12, 24, 5, 40)
    passive_score = clamp(100.0 - (m["passive_ratio"] * 220.0))
    paragraph_score = score_range(m["avg_paragraph_words"], 35, 130, 10, 220)
    length_gate = score_range(m["word_count"], 220, 1600, 60, 3000)
    clarity = (
        readability * 0.35
        + sentence_len_score * 0.25
        + passive_score * 0.15
        + paragraph_score * 0.10
        + length_gate * 0.15
    )

    transition_score = score_range(m["transition_density_per_1k"], 1.2, 6.0, 0.0, 12.0)
    repetition_score = clamp(100.0 - (m["top_term_ratio"] * 420.0))
    duplicate_score = clamp(100.0 - (m["duplicate_sentence_ratio"] * 200.0))
    structure_score = score_range(m["paragraph_count"], 3, 20, 1, 40)
    coherence = (
        transition_score * 0.30
        + repetition_score * 0.25
        + duplicate_score * 0.20
        + structure_score * 0.25
    )

    lexical_score = score_range(m["root_ttr"], 5.8, 9.8, 3.0, 14.0)
    long_word_score = score_range(m["long_word_ratio"], 0.16, 0.33, 0.05, 0.48)
    content_word_score = score_range(m["content_ratio"], 0.44, 0.72, 0.28, 0.90)
    richness = lexical_score * 0.45 + long_word_score * 0.25 + content_word_score * 0.30

    number_score = score_range(m["number_density_per_100"], 0.5, 8.0, 0.0, 14.0)
    proper_noun_score = score_range(m["proper_noun_density_per_100"], 0.4, 6.5, 0.0, 12.0)
    example_score = score_range(m["example_hits"], 1, 6, 0, 12)
    specificity = number_score * 0.35 + proper_noun_score * 0.30 + example_score * 0.35

    hedge_score = clamp(100.0 - (m["hedge_ratio"] * 1800.0))
    concision_len = score_range(m["word_count"], 250, 1300, 80, 2400)
    redundancy_score = clamp(100.0 - (m["duplicate_sentence_ratio"] * 240.0) - (m["top_term_ratio"] * 260.0))
    concision = hedge_score * 0.20 + concision_len * 0.35 + redundancy_score * 0.45

    rhythm_variety = score_range(m["sentence_cv"], 0.35, 0.95, 0.1, 1.6)
    rhythm = rhythm_variety

    code_penalty = clamp(100.0 - (m["code_line_ratio"] * 120.0))
    quote_balance = clamp(100.0 - (m["quote_line_ratio"] * 140.0))
    depth = (
        score_range(m["word_count"], 260, 1700, 90, 3400) * 0.55
        + score_range(m["paragraph_count"], 4, 26, 1, 55) * 0.25
        + quote_balance * 0.20
    )

    return {
        "clarity_100": round(clamp(clarity), 2),
        "coherence_100": round(clamp(coherence), 2),
        "richness_100": round(clamp(richness), 2),
        "specificity_100": round(clamp(specificity), 2),
        "concision_100": round(clamp(concision), 2),
        "rhythm_100": round(clamp(rhythm), 2),
        "depth_100": round(clamp(depth), 2),
        "code_cleanliness_100": round(clamp(code_penalty), 2),
        "quote_balance_100": round(clamp(quote_balance), 2),
    }


def percentile_from_rank(rank, total):
    return ((total - rank) / max(total - 1, 1)) * 100.0


def rank_desc(rows, score_key):
    if HAS_SCIPY:
        adjusted_scores = [
            float(row[score_key]) - (row["post_id"] * 1e-9)
            for row in rows
        ]
        ranks = rankdata([-score for score in adjusted_scores], method="ordinal")
        return {
            row["post_id"]: int(ranks[idx])
            for idx, row in enumerate(rows)
        }

    ordered = sorted(rows, key=lambda r: (r[score_key], -r["post_id"]), reverse=True)
    return {row["post_id"]: idx for idx, row in enumerate(ordered, start=1)}


def quick_summary(content, title="", language="en", min_sentences=3, max_sentences=5, max_chars=900):
    if not HAS_SUMY:
        raise RuntimeError(
            "Missing summarization dependencies. Install 'sumy' and 'nltk' to generate summaries."
        )

    nltk_data = Path(__file__).resolve().parent / ".venv" / "nltk_data"
    nltk.data.path.insert(0, str(nltk_data))
    tokenizer_lang = "turkish" if language == "tr" else "english"
    parser = PlaintextParser.from_string(content, Tokenizer(tokenizer_lang))
    summarizer = LsaSummarizer()
    est = max(1, round(len(split_sentences(content)) * 0.22))
    sentence_count = max(min_sentences, min(max_sentences, est))
    summary_sentences = [str(s).strip() for s in summarizer(parser.document, sentence_count)]
    summary = " ".join(s for s in summary_sentences if s)
    return summary[:max_chars].rstrip()


def aggregate_output_scores(scores):
    readability = (
        scores["clarity_100"] * 0.5
        + scores["coherence_100"] * 0.3
        + scores["concision_100"] * 0.2
    )
    substance = (
        scores["richness_100"] * 0.35
        + scores["specificity_100"] * 0.20
        + scores["depth_100"] * 0.45
    )
    readability = clamp(readability)
    substance = clamp(substance)

    harmonic_core = 0.0
    if readability > 0.0 and substance > 0.0:
        harmonic_core = (2.0 * readability * substance) / (readability + substance)

    overall = (
        harmonic_core * 0.7
        + ((readability + substance) / 2.0) * 0.2
        + scores["code_cleanliness_100"] * 0.1
    )

    return {
        "readability_rating_100": round(readability, 2),
        "substance_rating_100": round(substance, 2),
        "overall_rating_100": round(clamp(overall), 2),
    }


def main():
    src = INPUT_FILE.read_text(encoding="utf-8")
    posts = parse_posts(src)

    rows = []
    for p in posts:
        m = prose_metrics(p["content"], categories=p["categories"])
        s = prose_scores(m)
        aggregate_scores = aggregate_output_scores(s)
        rows.append(
            {
                "post_id": p["post_id"],
                "title": p["title"],
                "date": p["date"],
                "url": p["url"],
                "slug": p["slug"],
                "categories": p["categories"],
                "tags": p["tags"],
                "language": m["language"],
                "summary": quick_summary(p["content"], title=p["title"], language=m["language"]),
                **aggregate_scores,
            }
        )

    total = len(rows)
    if total == 0:
        print("No posts found")
        return

    final_ranks = rank_desc(rows, "overall_rating_100")
    for row in rows:
        rank = final_ranks[row["post_id"]]
        row["overall_rank"] = rank
        row["overall_percentile"] = round(percentile_from_rank(rank, total), 2)

    lang_groups = {}
    for row in rows:
        lang_groups.setdefault(row["language"], []).append(row)
    for lang, group in lang_groups.items():
        group_sorted = sorted(group, key=lambda r: (r["overall_rank"], r["post_id"]))
        lang_total = len(group_sorted)
        for i, row in enumerate(group_sorted, start=1):
            row["language_rank"] = i
            row["language_percentile"] = round(percentile_from_rank(i, lang_total), 2)

    ranked = sorted(rows, key=lambda r: (r["overall_rank"], r["post_id"]))
    export_fields = [
        "overall_rank",
        "overall_percentile",
        "language_rank",
        "language_percentile",
        "post_id",
        "title",
        "date",
        "url",
        "slug",
        "categories",
        "tags",
        "language",
        "overall_rating_100",
        "readability_rating_100",
        "substance_rating_100",
        "summary",
    ]

    with RANKED_OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=export_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in export_fields} for row in ranked)

    print(f"Wrote {RANKED_OUTPUT_FILE} with {len(ranked)} rows")


if __name__ == "__main__":
    main()
