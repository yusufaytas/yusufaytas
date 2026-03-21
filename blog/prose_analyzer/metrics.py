"""Calculate prose metrics from text."""

import math
import re
import textstat
from lexicalrichness import LexicalRichness
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import BE_VERBS, STOPWORDS, TRANSITIONS, REASONING_MARKERS, HEDGES, EXAMPLE_MARKERS
from .models import ProseMetrics
from .text_utils import (
    tokenize_words,
    split_sentences,
    normalize_ascii,
    syllable_count,
    detect_language,
    is_heading_like_line,
    is_code_like_sentence,
)


def calculate_metrics(content: str, categories: str = "") -> ProseMetrics:
    """Calculate all prose metrics for given content."""
    language = detect_language(content, categories=categories)
    lines = [ln.strip() for ln in content.splitlines()]
    nonempty_lines = [ln for ln in lines if ln]

    # Filter out code and identify headings
    prose_lines = []
    heading_count = 0
    for ln in lines:
        if not ln:
            prose_lines.append("")
            continue
        if is_code_like_sentence(ln):
            continue
        if is_heading_like_line(ln):
            heading_count += 1
        prose_lines.append(ln)

    prose_text = "\n".join(prose_lines).strip()

    # Basic tokenization
    words = tokenize_words(prose_text)
    lower_words = [w.lower() for w in words]
    alpha_words = [w for w in lower_words if re.search(r"[^\W\d_]", w, re.UNICODE)]
    sentences = [s for s in split_sentences(prose_text) if not is_code_like_sentence(s)]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", prose_text) if p.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)

    # Sentence length statistics
    sent_lengths = [len(re.findall(r"[A-Za-z][A-Za-z'-]*", s)) for s in sentences] or [0]
    avg_sentence_len = (sum(sent_lengths) / len(sent_lengths)) if sent_lengths else 0.0
    sent_std = (
        (sum((x - avg_sentence_len) ** 2 for x in sent_lengths) / len(sent_lengths)) ** 0.5
        if sent_lengths and avg_sentence_len > 0
        else 0.0
    )
    sentence_cv = (sent_std / avg_sentence_len) if avg_sentence_len else 0.0

    # Lexical diversity
    unique_words = len(set(alpha_words))
    ttr = (unique_words / len(alpha_words)) if alpha_words else 0.0
    root_ttr = (unique_words / math.sqrt(len(alpha_words))) if alpha_words else 0.0
    long_word_ratio = (
        sum(1 for w in alpha_words if len(w) >= 7) / len(alpha_words) if alpha_words else 0.0
    )
    content_ratio = (
        sum(1 for w in alpha_words if w not in STOPWORDS) / len(alpha_words) if alpha_words else 0.0
    )

    # Readability metrics (English only)
    ascii_words = [normalize_ascii(w) for w in alpha_words]
    total_syllables = sum(syllable_count(w) for w in ascii_words)
    syllables_per_word = (total_syllables / len(ascii_words)) if ascii_words else 0.0
    
    flesch = 62.0
    flesch_kincaid = 8.0
    smog = 8.0
    dale_chall = 7.0
    mtld = 70.0
    yule_k = 100.0
    adjacent_cosine = 0.15
    
    if language == "en" and content.strip():
        flesch = textstat.flesch_reading_ease(prose_text)
        flesch_kincaid = textstat.flesch_kincaid_grade(prose_text)
        smog = textstat.smog_index(prose_text)
        dale_chall = textstat.dale_chall_readability_score(prose_text)
        
        try:
            lex = LexicalRichness(prose_text)
            mtld = lex.mtld(threshold=0.72)
            yule_k = lex.yulek
        except Exception:
            pass
            
        # Adjacent sentence similarity
        valid_sentences = [s for s in sentences if len(s.split()) > 3]
        if len(valid_sentences) > 1:
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(valid_sentences)
                similarities = []
                for i in range(len(valid_sentences) - 1):
                    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])[0][0]
                    similarities.append(sim)
                adjacent_cosine = sum(similarities) / len(similarities)
            except Exception:
                pass
    elif language == "en" and ascii_words:
        flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * syllables_per_word)

    # Passive voice detection
    passive_hits = 0
    for i in range(len(lower_words) - 1):
        if lower_words[i] in BE_VERBS and re.search(r"(ed|en)$", lower_words[i + 1]):
            passive_hits += 1
    passive_ratio = passive_hits / max(sentence_count, 1)

    # Transition and reasoning markers
    text_lc = prose_text.lower()
    transition_hits = sum(text_lc.count(t) for t in TRANSITIONS)
    transition_density = transition_hits / max(word_count, 1) * 1000.0

    reasoning_hits = sum(text_lc.count(marker) for marker in REASONING_MARKERS)
    reasoning_density = reasoning_hits / max(word_count, 1) * 1000.0

    hedge_hits = sum(lower_words.count(h) for h in HEDGES)
    hedge_ratio = hedge_hits / max(word_count, 1)

    example_hits = sum(text_lc.count(m) for m in EXAMPLE_MARKERS)
    number_density = len(re.findall(r"\b\d+(\.\d+)?\b", prose_text)) / max(word_count, 1) * 100.0

    # Proper nouns
    caps_tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", prose_text)
    proper_noun_density = len(caps_tokens) / max(word_count, 1) * 100.0

    # Paragraph statistics
    para_word_counts = [len(re.findall(r"[A-Za-z][A-Za-z'-]*", p)) for p in paragraphs] or [0]
    avg_para_words = sum(para_word_counts) / len(para_word_counts) if para_word_counts else 0.0

    # Term frequency
    non_stop = [w for w in alpha_words if w not in STOPWORDS]
    freq = {}
    for w in non_stop:
        freq[w] = freq.get(w, 0) + 1
    top_ratio = (max(freq.values()) / max(len(non_stop), 1)) if freq else 0.0

    # Duplicate sentences
    sentence_norm = [" ".join(tokenize_words(s)) for s in sentences if s.strip()]
    duplicate_sentence_ratio = 0.0
    if sentence_norm:
        duplicate_sentence_ratio = 1.0 - (len(set(sentence_norm)) / len(sentence_norm))

    # Code and quote detection
    code_line_ratio = (len(nonempty_lines) - len(prose_lines)) / max(len(nonempty_lines), 1)
    quote_like_lines = 0
    for ln in prose_lines:
        wc = len(tokenize_words(ln))
        if wc == 0:
            continue
        starts_quote = ln.startswith(('"', '"', "'", "-", "—", "•"))
        if starts_quote and wc <= 18:
            quote_like_lines += 1
    quote_line_ratio = quote_like_lines / max(len(prose_lines), 1)

    # Short paragraph detection
    content_paragraphs = [p for p in paragraphs if not is_heading_like_line(p)]
    short_paragraphs = [
        p for p in content_paragraphs
        if 0 < len(tokenize_words(p)) <= 24
    ]
    short_paragraph_ratio = len(short_paragraphs) / max(len(content_paragraphs), 1)

    return ProseMetrics(
        language=language,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        heading_count=heading_count,
        avg_sentence_len=avg_sentence_len,
        sentence_cv=sentence_cv,
        ttr=ttr,
        root_ttr=root_ttr,
        mtld=mtld,
        yule_k=yule_k,
        long_word_ratio=long_word_ratio,
        content_ratio=content_ratio,
        flesch_reading_ease=flesch,
        flesch_kincaid_grade=flesch_kincaid,
        smog_index=smog,
        dale_chall=dale_chall,
        adjacent_cosine=adjacent_cosine,
        passive_ratio=passive_ratio,
        transition_density_per_1k=transition_density,
        reasoning_density_per_1k=reasoning_density,
        hedge_ratio=hedge_ratio,
        example_hits=example_hits,
        number_density_per_100=number_density,
        proper_noun_density_per_100=proper_noun_density,
        avg_paragraph_words=avg_para_words,
        top_term_ratio=top_ratio,
        duplicate_sentence_ratio=duplicate_sentence_ratio,
        code_line_ratio=code_line_ratio,
        quote_line_ratio=quote_line_ratio,
        short_paragraph_ratio=short_paragraph_ratio,
    )
