"""Data models for prose analysis."""

from dataclasses import dataclass


@dataclass
class Post:
    """Represents a blog post."""
    post_id: int
    title: str
    date: str
    url: str
    slug: str
    categories: str
    tags: str
    content: str


@dataclass
class ProseMetrics:
    """Raw prose quality metrics."""
    language: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    heading_count: int
    avg_sentence_len: float
    sentence_cv: float
    ttr: float
    root_ttr: float
    mtld: float
    yule_k: float
    long_word_ratio: float
    content_ratio: float
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    smog_index: float
    dale_chall: float
    adjacent_cosine: float
    passive_ratio: float
    transition_density_per_1k: float
    reasoning_density_per_1k: float
    hedge_ratio: float
    example_hits: int
    number_density_per_100: float
    proper_noun_density_per_100: float
    avg_paragraph_words: float
    top_term_ratio: float
    duplicate_sentence_ratio: float
    code_line_ratio: float
    quote_line_ratio: float
    short_paragraph_ratio: float


@dataclass
class ProseScores:
    """Computed prose quality scores (0-100)."""
    clarity_100: float
    coherence_100: float
    richness_100: float
    specificity_100: float
    concision_100: float
    rhythm_100: float
    depth_100: float
    reasoning_100: float
    structure_100: float
    format_discipline_100: float
    code_cleanliness_100: float
    quote_balance_100: float
    readability_rating_100: float
    substance_rating_100: float
    overall_rating_100: float
