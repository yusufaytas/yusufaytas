"""Convert metrics to quality scores - optimized for insightful blog posts."""

from .models import ProseMetrics, ProseScores
from .scoring import clamp, score_range, score_min


def calculate_scores(m: ProseMetrics) -> ProseScores:
    """
    Calculate quality score optimized for thoughtful, insightful blog posts.
    
    Philosophy:
    - Substance > Style: Deep thinking matters more than perfect grammar
    - Clarity > Cleverness: Direct communication beats academic formality  
    - Insight > Length: Quality of ideas, not word count
    - Natural > Formal: Conversational clarity is good, not bad
    """
    
    # === READABILITY (25%): Easy and engaging to read ===
    sentence_score = score_range(m.avg_sentence_len, 15, 22, 8, 35)
    rhythm_score = score_range(m.sentence_cv, 0.4, 0.8, 0.15, 1.4)
    passive_score = clamp(100.0 - (m.passive_ratio * 150.0))
    para_score = score_range(m.avg_paragraph_words, 50, 120, 20, 200)
    flesch_score = score_range(m.flesch_reading_ease, 55, 70, 30, 85)
    
    readability = (
        sentence_score * 0.25 +
        rhythm_score * 0.25 +
        passive_score * 0.15 +
        para_score * 0.20 +
        flesch_score * 0.15
    )
    
    # === SUBSTANCE (30%): Depth and insight ===
    length_score = score_range(m.word_count, 800, 2500, 300, 5000)
    mtld_score = score_range(m.mtld, 70, 140, 40, 200)
    content_score = score_range(m.content_ratio, 0.50, 0.70, 0.35, 0.85)
    example_score = score_range(m.example_hits, 2, 8, 0, 15)
    number_score = score_range(m.number_density_per_100, 1.0, 6.0, 0.0, 12.0)
    structure_score = (
        score_min(m.paragraph_count, 8, 2) * 0.6 +
        score_min(m.heading_count, 4, 0) * 0.4
    )
    
    substance = (
        length_score * 0.20 +
        mtld_score * 0.20 +
        content_score * 0.15 +
        example_score * 0.15 +
        number_score * 0.10 +
        structure_score * 0.20
    )
    
    # === COHERENCE (20%): Flow of ideas ===
    transition_score = score_range(m.transition_density_per_1k, 2.0, 8.0, 0.5, 15.0)
    flow_score = score_range(m.adjacent_cosine, 0.10, 0.40, 0.0, 0.65)
    repetition_penalty = clamp(100.0 - (m.top_term_ratio * 300.0))
    duplicate_penalty = clamp(100.0 - (m.duplicate_sentence_ratio * 150.0))
    
    coherence = (
        transition_score * 0.30 +
        flow_score * 0.30 +
        repetition_penalty * 0.20 +
        duplicate_penalty * 0.20
    )
    
    # === CLARITY (15%): Precision and directness ===
    hedge_penalty = clamp(100.0 - (m.hedge_ratio * 1200.0))
    reasoning_score = score_range(m.reasoning_density_per_1k, 3.0, 12.0, 0.5, 20.0)
    para_structure = score_range(m.avg_paragraph_words, 50, 120, 20, 200)
    
    clarity = (
        hedge_penalty * 0.30 +
        reasoning_score * 0.35 +
        para_structure * 0.35
    )
    
    # === FORMAT (10%): Presentation quality ===
    code_penalty = clamp(100.0 - (m.code_line_ratio * 80.0))
    short_para_penalty = clamp(100.0 - (m.short_paragraph_ratio * 100.0))
    quote_penalty = clamp(100.0 - (m.quote_line_ratio * 120.0))
    
    format_quality = (
        code_penalty * 0.40 +
        short_para_penalty * 0.30 +
        quote_penalty * 0.30
    )
    
    # === OVERALL QUALITY ===
    overall = (
        clamp(readability) * 0.25 +
        clamp(substance) * 0.30 +
        clamp(coherence) * 0.20 +
        clamp(clarity) * 0.15 +
        clamp(format_quality) * 0.10
    )
    
    return ProseScores(
        clarity_100=round(clamp(clarity), 2),
        coherence_100=round(clamp(coherence), 2),
        richness_100=round(clamp(substance), 2),
        specificity_100=round(clamp(substance), 2),
        concision_100=round(clamp(clarity), 2),
        rhythm_100=round(clamp(readability), 2),
        depth_100=round(clamp(substance), 2),
        reasoning_100=round(clamp(reasoning_score), 2),
        structure_100=round(clamp(structure_score), 2),
        format_discipline_100=round(clamp(format_quality), 2),
        code_cleanliness_100=round(code_penalty, 2),
        quote_balance_100=round(quote_penalty, 2),
        readability_rating_100=round(clamp(readability), 2),
        substance_rating_100=round(clamp(substance), 2),
        overall_rating_100=round(clamp(overall), 2),
    )
