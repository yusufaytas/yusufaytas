# WordPress Prose Analyzer

Converts WordPress XML exports to structured text and analyzes prose quality with detailed metrics and rankings.

## Structure

```
prose_analyzer/          # Core analysis package
├── __init__.py         # Package exports
├── models.py           # Data models (Post, ProseMetrics, ProseScores)
├── constants.py        # Regex patterns, stop words, markers
├── text_utils.py       # Text processing utilities
├── scoring.py          # Scoring helper functions
├── metrics.py          # Calculate raw prose metrics
├── scorer.py           # Convert metrics to quality scores
├── parser.py           # Parse WordPress export format
├── summarizer.py       # Generate content summaries
├── ranker.py           # Rank posts by quality
└── analyzer.py         # Main orchestration

convert_wp_xml_to_posts.py      # Convert WP XML → posts.txt
generate_prose_ratings_new.py   # Analyze posts → rankings CSV
```

## Installation

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install package with dependencies

```bash
pip install -e .
```

This installs:
- All dependencies (nltk, sumy, scipy, textstat, lexicalrichness, scikit-learn)
- CLI commands: `convert-wp` and `generate-ratings`

### 3. Download NLTK tokenizer data

The summarizer expects NLTK tokenizers to exist under `.venv/nltk_data`.

```bash
python -c "import nltk; nltk.download('punkt', download_dir='.venv/nltk_data'); nltk.download('punkt_tab', download_dir='.venv/nltk_data')"
```

### Quick Setup (one-liner)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e . && python -c "import nltk; nltk.download('punkt', download_dir='.venv/nltk_data'); nltk.download('punkt_tab', download_dir='.venv/nltk_data')"
```

## Usage

### Convert WordPress Export

```bash
# Activate venv first
source .venv/bin/activate

# Convert to JSONL (LLM optimized)
python3 convert_wp_xml_to_posts.py your-export.xml

# Custom output path
python3 convert_wp_xml_to_posts.py your-export.xml -o my-posts.jsonl

# Using CLI command
convert-wp your-export.xml
```

Outputs `posts.jsonl` - JSONL format optimized for LLM use (one JSON object per line).

### Generate Quality Rankings

```bash
# Activate venv first
source .venv/bin/activate

# Run analyzer (auto-detects input format)
python3 generate_prose_ratings.py
# or
generate-ratings
```

**Outputs:**
- `post_rankings.csv` - Spreadsheet format (Excel, Google Sheets)
- `post_rankings.jsonl` - LLM/RAG optimized (one JSON per line)
- `post_rankings.md` - Human-readable report with top 10 and stats

## Quality Metrics

The analyzer evaluates:

- **Clarity**: Readability, sentence length, passive voice, paragraph structure
- **Coherence**: Transitions, semantic similarity, repetition, structure
- **Richness**: Lexical diversity (MTLD, Yule's K), word complexity
- **Specificity**: Numbers, proper nouns, examples, technical depth
- **Concision**: Hedge words, redundancy, optimal length
- **Rhythm**: Sentence length variation
- **Depth**: Content substance, heading structure, paragraph quality
- **Reasoning**: Logical markers and argumentation
- **Structure**: Organization and formatting

## Output

CSV with columns:
- Rankings: overall_rank, overall_percentile, language_rank, language_percentile
- Metadata: post_id, title, date, url, slug, categories, tags, language
- Scores: overall_rating_100, readability_rating_100, substance_rating_100
- Summary: Auto-generated content summary

## Architecture Benefits

- **Modular**: Each module has single responsibility
- **Testable**: Pure functions, clear interfaces
- **Maintainable**: Easy to modify scoring algorithms
- **Extensible**: Add new metrics or scorers independently
- **Type-safe**: Dataclasses for all models
- **Reusable**: Import `prose_analyzer` in other projects
