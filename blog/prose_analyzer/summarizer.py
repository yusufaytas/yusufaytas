"""Generate summaries of content."""

from pathlib import Path
import nltk
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

from .text_utils import split_sentences


def generate_summary(
    content: str,
    title: str = "",
    language: str = "en",
    min_sentences: int = 3,
    max_sentences: int = 5,
    max_chars: int = 900
) -> str:
    """Generate a summary of the content using LSA."""
    nltk_data = Path(__file__).resolve().parent.parent / ".venv" / "nltk_data"
    nltk.data.path.insert(0, str(nltk_data))

    tokenizer_lang = "turkish" if language == "tr" else "english"
    parser = PlaintextParser.from_string(content, Tokenizer(tokenizer_lang))
    summarizer = LsaSummarizer()

    est = max(1, round(len(split_sentences(content)) * 0.22))
    sentence_count = max(min_sentences, min(max_sentences, est))

    summary_sentences = [str(s).strip() for s in summarizer(parser.document, sentence_count)]
    summary = " ".join(s for s in summary_sentences if s)

    return summary[:max_chars].rstrip()
