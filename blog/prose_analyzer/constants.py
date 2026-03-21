"""Constants for prose analysis."""

import re

# Regular expressions
WORD_RE = re.compile(r"[^\W\d_]+(?:[''-][^\W\d_]+)?", re.UNICODE)
TR_CHARS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

# Verb sets
BE_VERBS = {"is", "are", "was", "were", "be", "been", "being", "am"}

# Stop words for content analysis
STOPWORDS = {
    "the", "and", "or", "with", "for", "is", "are", "that", "this", "from",
    "have", "was", "were", "not", "you", "your", "we", "they", "their", "it",
    "as", "at", "by", "an", "be", "to", "of", "in", "on", "will", "can",
    "has", "had", "but", "if", "when", "which", "who", "what", "where", "how",
}

# Transition words
TRANSITIONS = {
    "however", "therefore", "moreover", "furthermore", "consequently",
    "nevertheless", "nonetheless", "meanwhile", "additionally", "similarly",
    "conversely", "alternatively", "specifically", "particularly", "notably",
    "essentially", "ultimately", "generally", "typically", "basically",
}

# Reasoning markers
REASONING_MARKERS = {
    "because", "since", "thus", "hence", "therefore", "consequently",
    "as a result", "due to", "given that", "considering", "assuming",
    "implies", "suggests", "indicates", "demonstrates", "proves",
}

# Hedge words
HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "somewhat", "fairly", "rather", "quite", "relatively", "seemingly",
    "apparently", "arguably", "presumably", "supposedly",
}

# Example markers
EXAMPLE_MARKERS = {
    "for example", "for instance", "such as", "like", "including",
    "e.g.", "i.e.", "namely", "specifically",
}

# Turkish stop words
TR_STOPWORDS = {
    "ve", "bir", "bu", "için", "ile", "ama", "gibi", "çok", "daha",
    "olarak", "mi", "de", "da", "şu", "ise", "ya", "hem", "kadar",
    "çünkü", "ancak", "sonra",
}

# English stop words for language detection
EN_STOPWORDS = {
    "the", "and", "or", "with", "for", "is", "are", "that", "this",
    "from", "have", "was", "were", "not", "you", "your", "we", "they",
    "their", "it", "as",
}
