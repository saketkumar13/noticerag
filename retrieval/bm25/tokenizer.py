"""
BM25 Tokenizer Module.
Handles case folding, punctuation normalization, stopword filtering,
and token extraction for university notices.
"""

import re
from typing import List, Set

# Standard English stopwords
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class BM25Tokenizer:
    """
    Tokenizes text for BM25 indexing and querying.
    """

    # Matches alphanumeric words, preserving hyphenated terms and dots (e.g. ph.d, b.tech)
    TOKEN_REGEX = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")

    @classmethod
    def tokenize(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Extracts lowercase tokens, removing punctuation and optional stopwords.
        """
        if not text:
            return []

        # Find all matching words
        raw_tokens = cls.TOKEN_REGEX.findall(text.lower())

        clean_tokens = []
        for token in raw_tokens:
            token = token.strip(".-_")
            if not token:
                continue

            # Length filter
            if len(token) < 2 and not token.isdigit():
                continue

            if remove_stopwords and token in ENGLISH_STOPWORDS:
                continue

            clean_tokens.append(token)

        return clean_tokens
