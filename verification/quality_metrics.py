"""
Quality Metrics and OCR Health Scoring Module.
Calculates text statistics, linguistic distributions, noise ratios,
detects OCR anomalies and garbage repetitions, and computes a 0-100 Health Score.
"""

import re
from typing import Any, Dict, List, Tuple


class QualityMetrics:
    """
    Evaluates OCR quality, flags suspicious text anomalies,
    and assigns a quality score from 0 to 100.
    """

    # Patterns indicating OCR failure, hallucination, or scanner artifacts
    SUSPICIOUS_PATTERNS = [
        re.compile(r"%{4,}"),          # %%%%%
        re.compile(r"I{5,}", re.I),    # IIIIII or iiiiii
        re.compile(r"O{5,}", re.I),    # OOOOOO or oooooo
        re.compile(r"1{5,}"),          # 111111
        re.compile(r"0{5,}"),          # 000000
        re.compile(r"\|{4,}"),         # ||||
        re.compile(r"_{5,}"),          # _____
        re.compile(r"~{4,}"),          # ~~~~
        re.compile(r"={5,}"),          # =====
        re.compile(r"\.{6,}"),         # ......
        re.compile(r"([^\w\s])\1{4,}"), # Any non-word char repeated 5+ times
    ]

    @classmethod
    def calculate_text_stats(cls, text: str) -> Dict[str, int]:
        """
        Calculates character, word, line, and paragraph counts.
        """
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "line_count": 0,
                "paragraph_count": 0,
            }

        chars = len(text)
        words = text.split()
        word_count = len(words)

        lines = text.splitlines()
        line_count = len(lines)

        # Paragraphs separated by blank lines
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        paragraph_count = max(len(paragraphs), 1 if text.strip() else 0)

        return {
            "char_count": chars,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count,
        }

    @classmethod
    def calculate_quality_metrics(cls, text: str) -> Dict[str, float]:
        """
        Calculates linguistic and structural quality indicators:
        - average word length
        - unique word count
        - empty line ratio
        - non-alphanumeric ratio
        """
        if not text or not text.strip():
            return {
                "avg_word_length": 0.0,
                "unique_word_count": 0,
                "unique_word_ratio": 0.0,
                "empty_line_ratio": 0.0,
                "non_alphanumeric_ratio": 1.0,
            }

        words = text.split()
        total_words = len(words)
        if total_words == 0:
            return {
                "avg_word_length": 0.0,
                "unique_word_count": 0,
                "unique_word_ratio": 0.0,
                "empty_line_ratio": 0.0,
                "non_alphanumeric_ratio": 1.0,
            }

        clean_words = [re.sub(r"[^\w]", "", w.lower()) for w in words]
        clean_words = [w for w in clean_words if w]
        unique_words = set(clean_words)
        unique_count = len(unique_words)

        avg_word_len = (
            sum(len(w) for w in clean_words) / len(clean_words) if clean_words else 0.0
        )
        unique_ratio = unique_count / total_words if total_words else 0.0

        lines = text.splitlines()
        total_lines = len(lines)
        empty_lines = sum(1 for line in lines if not line.strip())
        empty_line_ratio = empty_lines / total_lines if total_lines else 0.0

        # Non-alphanumeric characters (excluding standard whitespace)
        total_chars = len(text)
        non_alnum_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        non_alnum_ratio = non_alnum_chars / total_chars if total_chars else 0.0

        return {
            "avg_word_length": round(avg_word_len, 2),
            "unique_word_count": unique_count,
            "unique_word_ratio": round(unique_ratio, 3),
            "empty_line_ratio": round(empty_line_ratio, 3),
            "non_alphanumeric_ratio": round(non_alnum_ratio, 3),
        }

    @classmethod
    def detect_suspicious_patterns(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Detects OCR garbage patterns such as %%%%%, IIIIII, repeated symbols, etc.
        Returns (is_suspicious, list_of_matched_patterns).
        """
        if not text:
            return False, []

        matches = []
        for pattern in cls.SUSPICIOUS_PATTERNS:
            found = pattern.findall(text)
            if found:
                for match in found[:3]:  # preview up to 3 per pattern
                    matches.append(match if isinstance(match, str) else match[0])

        return len(matches) > 0, matches

    @classmethod
    def compute_health_score(cls, text: str) -> Dict[str, Any]:
        """
        Computes an OCR Quality Health Score between 0 and 100.
        Score components:
          1. Completeness & Length (max 25 pts)
          2. Word Diversity / TTR (max 25 pts)
          3. Readability & Avg Word Length (max 25 pts)
          4. Cleanliness & Noise Penalty (max 25 pts)
          5. Suspicious Anomaly Deduction (-15 to -30 pts)
        """
        if not text or not text.strip():
            return {
                "quality_score": 0,
                "tier": "Failed",
                "is_suspicious": True,
                "suspicious_patterns": ["Empty or whitespace-only document"],
            }

        stats = cls.calculate_text_stats(text)
        metrics = cls.calculate_quality_metrics(text)
        is_suspicious, patterns = cls.detect_suspicious_patterns(text)

        words = stats["word_count"]
        chars = stats["char_count"]

        # 1. Length & Completeness (25 points)
        if words >= 120:
            score_length = 25.0
        elif words >= 50:
            score_length = 15.0 + (words - 50) / 70.0 * 10.0
        elif words >= 20:
            score_length = 5.0 + (words - 20) / 30.0 * 10.0
        else:
            score_length = max(1.0, words * 0.25)

        # 2. Word Diversity (25 points)
        unique_ratio = metrics["unique_word_ratio"]
        if 0.35 <= unique_ratio <= 0.85:
            score_diversity = 25.0
        elif 0.20 <= unique_ratio < 0.35:
            score_diversity = 15.0 + (unique_ratio - 0.20) / 0.15 * 10.0
        elif unique_ratio > 0.85:
            # Very short text or lists
            score_diversity = 20.0
        else:
            score_diversity = max(2.0, unique_ratio * 50.0)

        # 3. Readability & Avg Word Length (25 points)
        avg_wlen = metrics["avg_word_length"]
        if 4.0 <= avg_wlen <= 8.5:
            score_readability = 25.0
        elif 3.0 <= avg_wlen < 4.0:
            score_readability = 18.0
        elif 8.5 < avg_wlen <= 12.0:
            score_readability = 16.0
        elif avg_wlen < 3.0:
            score_readability = max(5.0, avg_wlen * 3.0)
        else:
            score_readability = 8.0

        # 4. Cleanliness & Noise (25 points)
        noise_ratio = metrics["non_alphanumeric_ratio"]
        if noise_ratio <= 0.08:
            score_clean = 25.0
        elif noise_ratio <= 0.16:
            score_clean = 25.0 - (noise_ratio - 0.08) / 0.08 * 8.0
        elif noise_ratio <= 0.30:
            score_clean = 17.0 - (noise_ratio - 0.16) / 0.14 * 10.0
        else:
            score_clean = max(2.0, 7.0 - (noise_ratio - 0.30) * 20.0)

        raw_score = score_length + score_diversity + score_readability + score_clean

        # 5. Penalties for suspicious OCR artifacts
        if is_suspicious:
            penalty = min(25.0, len(patterns) * 5.0)
            raw_score -= penalty

        # Cap score between 0 and 100
        final_score = int(max(0, min(100, round(raw_score))))

        # Tier classification
        if final_score >= 80:
            tier = "High Quality"
        elif final_score >= 50:
            tier = "Medium Quality"
        else:
            tier = "Low Quality"

        return {
            "quality_score": final_score,
            "tier": tier,
            "is_suspicious": is_suspicious,
            "suspicious_patterns": patterns[:5],
            "stats": stats,
            "metrics": metrics,
        }
