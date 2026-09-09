"""
Document Sampling Module for Manual OCR Quality Verification.
Produces randomized and stratified samples for visual inspection.
"""

import random
from typing import Any, Dict, List, Optional


class QualitySampler:
    """
    Selects representative sample documents for human quality inspection.
    """

    @classmethod
    def sample_documents(
        cls,
        audit_results: List[Dict[str, Any]],
        sample_size: int = 10,
        seed: Optional[int] = 42,
    ) -> List[Dict[str, Any]]:
        """
        Samples documents from audit results and formats output matching schema:
        {
          "document_id": "",
          "title": "",
          "word_count": 0,
          "quality_score": 0,
          "first_500_characters": ""
        }
        """
        if not audit_results:
            return []

        if seed is not None:
            rng = random.Random(seed)
            shuffled = list(audit_results)
            rng.shuffle(shuffled)
        else:
            shuffled = random.sample(audit_results, len(audit_results))

        # Stratify if possible: pick from each available tier
        tiers = {"High Quality": [], "Medium Quality": [], "Low Quality": [], "Failed": []}
        for item in shuffled:
            tier = item.get("tier", "Medium Quality")
            if tier in tiers:
                tiers[tier].append(item)
            else:
                tiers.setdefault(tier, []).append(item)

        selected = []
        # Ensure at least 1 from lower tiers if present
        for tier in ["Low Quality", "Medium Quality", "High Quality"]:
            if tiers[tier] and len(selected) < sample_size:
                selected.append(tiers[tier].pop(0))

        # Fill remainder
        remaining = [item for sublist in tiers.values() for item in sublist]
        while len(selected) < sample_size and remaining:
            selected.append(remaining.pop(0))

        samples = []
        for doc in selected:
            text = doc.get("text", "")
            first_500 = text[:500] if len(text) > 500 else text

            samples.append(
                {
                    "document_id": doc.get("document_id", ""),
                    "title": doc.get("title", ""),
                    "word_count": doc.get("stats", {}).get("word_count", 0),
                    "quality_score": doc.get("quality_score", 0),
                    "first_500_characters": first_500,
                }
            )

        return samples
