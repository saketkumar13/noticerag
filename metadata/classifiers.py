"""
Document Classification Module.
Classifies NITA university notices into canonical categories using
hierarchical keyword rules, structural indicators, and confidence scoring.
"""

import re
from typing import Any, Dict, List, Tuple


class DocumentClassifier:
    """
    Multi-class classifier for university circulars and notices.
    Supported classes:
      - Holiday Notice
      - Recruitment
      - Tender
      - Scholarship
      - Academic Notice
      - Placement
      - Hostel Notice
      - Conference
      - Event
      - General Notice
      - Unknown
    """

    CATEGORIES = [
        "Holiday Notice",
        "Recruitment",
        "Tender",
        "Scholarship",
        "Academic Notice",
        "Placement",
        "Hostel Notice",
        "Conference",
        "Event",
        "General Notice",
    ]

    # Weighted keywords per category (higher weight = stronger discriminator)
    CATEGORY_SIGNALS = {
        "Holiday Notice": [
            ("declaration of holiday", 5.0),
            ("holiday list", 4.0),
            ("account of holiday", 4.0),
            ("institute shall remain closed", 4.0),
            ("closed on account of", 4.0),
            ("janmashtami", 3.0),
            ("milad-un-nabi", 3.0),
            ("id-ul-fitr", 3.0),
            ("diwali", 3.0),
            ("durga puja", 3.0),
            ("holiday", 2.0),
        ],
        "Tender": [
            ("price bid", 4.5),
            ("financial bid", 4.5),
            ("technical bid", 4.0),
            ("spot quotation", 4.0),
            ("e-tender", 4.0),
            ("notice inviting tender", 4.0),
            ("sealed quotations", 3.5),
            ("tender notice", 3.5),
            ("quotation", 2.5),
            ("bidder", 2.5),
            ("procurement", 2.0),
            ("corrigendum-wifiservice", 3.5),
            ("boom barrier", 3.0),
        ],
        "Hostel Notice": [
            ("hostel admission", 5.0),
            ("hostel affairs", 4.5),
            ("hostel committee", 4.5),
            ("chief warden", 4.0),
            ("warden", 3.5),
            ("hostel fee", 3.5),
            ("room allotment", 3.5),
            ("mess committee", 3.5),
            ("hostel", 2.5),
        ],
        "Recruitment": [
            ("advertisement for", 4.5),
            ("walk-in-interview", 4.5),
            ("jrf", 4.0),
            ("srf", 4.0),
            ("project fellow", 4.0),
            ("recruitment", 3.5),
            ("shortlisted candidates for the post", 4.0),
            ("non-teaching", 3.5),
            ("appointment", 3.0),
            ("vacancy", 3.0),
            ("application for the post", 3.5),
        ],
        "Placement": [
            ("training & placement", 5.0),
            ("t&p cell", 4.5),
            ("campus drive", 4.5),
            ("campus recruitment", 4.0),
            ("internship offer", 4.0),
            ("placement officer", 3.5),
            ("pre-placement", 3.5),
            ("hiring", 2.5),
        ],
        "Scholarship": [
            ("national scholarship", 5.0),
            ("nsp", 4.5),
            ("scholarship scheme", 4.5),
            ("stipend", 3.5),
            ("financial assistance", 3.5),
            ("fellowship", 3.0),
            ("fee waiver", 3.0),
            ("scholarship", 2.5),
        ],
        "Academic Notice": [
            ("academic calendar", 5.0),
            ("provisional admission", 4.5),
            ("admission in ph.d", 4.5),
            ("admission in m.tech", 4.5),
            ("ph.d admission", 4.5),
            ("m.tech admission", 4.5),
            ("ph.d", 3.5),
            ("m.tech", 3.5),
            ("b.tech", 3.5),
            ("b.sc-b.ed", 4.5),
            ("itep", 4.0),
            ("non-ccmt", 4.0),
            ("non-ccmn", 4.0),
            ("branch change", 4.0),
            ("end semester examination", 4.0),
            ("mid semester examination", 4.0),
            ("course registration", 3.5),
            ("supplementary examination", 3.5),
            ("dean academic", 3.5),
            ("curriculum", 3.0),
            ("grade card", 3.0),
            ("admission", 2.5),
            ("provisional list", 2.5),
            ("academic", 2.5),
        ],
        "Event": [
            ("fit india", 5.0),
            ("international day of yoga", 5.0),
            ("sports day", 4.5),
            ("independence day", 4.5),
            ("republic day", 4.5),
            ("gymkhana", 4.0),
            ("cultural festival", 4.0),
            ("kalantar", 4.0),
            ("celebration of", 3.0),
            ("student activity", 2.5),
        ],
        "Conference": [
            ("international conference", 5.0),
            ("national conference", 4.5),
            ("call for papers", 4.5),
            ("symposium", 4.0),
            ("workshop", 3.0),
            ("seminar", 2.5),
        ],
        "General Notice": [
            ("anti-ragging committee", 4.0),
            ("anti-ragging", 3.5),
            ("swachhata pakhwada", 3.5),
            ("circular", 2.0),
            ("notification", 1.5),
            ("memorandum", 1.5),
            ("competent authority", 1.5),
        ],
    }

    @classmethod
    def classify(
        cls,
        text: str,
        title: str = "",
        filename: str = "",
    ) -> Dict[str, Any]:
        """
        Classifies document text and title.
        Returns:
          {
            "document_type": str,
            "confidence": float
          }
        """
        combined = f"{title}\n{filename}\n{text[:3000]}".lower()

        scores: Dict[str, float] = {cat: 0.0 for cat in cls.CATEGORIES}

        for cat, signals in cls.CATEGORY_SIGNALS.items():
            for phrase, weight in signals:
                if phrase in combined:
                    # Give extra boost if found in title or filename
                    if phrase in title.lower() or phrase in filename.lower():
                        scores[cat] += weight * 2.0
                    else:
                        scores[cat] += weight

        # Find category with highest score
        best_cat, best_score = max(scores.items(), key=lambda item: item[1])

        if best_score == 0:
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
            }

        # Calculate normalized confidence score
        # A score of >= 8 is high confidence (0.90 - 0.99)
        conf = min(0.98, round(0.50 + (best_score / (best_score + 6.0)) * 0.48, 2))

        return {
            "document_type": best_cat,
            "confidence": conf,
        }
