"""
Report Generation Module.
Generates structured JSON and responsive, beautifully styled HTML reports
for OCR quality verification and manual audit.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class ReportGenerator:
    """
    Generates data/reports/ocr_report.json and data/reports/ocr_report.html.
    """

    @classmethod
    def generate_summary(cls, audit_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates aggregate statistics across all audited documents.
        """
        total = len(audit_results)
        if total == 0:
            return {
                "total_documents": 0,
                "average_word_count": 0.0,
                "average_character_count": 0.0,
                "average_quality_score": 0.0,
                "high_quality_documents": 0,
                "medium_quality_documents": 0,
                "low_quality_documents": 0,
                "failed_documents": 0,
                "suspicious_documents_count": 0,
            }

        total_words = sum(doc.get("stats", {}).get("word_count", 0) for doc in audit_results)
        total_chars = sum(doc.get("stats", {}).get("char_count", 0) for doc in audit_results)
        total_score = sum(doc.get("quality_score", 0) for doc in audit_results)

        high_count = sum(1 for doc in audit_results if doc.get("tier") == "High Quality")
        med_count = sum(1 for doc in audit_results if doc.get("tier") == "Medium Quality")
        low_count = sum(1 for doc in audit_results if doc.get("tier") == "Low Quality")
        failed_count = sum(1 for doc in audit_results if doc.get("tier") == "Failed")
        suspicious_count = sum(1 for doc in audit_results if doc.get("is_suspicious", False))

        return {
            "total_documents": total,
            "average_word_count": round(total_words / total, 2),
            "average_character_count": round(total_chars / total, 2),
            "average_quality_score": round(total_score / total, 2),
            "high_quality_documents": high_count,
            "medium_quality_documents": med_count,
            "low_quality_documents": low_count,
            "failed_documents": failed_count,
            "suspicious_documents_count": suspicious_count,
        }

    @classmethod
    def save_json_report(
        cls,
        summary: Dict[str, Any],
        samples: List[Dict[str, Any]],
        all_scores: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """
        Writes data/reports/ocr_report.json.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "summary": summary,
            "sample_verification": samples,
            "document_scores": [
                {
                    "document_id": doc["document_id"],
                    "title": doc.get("title", ""),
                    "quality_score": doc.get("quality_score", 0),
                    "tier": doc.get("tier", "Medium Quality"),
                    "word_count": doc.get("stats", {}).get("word_count", 0),
                    "char_count": doc.get("stats", {}).get("char_count", 0),
                    "is_suspicious": doc.get("is_suspicious", False),
                }
                for doc in all_scores
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def save_html_report(
        cls,
        summary: Dict[str, Any],
        samples: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """
        Writes responsive, modern data/reports/ocr_report.html.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples_html = ""
        for idx, s in enumerate(samples, 1):
            badge_color = (
                "#22c55e" if s["quality_score"] >= 80 else ("#eab308" if s["quality_score"] >= 50 else "#ef4444")
            )
            preview_escaped = (
                s["first_500_characters"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            samples_html += f"""
            <div class="sample-card">
                <div class="sample-header">
                    <span class="sample-index">#{idx}</span>
                    <span class="sample-title">{s['title']}</span>
                    <span class="badge" style="background-color: {badge_color};">Score: {s['quality_score']}</span>
                </div>
                <div class="sample-meta">
                    <span><strong>Doc ID:</strong> <code>{s['document_id']}</code></span>
                    <span><strong>Words:</strong> {s['word_count']}</span>
                </div>
                <div class="sample-preview">
                    <pre>{preview_escaped}</pre>
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NITA NoticeRAG - OCR Quality Verification Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent: #38bdf8;
            --border: #334155;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
            color: #ffffff;
        }}
        .subtitle {{
            color: var(--text-sub);
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.2rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent);
            margin-top: 0.4rem;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .section-title {{
            font-size: 1.3rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
        }}
        .sample-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .sample-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        .sample-index {{
            font-weight: bold;
            color: var(--accent);
        }}
        .sample-title {{
            font-weight: 600;
            flex-grow: 1;
        }}
        .badge {{
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            color: #000;
        }}
        .sample-meta {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-top: 0.5rem;
            margin-bottom: 0.6rem;
        }}
        .sample-preview pre {{
            background: #090d16;
            border-radius: 4px;
            padding: 0.8rem;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #cbd5e1;
            white-space: pre-wrap;
            word-break: break-word;
            margin: 0;
            max-height: 160px;
            overflow-y: auto;
        }}
        code {{
            color: #38bdf8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NITA Campus Intelligence - OCR Quality Verification Report</h1>
        <div class="subtitle">Automated dataset health audit and quality validation for RAG ingestion</div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Documents</div>
                <div class="metric-value">{summary['total_documents']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Quality Score</div>
                <div class="metric-value">{summary['average_quality_score']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Word Count</div>
                <div class="metric-value">{summary['average_word_count']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">High Quality (&ge;80)</div>
                <div class="metric-value" style="color: var(--success);">{summary['high_quality_documents']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Medium Quality</div>
                <div class="metric-value" style="color: var(--warning);">{summary['medium_quality_documents']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Low Quality / Failed</div>
                <div class="metric-value" style="color: var(--danger);">{summary['low_quality_documents'] + summary['failed_documents']}</div>
            </div>
        </div>

        <div class="section-title">Verified Inspection Samples (10 Documents)</div>
        <div class="samples-list">
            {samples_html}
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
