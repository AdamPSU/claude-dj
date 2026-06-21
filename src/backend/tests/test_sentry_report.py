import json
import tempfile
import unittest
from pathlib import Path

from claude_dj.reporting.sentry_report import write_report_pdf


class SentryReportTests(unittest.TestCase):
    def test_write_report_pdf_includes_agent_performance_and_findings(self) -> None:
        report = {
            "title": "A Multi-Agent Sentry Evaluation of ClaudeDJ",
            "authors": ["ClaudeDJ Pipeline Agents"],
            "collaboration_id": "tiny-ux-pass",
            "abstract": "We evaluated ClaudeDJ with isolated agents and Sentry telemetry.",
            "keywords": ["ClaudeDJ", "Sentry", "agent evaluation"],
            "introduction": "The study measures whether tiny UX fixes improve the DJ flow.",
            "agents": [
                {
                    "name": "Beat Beacon",
                    "scenario": "startup_seed",
                    "status": "passed",
                    "summary": "Startup prompt kept the queue small and started playback.",
                }
            ],
            "findings": [
                {
                    "severity": "medium",
                    "component": "narration",
                    "summary": "Interrupted afplay can surface as a demo-visible narration failure.",
                }
            ],
            "sentry": {
                "dashboard_url": "https://pennsylvania-state-universi-og.sentry.io/dashboard/7339119/",
                "queries": [
                    {
                        "name": "Backend errors",
                        "summary": "Sentry showed no new backend errors after the fix window.",
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "report.json"
            output_path = Path(tmp) / "report.pdf"
            input_path.write_text(json.dumps(report), encoding="utf-8")

            write_report_pdf(input_path, output_path)

            content = output_path.read_bytes()

        self.assertTrue(content.startswith(b"%PDF-1.4"))
        self.assertIn(b"A Multi-Agent Sentry Evaluation", content)
        self.assertIn(b"Abstract", content)
        self.assertIn(b"1. Introduction", content)
        self.assertIn(b"2. Methods", content)
        self.assertIn(b"3. Results", content)
        self.assertIn(b"6. Limitations", content)
        self.assertIn(b"Beat Beacon", content)
        self.assertIn(b"Interrupted afplay", content)


if __name__ == "__main__":
    unittest.main()
