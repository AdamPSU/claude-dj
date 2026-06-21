from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap
from typing import Any


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
TOP_MARGIN = 742
LINE_HEIGHT = 14
MAX_CHARS = 86


def write_report_pdf(input_path: Path, output_path: Path) -> None:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    lines = _report_lines(report)
    _write_text_pdf(output_path, lines)


def _report_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = [str(report.get("title") or "ClaudeDJ Sentry Agent Report")]
    authors = report.get("authors") or ["ClaudeDJ Pipeline Agents"]
    lines.append("Authors: " + ", ".join(str(author) for author in authors))
    collaboration_id = report.get("collaboration_id")
    if collaboration_id:
        lines.append(f"Collaboration: {collaboration_id}")
    generated_at = report.get("generated_at")
    if generated_at:
        lines.append(f"Generated: {generated_at}")

    lines.extend(["", "Abstract", str(report.get("abstract") or report.get("summary") or "No abstract supplied.")])
    keywords = report.get("keywords") or []
    if keywords:
        lines.append("Keywords: " + ", ".join(str(keyword) for keyword in keywords))

    lines.extend(["", "1. Introduction", str(report.get("introduction") or report.get("summary") or "")])

    lines.extend(["", "2. Methods"])
    methods = report.get("methods")
    if methods:
        lines.append(str(methods))
    else:
        lines.append(
            "Fifteen named agents evaluated isolated pipeline slices. Each agent used unit or read-only "
            "checks by default, avoided shared live Spotify state, and used Sentry evidence only for "
            "observability or failure analysis."
        )

    verification = report.get("verification") or []
    if verification:
        lines.append("Verification commands:")
        for check in verification:
            command = check.get("command", "")
            result = check.get("result", "")
            workdir = check.get("workdir", "")
            lines.append(f"- {workdir}: {command} -> {result}")

    lines.extend(["", "3. Results"])

    agents = report.get("agents") or []
    if agents:
        lines.append("Agent performance:")
        for agent in agents:
            name = agent.get("name", "Unnamed agent")
            scenario = agent.get("scenario", "unknown scenario")
            status = agent.get("status", "unknown")
            summary = agent.get("summary", "")
            lines.append(f"- {name}: {status} on {scenario}. {summary}")

    lines.extend(["", "4. Sentry Observability Results"])
    sentry = report.get("sentry") or {}
    dashboard_url = sentry.get("dashboard_url")
    sample_trace_id = sentry.get("sample_trace_id")
    queries = sentry.get("queries") or []
    if dashboard_url:
        lines.append(f"Dashboard: {dashboard_url}")
    if sample_trace_id:
        lines.append(f"Sample trace: {sample_trace_id}")
    for query in queries:
        name = query.get("name", "Sentry query")
        summary = query.get("summary", "")
        lines.append(f"- {name}: {summary}")

    lines.extend(["", "5. Findings"])

    findings = report.get("findings") or []
    if findings:
        for finding in findings:
            severity = finding.get("severity", "info")
            component = finding.get("component", "unknown")
            summary = finding.get("summary", "")
            lines.append(f"- {severity} / {component}: {summary}")

    lines.extend(["", "6. Limitations"])
    limitations = report.get("limitations") or [
        "The fifteen agents used isolated checks instead of concurrent live Spotify playback to avoid shared device interference.",
        "Custom Sentry field lookup timed out, so attribution is verified locally and by a sample trace rather than a full field table.",
        "The live E2E path was not re-run during this pass because it requires explicit environment enablement and shared external services.",
    ]
    for limitation in limitations:
        lines.append(f"- {limitation}")

    lines.extend(["", "7. Discussion"])
    discussion = report.get("discussion") or (
        "The highest-value user experience improvements were small alignment and observability changes. "
        "The remaining risks are operational: recommendation latency, local narration interruption, and setup-sensitive webcam assets."
    )
    lines.append(str(discussion))

    lines.extend(["", "8. Conclusion"])
    conclusion = report.get("conclusion") or (
        "ClaudeDJ's current pipeline is stable under isolated verification, and the smallest useful changes improve "
        "the listener-facing DJ behavior without changing architecture."
    )
    lines.append(str(conclusion))

    references = report.get("references") or []
    if references:
        lines.extend(["", "References"])
        for index, reference in enumerate(references, start=1):
            lines.append(f"[{index}] {reference}")

    return lines


def _write_text_pdf(output_path: Path, lines: list[str]) -> None:
    pages = _paginate(lines)
    objects: list[bytes] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    page_refs = " ".join(f"{3 + page_index * 2} 0 R" for page_index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{page_refs}] /Count {len(pages)} >>".encode("ascii"))

    for page_index, page_lines in enumerate(pages):
        page_object_id = 3 + page_index * 2
        content_object_id = page_object_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {content_object_id} 0 R >>".encode("ascii")
        )
        stream = _page_stream(page_lines)
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    output_path.write_bytes(bytes(output))


def _paginate(lines: list[str]) -> list[list[str]]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(wrap(line, width=MAX_CHARS) or [""])
    lines_per_page = int((TOP_MARGIN - 54) / LINE_HEIGHT)
    return [wrapped[index : index + lines_per_page] for index in range(0, len(wrapped), lines_per_page)] or [[]]


def _page_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 11 Tf", "14 TL", f"{LEFT_MARGIN} {TOP_MARGIN} Td"]
    for index, line in enumerate(lines):
        if index:
            parts.append("T*")
        parts.append(f"({_escape_pdf_text(line)}) Tj")
    parts.append("ET")
    return ("\n".join(parts) + "\n").encode("utf-8")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a compact ClaudeDJ Sentry report PDF from JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_report_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
