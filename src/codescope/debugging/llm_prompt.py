from __future__ import annotations

from codescope.debugging.llm_context import LLMChunkContext, LLMDiagnosisContext


def build_llm_diagnosis_prompt(context: LLMDiagnosisContext) -> str:
    """Build a deterministic grounded prompt for optional LLM diagnosis."""

    sections: list[str] = []
    sections.append(_instructions_section())
    sections.append(_failure_section(context))
    sections.append(_summary_section(context))
    sections.append(_chunks_section(context.chunks))
    sections.append(_output_format_section())
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def _instructions_section() -> str:
    return "\n".join(
        [
            "Role:",
            "You are CodeScope's optional LLM diagnosis layer.",
            "CodeScope has already run deterministic retrieval and ranking.",
            "",
            "Grounding rules:",
            "- Use only the provided CodeScope context.",
            "- Do not invent files, functions, classes, tests, dependencies, or runtime behavior.",
            "- Do not claim certainty; explain uncertainty when evidence is weak.",
            "- Do not generate patches or diffs.",
            "- Do not modify files.",
            "- Suggest only a high-level fix direction.",
            "- Cite or mention chunk names and paths from the provided context.",
            "- Do not suggest editing unrelated code.",
        ]
    )


def _failure_section(context: LLMDiagnosisContext) -> str:
    failure = context.failure
    location = failure.file_path
    if failure.line_number is not None:
        location = f"{location}:{failure.line_number}"

    lines = [
        "Failure:",
        f"- Test: {failure.test_name}",
        f"- Location: {location}",
        f"- Error: {failure.error_type or '<unknown>'}",
        f"- Message: {failure.message or '<none>'}",
    ]
    if failure.traceback_excerpt:
        lines.extend(
            [
                "- Traceback excerpt:",
                _indent_block(failure.traceback_excerpt),
            ]
        )
    return "\n".join(lines)


def _summary_section(context: LLMDiagnosisContext) -> str:
    lines = [
        "Deterministic CodeScope diagnosis summary:",
        _indent_block(context.diagnosis_summary),
    ]
    if context.possible_issue:
        lines.extend(
            [
                "",
                "Rule-based possible issue:",
                _indent_block(context.possible_issue),
            ]
        )
    return "\n".join(lines)


def _chunks_section(chunks: tuple[LLMChunkContext, ...]) -> str:
    lines = ["Retrieved chunks:"]
    if not chunks:
        lines.append("- <none>")
        return "\n".join(lines)

    for chunk in chunks:
        lines.extend(_chunk_lines(chunk))
    return "\n".join(lines)


def _chunk_lines(chunk: LLMChunkContext) -> list[str]:
    location = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
    lines = [
        f"{chunk.rank}. {chunk.name}",
        f"   - Path: {location}",
        f"   - Kind: {chunk.chunk_type}",
        f"   - Source: {chunk.source_label}",
    ]
    if chunk.score is not None:
        lines.append(f"   - Score: {chunk.score:.2f}")

    lines.append("   - Reasons:")
    if chunk.reasons:
        lines.extend(f"     - {reason}" for reason in chunk.reasons)
    else:
        lines.append("     - <none>")

    lines.append("   - Dependencies:")
    if chunk.dependencies:
        lines.extend(f"     - {dependency}" for dependency in chunk.dependencies)
    else:
        lines.append("     - <none>")

    lines.extend(
        [
            "   - Code:",
            "```python",
            chunk.code_snippet,
            "```",
        ]
    )
    return lines


def _output_format_section() -> str:
    return "\n".join(
        [
            "Write the answer in this exact structure:",
            "",
            "Do not include a title or repeat the `LLM Diagnosis` heading.",
            "Start directly with these bullet sections:",
            "- Likely root cause:",
            "- Inspect first:",
            "- Why these chunks matter:",
            "- Possible fix direction:",
            "- Uncertainty:",
        ]
    )


def _indent_block(value: str) -> str:
    if not value:
        return "  <none>"
    return "\n".join(f"  {line}" if line else "" for line in value.splitlines())
