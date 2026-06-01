from __future__ import annotations

from codescope.investigation.llm_context import (
    LLMInvestigationChunkContext,
    LLMInvestigationContext,
)


def build_llm_investigation_prompt(context: LLMInvestigationContext) -> str:
    """Build a deterministic grounded prompt for optional LLM investigation."""

    sections: list[str] = []
    sections.append(_instructions_section())
    sections.append(_query_section(context))
    sections.append(_chunks_section("Likely relevant code", context.likely_relevant_code))
    sections.append(_chunks_section("Related context", context.related_context))
    sections.append(_output_format_section())
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def _instructions_section() -> str:
    return "\n".join(
        [
            "Role:",
            "You are CodeScope's optional LLM investigation layer.",
            "CodeScope has already run deterministic retrieval and ranking.",
            "",
            "Grounding rules:",
            "- Use only the provided CodeScope context.",
            "- Do not invent files, functions, classes, dependencies, or runtime behavior.",
            "- Do not claim certainty; explain uncertainty when evidence is weak.",
            "- Do not generate patches or diffs.",
            "- Do not modify files.",
            "- Suggest only a high-level debugging direction.",
            "- Cite or mention chunk names and paths from the provided context.",
            "- Do not suggest editing unrelated code.",
        ]
    )


def _query_section(context: LLMInvestigationContext) -> str:
    return "\n".join(
        [
            "Investigation query:",
            f"- Repository: {context.repo_path}",
            f"- Bug description: {context.query or '<none>'}",
        ]
    )


def _chunks_section(
    title: str, chunks: tuple[LLMInvestigationChunkContext, ...]
) -> str:
    lines = [f"{title}:"]
    if not chunks:
        lines.append("- <none>")
        return "\n".join(lines)

    for chunk in chunks:
        lines.extend(_chunk_lines(chunk))
    return "\n".join(lines)


def _chunk_lines(chunk: LLMInvestigationChunkContext) -> list[str]:
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
            "Do not include a title or repeat the `LLM Investigation` heading.",
            "Start directly with these bullet sections:",
            "- Likely relevant area:",
            "- Inspect first:",
            "- Why these chunks matter:",
            "- Possible debugging direction:",
            "- Uncertainty:",
        ]
    )
