"""Renders a ``Delta`` into JSON (retrieval chunks for grounded chat) and a Markdown summary.

The JSON shape is a contract with retrieval (it treats the delta report as one chunk per change,
each carrying page/sheet/bbox for citation) and markup (it consumes ``location.bbox`` + ``type``).
Keep the ``delta:CHG-000N`` key stable — downstream code indexes on it.
"""

from __future__ import annotations

import html as _html

from ..canonical.model import Delta


def render_report(delta: Delta) -> tuple[str, dict]:
    """Return ``(markdown, json_dict)``. Markdown counts-by-type always match the JSON counts since
    both are derived from the same ``delta.counts``."""
    entries: dict[str, dict] = {}
    for ch in delta.changes:
        entries[f"delta:{ch.id}"] = {
            "type": ch.type,
            "item_type": ch.item_type,
            "location": {"page": ch.page, "sheet": ch.sheet, "bbox": list(ch.bbox.as_tuple())},
            "description": ch.description,
            "old": ch.old,
            "new": ch.new,
            "match_conf": ch.match_conf,
            "class_conf": ch.class_conf,
            "change_set_id": ch.change_set_id,
        }

    counts = dict(delta.counts)
    json_dict = {
        "pid_a": delta.pid_a,
        "pid_b": delta.pid_b,
        "dataset_version": delta.dataset_version,
        "counts": counts,
        "changes": entries,
    }
    markdown = _render_markdown(delta, counts)
    return markdown, json_dict


def _render_markdown(delta: Delta, counts: dict) -> str:
    lines = [f"# Delta: {delta.pid_a} -> {delta.pid_b}", "", "## Summary"]
    for change_type in sorted(counts):
        lines.append(f"- **{change_type}**: {counts[change_type]}")
    lines.append("")
    lines.append("## Changes")
    for ch in delta.changes:
        loc = f"p{ch.page}" + (f" sheet {ch.sheet}" if ch.sheet else "")
        lines.append(
            f"- `{ch.id}` [{ch.type}/{ch.item_type}] {loc} — {_html.escape(ch.description)} "
            f"(old={ch.old!r}, new={ch.new!r}, match_conf={ch.match_conf:.2f}, class_conf={ch.class_conf:.2f})"
        )
    return "\n".join(lines)
