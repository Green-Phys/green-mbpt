#!/usr/bin/env python3
"""Validate the repository's downstream GREEN usage skill."""

from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/green-mbpt-run"
REFERENCES = (
    "workflow.md",
    "build-and-releases.md",
    "data-contracts.md",
    "pitfalls.md",
    "citations.md",
)
REQUIRED_URLS = (
    "https://green-phys.org/llms.txt",
    "https://green-phys.org/docs/getting-started/preparing_input/",
    "https://green-phys.org/docs/getting-started/running_greens/",
    "https://green-phys.org/docs/components/data-formats/input-file/",
    "https://green-phys.org/docs/components/data-formats/output-files/",
    "https://green-phys.org/docs/installation/from_sources/",
    "https://green-phys.org/docs/installation/seet_solver/",
    "https://github.com/Green-Phys/green-citations",
    "https://github.com/Green-Phys/green-citations/blob/main/CITATION.md",
    "https://github.com/Green-Phys/green-citations/blob/main/CITATION.cff",
    "https://github.com/Green-Phys/green-citations/blob/main/references.bib",
    "https://green-phys.org/docs/publications/",
    "https://green-phys.org/docs/acknowledgment/",
)


def main() -> int:
    errors: list[str] = []
    entry = SKILL / "SKILL.md"
    if not entry.is_file():
        errors.append("missing .agents/skills/green-mbpt-run/SKILL.md")
        text = ""
    else:
        text = entry.read_text(encoding="utf-8")
        if not re.match(
            r"^---\nname: green-mbpt-run\ndescription: .+\n---\n", text
        ):
            errors.append("SKILL.md has invalid or incomplete frontmatter")

    combined = text
    for name in REFERENCES:
        path = SKILL / "references" / name
        if not path.is_file():
            errors.append(f"missing skill reference: references/{name}")
            continue
        if f"references/{name}" not in text:
            errors.append(f"SKILL.md does not route to references/{name}")
        combined += "\n" + path.read_text(encoding="utf-8")

    for url in REQUIRED_URLS:
        if url not in combined:
            errors.append(f"skill does not link required canonical page: {url}")

    if errors:
        print("agent documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("agent-docs-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

