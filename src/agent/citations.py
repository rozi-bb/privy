"""Check that an answer cites only the passages it was given.

PRD §10: every article number mentioned must appear in a retrieved passage.
The answer's citations ("Article 17", "Articles 15 and 16", "Recital 26",
"Article 4(1)") are parsed and compared with what the passages contain:
their own labels, plus the articles their text refers to. Article 83, for
example, lists the infringements "referred to in Articles 25 to 39", so an
answer that repeats those numbers is quoting the passage, not inventing.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from langchain_core.documents import Document

# Article 4 is chunked per definition, so "Article 4(1)" must match the
# exact point. For every other article "(1)" is a paragraph inside one
# chunk, so only the article number is checked.
DEFINITIONS_ARTICLE = 4

# References to other legal acts ("Article 8(1) of the Charter") are not
# GDPR citations. The number list is an atomic group so the regex cannot
# backtrack to "Article 8" and slip past the "of the Charter" check.
_REFERENCE_RE = re.compile(
    r"\b(Article|Recital)s?\s+"
    r"((?>\d+(?:\(\d+\))?(?:\([a-z]\))?"
    r"(?:\s*(?:,|and|or|to|–|-)\s*\d+(?:\(\d+\))?(?:\([a-z]\))?)*))"
    r"(?!\s+of\s+(?:the\s+)?(?:Charter|Treaty|Directive|Decision|TFEU|TEU|Council|Regulation\s+\(E[CU]\)))"
)
_ITEM_RE = re.compile(r"(?:(to|–|-)\s*)?(\d+)(?:\((\d+)\))?")
MAX_RANGE = 100


class Citation(NamedTuple):
    """One cited unit: kind ("article" or "recital"), number, and point."""

    kind: str
    number: int
    point: int | None = None

    def __str__(self) -> str:
        """Format as the assistant cites it, e.g. "Article 4(1)"."""
        label = f"{self.kind.title()} {self.number}"
        return f"{label}({self.point})" if self.point else label


def find_citations(text: str) -> set[Citation]:
    """Return every GDPR article or recital the text cites."""
    citations: set[Citation] = set()
    for match in _REFERENCE_RE.finditer(text):
        kind = match.group(1).lower()
        previous: int | None = None
        for range_mark, number_text, point in _ITEM_RE.findall(match.group(2)):
            number = int(number_text)
            # "Articles 25 to 39" mentions every article in between.
            if (
                range_mark
                and previous is not None
                and 0 < number - previous <= MAX_RANGE
            ):
                citations.update(Citation(kind, n) for n in range(previous + 1, number))
            keep_point = kind == "article" and number == DEFINITIONS_ARTICLE
            citations.add(
                Citation(kind, number, int(point) if keep_point and point else None)
            )
            previous = number
    return citations


def _passage_labels(documents: list[Document]) -> set[Citation]:
    allowed: set[Citation] = set()
    for doc in documents:
        meta = doc.metadata or {}
        if meta.get("kind") not in ("article", "recital"):
            continue
        allowed.add(Citation(meta["kind"], meta["number"]))
        if meta.get("point"):
            allowed.add(Citation(meta["kind"], meta["number"], meta["point"]))
    return allowed


def invalid_citations(text: str, documents: list[Document]) -> set[Citation]:
    """Return the citations in the text that no given passage supports.

    A citation is supported by a passage's label or by a reference in its
    text. A bare "Article 4" is supported by any of its definitions;
    "Article 4(1)" only by definition 1.
    """
    mentioned = set().union(*(find_citations(doc.page_content) for doc in documents))
    return find_citations(text) - _passage_labels(documents) - mentioned


def allowed_labels(documents: list[Document]) -> list[str]:
    """List the citations the passages support, with Article 4 by definition."""
    allowed = _passage_labels(documents)
    specific = {c for c in allowed if c.point} | {
        c for c in allowed if not any(o.point and o[:2] == c[:2] for o in allowed)
    }
    return sorted(
        (str(c) for c in specific), key=lambda label: (label.split()[0], label)
    )
