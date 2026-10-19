"""Chunk the GDPR corpus.

The GDPR's natural units are the articles (99) and the recitals (173). Each is
one chunk, with two exceptions: an article made of numbered points (Article 4,
the definitions) gets one chunk per point, and other very long articles are
split into paragraph groups so no chunk is much longer than MAX_CHARS. Every
chunk carries its kind and number in the metadata, which is what the
assistant cites.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"
GDPR_TEXT = CORPUS_DIR / "gdpr.txt"

MAX_CHARS = 1600
SPLIT_CHARS = 1200
OVERLAP_CHARS = 200

# Recital headers and Article 4's definition points look the same: a line that
# is exactly "(17)". Only the position tells them apart, so each pattern is
# searched only in its own part of the text.
_NUMBER_RE = re.compile(r"^\((\d+)\)$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^Article (\d+)$", re.MULTILINE)

# "CHAPTER II" or "Section 1" followed by its title line; without stripping,
# the heading sticks to the end of the article before it.
_HEADING_RE = re.compile(r"^(?:CHAPTER [IVX]+|Section \d+)\n\n[^\n]+$", re.MULTILINE)

# The recitals end and the articles begin here.
_ENACTING_MARKER = "HAVE ADOPTED THIS REGULATION:"

# Where the enacting terms end and the signatures and footnotes begin.
_END_MARKER = "This Regulation shall be binding in its entirety"


def _normalize(text: str) -> str:
    """Blank out whitespace-only lines and collapse runs of blank lines."""
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text)


def _sections(text: str, header: re.Pattern[str]) -> list[tuple[int, str]]:
    """Split text at each header match into (number, body) pairs."""
    matches = list(header.finditer(text))
    ends = [m.start() for m in matches[1:]] + [len(text)]
    sections: list[tuple[int, str]] = []
    for match, end in zip(matches, ends, strict=False):
        body = text[match.end() : end].strip()
        if body:
            sections.append((int(match.group(1)), body))
    return sections


def _paragraph_chunks(text: str, max_chars: int = SPLIT_CHARS) -> list[str]:
    """Group paragraphs into chunks of at most about max_chars characters."""
    chunks: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def _article_chunks(number: int, body: str) -> list[Document]:
    """Chunk one article: per numbered point, per paragraph group, or whole."""
    points = _sections(body, _NUMBER_RE)
    if points:
        return [
            Document(
                page_content=text,
                metadata={"kind": "article", "number": number, "point": point},
            )
            for point, text in points
        ]
    if len(body) > MAX_CHARS:
        return [
            Document(
                page_content=piece,
                metadata={"kind": "article", "number": number, "part": i},
            )
            for i, piece in enumerate(_paragraph_chunks(body), start=1)
        ]
    return [Document(page_content=body, metadata={"kind": "article", "number": number})]


def chunk_gdpr(text: str) -> list[Document]:
    """Split the GDPR text into recital and article chunks."""
    end = text.find(_END_MARKER)
    if end != -1:
        text = text[:end]

    split = text.find(_ENACTING_MARKER)
    if split == -1:
        raise ValueError(f"GDPR text has no {_ENACTING_MARKER!r} line")
    preamble = _normalize(text[:split])
    enacting = _HEADING_RE.sub("", _normalize(text[split:]))

    chunks = [
        Document(page_content=body, metadata={"kind": "recital", "number": number})
        for number, body in _sections(preamble, _NUMBER_RE)
    ]
    for number, body in _sections(enacting, _ARTICLE_RE):
        chunks.extend(_article_chunks(number, body))
    return chunks


def load_gdpr_chunks() -> list[Document]:
    """Read the corpus file and chunk it."""
    return chunk_gdpr(GDPR_TEXT.read_text())
