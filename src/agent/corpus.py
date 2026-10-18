"""Chunk the GDPR corpus.

The GDPR's natural units are the articles (99) and the recitals (173). Each is
one chunk; very long articles are split into paragraph groups so no chunk is
much longer than MAX_CHARS. Every chunk carries its kind and number in the
metadata, which is what the assistant cites.
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

# A line that is exactly a header: "(17)" or "Article 32".
_HEADER_RE = re.compile(r"^\s*(?:\(\d+\)|Article \d+)\s*$", re.MULTILINE)

# Where the enacting terms end and the signatures and footnotes begin.
_END_MARKER = "This Regulation shall be binding in its entirety"


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


def chunk_gdpr(text: str) -> list[Document]:
    """Split the GDPR text into recital and article chunks."""
    end = text.find(_END_MARKER)
    if end != -1:
        text = text[:end]

    parts = _HEADER_RE.split(text)
    headers = _HEADER_RE.findall(text)
    chunks: list[Document] = []

    for header, body in zip(headers, parts[1:], strict=False):
        header = header.strip()
        body = body.strip()
        if not body:
            continue

        if header.startswith("Article"):
            number = int(header.split()[1])
            if len(body) > MAX_CHARS:
                for i, piece in enumerate(_paragraph_chunks(body), start=1):
                    chunks.append(
                        Document(
                            page_content=piece,
                            metadata={"kind": "article", "number": number, "part": i},
                        )
                    )
            else:
                chunks.append(
                    Document(
                        page_content=body,
                        metadata={"kind": "article", "number": number},
                    )
                )
        else:
            number = int(header.strip("()"))
            chunks.append(
                Document(
                    page_content=body,
                    metadata={"kind": "recital", "number": number},
                )
            )

    return chunks


def load_gdpr_chunks() -> list[Document]:
    """Read the corpus file and chunk it."""
    return chunk_gdpr(GDPR_TEXT.read_text())
