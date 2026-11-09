"""Tell small talk apart from questions about the GDPR, with Jev (TypeSafe).

Jev answers one yes/no question about the user's last message: "Is this small
talk that needs nothing from the regulation?" Its answer is a calibrated
probability, so the graph can skip retrieval only when Jev is confident. A
real question wrongly taken for small talk would be answered without the
regulation's text, which is worse than a greeting going through retrieval, so
anything below `MIN_SMALL_TALK`, and any failed call, takes the retrieval path.
"""

from __future__ import annotations

import asyncio
from functools import cache

from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

JEV_MODEL = "jev-latest"
JEV_TIMEOUT = 5.0
MIN_SMALL_TALK = 0.9

SMALL_TALK_QUESTION = Noul(
    instructions=(
        "Is the user's last message small talk that needs no information from "
        "the GDPR, such as a greeting, thanks, a goodbye, or a question about "
        "what this assistant can do? Answer no if the message asks anything "
        "about data protection, privacy or the regulation, or follows up on an "
        "earlier question, even when it also contains a greeting."
    ),
    criteria={
        "true": "The message is only small talk and needs nothing from the GDPR.",
        "false": "The message asks for, or follows up on, information about "
        "data protection or the GDPR.",
    },
)


def _state_text(message: str, history: str) -> str:
    """Format the Jev input: the conversation so far, then the last message."""
    context = f"Conversation so far:\n{history}\n\n" if history else ""
    return f"{context}User's last message: {message}"


@cache
def _client(loop: asyncio.AbstractEventLoop) -> AsyncTypeSafeClient:
    # One client per event loop, as in reranking: its connection pool belongs
    # to the loop that created it.
    return AsyncTypeSafeClient(
        model=JEV_MODEL, retry=RetryPolicy(max_retries=0), timeout=JEV_TIMEOUT
    )


async def small_talk_probability(message: str, history: str) -> float | None:
    """Ask Jev how likely the message is small talk; None if the call fails."""
    client = _client(asyncio.get_running_loop())
    try:
        async with asyncio.timeout(JEV_TIMEOUT):
            response = await client.system_one(
                state=_state_text(message, history),
                questions={"small_talk": SMALL_TALK_QUESTION},
            )
    except Exception:
        return None
    return response.nouls["small_talk"].noul


def is_small_talk(probability: float | None) -> bool:
    """Skip retrieval only when Jev answered and is confident."""
    return probability is not None and probability >= MIN_SMALL_TALK
