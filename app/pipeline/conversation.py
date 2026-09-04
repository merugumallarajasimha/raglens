"""Conversation handler — manages multi-turn conversation context.

Supports follow-up questions by maintaining conversation history and
creating standalone queries from conversational context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("raglens.pipeline.conversation")


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    query: str
    answer: Optional[str] = None
    citations: list[dict] = field(default_factory=list)


class ConversationManager:
    """Manages conversation history and query rewriting for follow-ups.

    For follow-up questions, the manager detects contextual references
    (pronouns like "it", "they") and rewrites the query to be standalone.
    """

    def __init__(self, max_history: int = 10) -> None:
        self._history: list[ConversationTurn] = []
        self._max_history = max_history

    def add_turn(
        self,
        query: str,
        answer: Optional[str] = None,
        citations: Optional[list[dict]] = None,
    ) -> None:
        """Add a user-assistant turn to the conversation history."""
        if citations is None:
            citations = []

        # Mark as user turn
        user_turn = ConversationTurn(
            role="user",
            query=query,
        )

        # Mark as assistant turn if answer provided
        turns = [user_turn]
        if answer:
            turns.append(ConversationTurn(
                role="assistant",
                query=answer,
                answer=answer,
                citations=citations,
            ))

        for turn in turns:
            self._history.append(turn)

        # Trim history
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2:]

    def get_standalone_query(self, query: str) -> tuple[str, bool]:
        """Rewrite a follow-up query to be standalone.

        If the query contains contextual references (pronouns), it is rewritten
        using the conversation history. Otherwise, the original query is returned.

        Args:
            query: The user's follow-up query.

        Returns:
            Tuple of (standalone_query, was_rewritten).
        """
        if not self._history:
            return query, False

        # Check if query contains pronouns that need context
        import re
        pronouns = re.findall(
            r"\b(it|this|that|they|them|this paper|these papers|that paper|those papers)\b",
            query,
            re.IGNORECASE,
        )

        if not pronouns:
            return query, False

        # Get the last relevant topic from history
        last_topics = []
        for turn in reversed(self._history):
            if turn.role == "user":
                last_topics.append(turn.query)
                if len(last_topics) >= 2:
                    break

        if last_topics:
            context = "; ".join(last_topics)
            standalone = f"{query} (context: {context})"
            return standalone, True

        return query, False

    def get_history(self) -> list[dict]:
        """Return conversation history as a list of dicts."""
        return [
            {"role": t.role, "query": t.query}
            for t in self._history
        ]

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()
