"""When (not) to offer tools to the local model."""

from __future__ import annotations

import re

# Trivial turns that must never trigger web_search.
_SMALLTALK = re.compile(
    r"^(hi+|hello|hey+|yo|sup|hiya|howdy|thanks|thank you|thx|ok|okay|"
    r"k|cool|nice|bye|goodbye|good morning|good night|gm|gn|"
    r"what's up|whats up|how are you|test|ping)\b[\s!.?]*$",
    re.IGNORECASE,
)

# Explicit research intent — only then do we even mention tools.
_RESEARCH = re.compile(
    r"\b(search|look up|lookup|google|find online|latest|news|"
    r"current|today|who won|price of|weather|what happened)\b",
    re.IGNORECASE,
)


def is_smalltalk(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) <= 24 and _SMALLTALK.match(t):
        return True
    return False


def should_offer_tools(user_text: str, *, tools_enabled: bool, use_search: bool) -> bool:
    """
    Tools only when:
      - the user turned search on for this send, AND
      - settings allow tools, AND
      - the message is not a greeting / smalltalk.
    Even then, the model is told to search only for current/external facts.
    """
    if not tools_enabled or not use_search:
        return False
    if is_smalltalk(user_text):
        return False
    return True


CONSERVATIVE_TOOL_PROMPT = """You are a helpful local assistant.

You have an optional web_search tool. Use it ONLY when the user needs
current or external facts you cannot know (news, live data, a specific page).

Do NOT search for greetings, chit-chat, opinions, coding help, or anything
you can answer from the conversation.

If you need the tool, emit exactly:

<tool_call>
web_search
the search query
</tool_call>

Then stop. Otherwise answer normally in a few sentences. Do not invent search results."""
