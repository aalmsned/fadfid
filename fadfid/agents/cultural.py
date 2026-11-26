"""
Cultural-adaptation agent.

Adjusts the listener's draft with faith or family framing ONLY when the
tracker shows the user turned on the setting or raised the theme first (P2).
It may mirror what the user raised but never introduces it. When not
permitted, the agent is disabled and returns the draft unchanged.
"""

from .tracker import FAITH_WORDS, FAMILY_WORDS, containsAny
from ..textclean import cleanReply

CULTURAL_PROMPT = """You improve one reply from a wellbeing support chatbot so it gently mirrors themes the user raised.
Allowed themes: {allowed}. Only use an allowed theme if the user's own words mention it.
Never add any theme that is not allowed. Never preach, never tell the user what they should
believe or do religiously, never pressure them about family. Keep the same language, dialect and length.
Output ONLY the final reply the user will read: no labels like "Draft:" or "Revised:", no notes,
no explanations, no quotation marks."""

THEME_WORDS = {"faith": FAITH_WORDS, "family": FAMILY_WORDS}


class CulturalAgent:
    name = "cultural"

    def __init__(self, llm):
        self.llm = llm

    def run(self, draft: str, userText: str, permissions: dict) -> tuple[str, bool]:
        # Rewrite only when this message actually raises an allowed theme; otherwise there is nothing to mirror.
        allowed = [theme for theme, ok in permissions.items() if ok and containsAny(userText, THEME_WORDS.get(theme, []))]
        if not allowed:
            return draft, False
        prompt = CULTURAL_PROMPT.format(allowed=", ".join(allowed))
        task = f"USER MESSAGE: {userText}\n\nREPLY TO IMPROVE: {draft}\n\nWrite the improved reply only."
        rewritten = cleanReply(self.llm.generate(prompt, [{"role": "user", "content": task}]))
        return (rewritten or draft), True
