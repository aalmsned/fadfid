"""
Conversation tracker.

Holds session-only state in memory: whether the user raised faith or family,
the cultural opt-in setting, distress building across messages, age cues and
the consent step. Nothing here is written to disk; a session is deleted when it
ends or after the idle timeout (P3).
"""

import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

FAITH_WORDS = [
    "الله", "ربي", "صلاة", "صلاتي", "اصلي", "أصلي", "القرآن", "قرآن", "دعاء", "ادعي", "أدعي", "إيمان", "ايمان",
    "الدين", "ديني", "مسجد", "استغفر", "god", "allah", "pray", "prayer", "quran", "faith", "dua", "mosque", "religion",
]
FAMILY_WORDS = [
    "امي", "أمي", "يمه", "ابوي", "أبوي", "يبه", "اهلي", "أهلي", "عائلتي", "عيالي", "زوجي", "زوجتي", "اخوي", "أخوي",
    "اختي", "أختي", "family", "mom", "mother", "dad", "father", "parents", "husband", "wife", "brother", "sister",
]
AGE_CUE_PATTERNS = [
    r"\b(1[0-7])\s*(سنة|سنين|سنه|years?\s*old|yo|y/o)\b",
    r"عمري\s*(1[0-7])\b",
    r"\bi'?m\s*(1[0-7])\b",
    r"(صف|الصف)\s*(السابع|الثامن|التاسع|العاشر|الحادي عشر|الثاني عشر|سابع|ثامن|تاسع|عاشر)",
    r"\bgrade\s*(7|8|9|10|11|12)\b",
    r"(مدرستي|المدرسة الثانوية|ثانوية|متوسطة|high school|middle school|my teacher|معلمتي|أستاذي)",
]


def containsAny(text: str, words: list[str]) -> bool:
    """Whole-word match. English words must end at a word boundary (plural -s allowed), so "mom"
    does not match "moment"; Arabic words may carry suffixes (أمي، أمها)."""
    lowered = text.lower()
    for w in words:
        end = r"(s|es|'s)?(?![a-z])" if re.fullmatch(r"[a-z' ]+", w) else ""
        if re.search(rf"(^|\W){re.escape(w)}{end}", lowered):
            return True
    return False


@dataclass
class SessionState:
    sessionId: str
    culturalOptIn: bool = False
    faithRaised: bool = False
    familyRaised: bool = False
    ageCue: bool = False
    minorNoticeShown: bool = False
    consentState: str = "none"  # none | offered | accepted | declined
    consentKind: str = "self"   # self | harm: which kind of counsellor offer is open
    lastHarm: str = "none"      # most serious harm-to-others label so far (P8)
    harmTurns: int = 0
    turnsSinceHarm: int = 99    # user turns since serious harm to others was last raised
    crisisTurns: int = 0
    lastCrisisTier: int = 0
    lastReplyLang: str | None = None
    lastVariety: str | None = None
    turnsSinceDecline: int = 0
    tierHistory: deque = field(default_factory=lambda: deque(maxlen=20))
    history: list = field(default_factory=list)  # in-memory chat context only, never persisted
    lastActive: float = field(default_factory=time.time)


class ConversationTracker:
    name = "tracker"

    def __init__(self, timeoutMin: int = 30, trendWindow: int = 5, enabled: bool = True):
        self.timeoutSec = timeoutMin * 60
        self.trendWindow = trendWindow
        self.enabled = enabled
        self.sessions: dict[str, SessionState] = {}

    def newSession(self, culturalOptIn: bool = False) -> str:
        sessionId = uuid.uuid4().hex
        self.sessions[sessionId] = SessionState(sessionId, culturalOptIn=culturalOptIn)
        return sessionId

    def get(self, sessionId: str) -> SessionState:
        self.expireIdle()
        state = self.sessions.get(sessionId)
        if state is None:
            state = SessionState(sessionId)
            self.sessions[sessionId] = state
        state.lastActive = time.time()
        return state

    def expireIdle(self):
        now = time.time()
        for sid in [s for s, st in self.sessions.items() if now - st.lastActive > self.timeoutSec]:
            del self.sessions[sid]

    def endSession(self, sessionId: str):
        self.sessions.pop(sessionId, None)

    def observeUser(self, sessionId: str, text: str):
        state = self.get(sessionId)
        if containsAny(text, FAITH_WORDS):
            state.faithRaised = True
        if containsAny(text, FAMILY_WORDS):
            state.familyRaised = True
        if any(re.search(p, text, re.IGNORECASE) for p in AGE_CUE_PATTERNS):
            state.ageCue = True

    def recordTier(self, sessionId: str, tier: int):
        self.get(sessionId).tierHistory.append(tier)

    def distressRising(self, sessionId: str) -> bool:
        """True when distress has built across recent messages (slow-build crisis signal)."""
        if not self.enabled:
            return False
        recent = list(self.get(sessionId).tierHistory)[-self.trendWindow:]
        if len(recent) < 3:
            return False
        lastThree = recent[-3:]
        return all(t >= 1 for t in lastThree) and lastThree == sorted(lastThree)

    def recentMaxTier(self, sessionId: str) -> int:
        if not self.enabled:
            return 0
        recent = list(self.get(sessionId).tierHistory)[-self.trendWindow:]
        return max(recent, default=0)

    def culturalPermissions(self, sessionId: str) -> dict:
        state = self.get(sessionId)
        if not self.enabled:
            return {"faith": state.culturalOptIn, "family": state.culturalOptIn}
        return {
            "faith": state.culturalOptIn or state.faithRaised,
            "family": state.culturalOptIn or state.familyRaised,
        }
