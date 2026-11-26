"""
Crisis support agent.

Takes over at tiers 2-4 instead of ending the conversation (P6). Its replies are
close to scripted: templates and entries from the verified resource file only
(P1). It encourages professional help and walks the user through consent for
human contact. At tier 4 the on-call counsellor is alerted at once (P7); the
escalation record holds time, tier, decision and counsellor ID, never text (P3).

It also handles serious harm to OTHER people (P8): it stays, de-escalates,
urges 999 and offers a licensed counsellor. Identity is collected only by that
counsellor, only with the user's consent, and the counsellor decides on any
warning. For imminent harm the counsellor is alerted to join the chat; Fadfid
itself never identifies the user.
"""

import json
import os
import re
from datetime import datetime, timezone

YES_WORDS = {"نعم", "ايه", "إيه", "اي", "أي", "ايوه", "أيوه", "اوكي", "تمام", "ابي", "أبي", "yes", "yeah", "yep", "ok", "okay", "sure", "y"}
NO_WORDS = {"لا", "لأ", "مابي", "ما ابي", "ما أبي", "no", "nope", "n", "not now"}


class CounsellorService:
    """Prototype stand-in for the on-call counsellor rota. Hours follow service hours."""

    def __init__(self, logPath, counsellorId: str = "C-01", startHour: int = 8, endHour: int = 18):
        self.logPath = logPath
        self.counsellorId = counsellorId
        self.startHour = startHour
        self.endHour = endHour

    def available(self) -> bool:
        now = datetime.now()
        # Service days Saturday-Thursday (Friday = 4 in Python's weekday()).
        return now.weekday() != 4 and self.startHour <= now.hour < self.endHour

    def record(self, tier: int, decision: str, kind: str = "self"):
        self.logPath.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tier": tier,
            "decision": decision,
            "kind": kind,
            "counsellorId": self.counsellorId,
        }
        with open(self.logPath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


PROTOTYPE_NO_COUNSELLOR = {
    "en": ("Thank you for saying yes. This is a research prototype, so no real counsellor is connected here. "
           "Please talk to someone now: the national mental health helpline 16000 (press 4), "
           "or 999 if you or anyone else is in danger. I'm still here with you."),
    "ar": ("شكراً إنك وافقت. هذا نموذج بحثي تجريبي، فما فيه مستشار حقيقي متصل هنا. "
           "تكلم مع أحد الحين: الخط الوطني للدعم النفسي 16000 (اضغط 4)، "
           "أو 999 إذا أنت أو أي أحد في خطر. أنا باقي معك."),
}


class CrisisSupportAgent:
    name = "crisis_support"

    def __init__(self, resourceFile, counsellors: CounsellorService):
        data = json.loads(resourceFile.read_text(encoding="utf-8"))
        self.resources = data["resources"]
        self.templates = data["templates"]
        self.counsellors = counsellors
        if os.environ.get("FADFID_PROTOTYPE") == "1":
            # Public demo: no real counsellor is on call, so never tell a user one is coming.
            for lang, t in self.templates.items():
                for key in ("counsellorAccepted", "counsellorAfterHours", "harmCounsellorAccepted", "harmCounsellorAfterHours"):
                    t[key] = PROTOTYPE_NO_COUNSELLOR[lang]

    def verifiedPhones(self) -> set[str]:
        return {r["phone"] for r in self.resources}

    def resourceLines(self, lang: str) -> str:
        t = self.templates[lang]
        lines = [t["resourcesIntro"]]
        for r in self.resources:
            name = r["nameAr"] if lang == "ar" else r["nameEn"]
            hours = r["hoursAr"] if lang == "ar" else r["hoursEn"]
            lines.append(f"• {name}: {r['phone']} ({hours})")
        return "\n".join(lines)

    @staticmethod
    def parseConsent(text: str) -> str | None:
        words = set(re.findall(r"[\w']+", text.lower()))
        cleaned = text.strip().lower()
        if words & NO_WORDS or cleaned in NO_WORDS:
            return "declined"
        if words & YES_WORDS or cleaned in YES_WORDS:
            return "accepted"
        return None

    def run(self, state, tier: int, lang: str, userText: str) -> str:
        t = self.templates[lang]

        # The user is answering an earlier offer of human contact.
        if state.consentState == "offered" and state.consentKind == "harm":
            reply = self.harmConsent(state, lang, userText)
            if reply:
                return reply
        if state.consentState == "offered" and state.consentKind != "harm":
            answer = self.parseConsent(userText)
            if answer == "accepted":
                state.consentState = "accepted"
                if self.counsellors.available():
                    self.counsellors.record(state.lastCrisisTier, "handoff_with_consent")
                    return t["counsellorAccepted"]
                self.counsellors.record(state.lastCrisisTier, "handoff_requested_after_hours")
                return t["counsellorAfterHours"] + "\n\n" + self.resourceLines(lang)
            if answer == "declined":
                state.consentState = "declined"
                self.counsellors.record(state.lastCrisisTier, "declined")
                return t["counsellorDeclined"] + "\n\n" + self.resourceLines(lang)

        state.crisisTurns += 1
        if state.consentState == "declined":
            state.turnsSinceDecline += 1

        riskRose = tier > state.lastCrisisTier
        state.lastCrisisTier = max(state.lastCrisisTier, tier)
        if tier == 4:
            self.counsellors.record(tier, "counsellor_alerted_no_consent_needed")

        if riskRose or tier == 4:
            parts = [t[f"tier{max(tier, 2)}"], self.resourceLines(lang), t["encourageHelp"]]
        else:
            parts = [t["stayEngaged"]]
            if state.crisisTurns % 3 == 0:
                parts.append(self.resourceLines(lang))

        # A refusal is never final: re-offer after a few turns, or at once if risk rises to tier 4.
        reoffer = state.consentState == "declined" and (state.turnsSinceDecline >= 3 or tier == 4)
        if state.consentState == "none" or reoffer:
            state.consentState = "offered"
            state.consentKind = "self"
            state.turnsSinceDecline = 0
            parts.append(t["offerCounsellor"])
        return "\n\n".join(parts)

    # Harm to other people (P8)

    HARM_TIER = {"intent": 3, "ongoing": 3, "imminent": 4}

    def harmConsent(self, state, lang: str, userText: str) -> str | None:
        t = self.templates[lang]
        answer = self.parseConsent(userText)
        level = self.HARM_TIER.get(state.lastHarm, 3)
        if answer == "accepted":
            state.consentState = "accepted"
            if self.counsellors.available():
                self.counsellors.record(level, "handoff_with_consent_identity_by_counsellor", kind="harm_to_others")
                return t["harmCounsellorAccepted"]
            self.counsellors.record(level, "handoff_requested_after_hours", kind="harm_to_others")
            return t["harmCounsellorAfterHours"]
        if answer == "declined":
            state.consentState = "declined"
            self.counsellors.record(level, "declined", kind="harm_to_others")
            return t["harmCounsellorDeclined"]
        return None

    def runHarm(self, state, harm: str, lang: str, userText: str) -> str:
        t = self.templates[lang]
        if state.consentState == "offered" and state.consentKind == "harm":
            reply = self.harmConsent(state, lang, userText)
            if reply:
                return reply

        rose = HARM_ORDER.index(harm) > HARM_ORDER.index(state.lastHarm)
        state.lastHarm = harm if rose else state.lastHarm
        state.harmTurns += 1
        if state.consentState == "declined":
            state.turnsSinceDecline += 1
        if harm == "imminent":
            self.counsellors.record(4, "counsellor_alerted_to_join_chat", kind="harm_to_others")

        key = {"imminent": "harmImminent", "ongoing": "harmOngoing"}.get(harm, "harmIntent")
        parts = [t[key]] if rose or harm == "imminent" else [t["harmStay"]]

        reoffer = state.consentState == "declined" and (state.turnsSinceDecline >= 3 or (rose and harm == "imminent"))
        firstHarmOffer = state.consentKind != "harm"
        if firstHarmOffer or reoffer:
            state.consentState = "offered"
            state.consentKind = "harm"
            state.turnsSinceDecline = 0
            parts.append(t["harmOfferCounsellor"])
        return "\n\n".join(parts)


HARM_ORDER = ["none", "venting", "past", "intent", "ongoing", "imminent"]
