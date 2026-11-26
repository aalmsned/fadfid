"""
Oversight agent.

Checks every outgoing reply before the user sees it.
  - Blocks diagnoses, medication advice, method details and unverified resources (P1).
  - Rewrites faith or family framing the user did not ask for (P2).
  - Escalates risk the safety agent missed to the crisis support agent (P6, P7).
  - Blocks anything that would help hurt another person or hide harm, and escalates
    serious harm to others the safety agent missed (P8).
Interventions are logged as policy, action and time only, never text (P3).
FanarGuard can be plugged in as a second check through `extraGuard`.
"""

import json
import re
from datetime import datetime, timezone

from .safety import SERIOUS_HARM, lexiconHarm, lexiconTier
from .tracker import FAITH_WORDS, FAMILY_WORDS, containsAny

DIAGNOSIS_PATTERNS = [
    r"(عندك|لديك|انت مصاب|أنت مصاب|تعاني من)\s*(اكتئاب|قلق مرضي|اضطراب|ثنائي القطب|وسواس|فصام|صدمة نفسية)",
    r"(يبدو|واضح)\s*(انك|أنك|إنك)\s*(مكتئب|مصاب)",
    r"\byou (have|are suffering from|might have|probably have|seem to have)\s+(clinical )?(depression|an? anxiety disorder|bipolar|ocd|ptsd|schizophrenia|a disorder)\b",
    r"\b(diagnos(e|is|ed))\b",
]
MEDICATION_WORDS = [
    "سيرترالين", "زولفت", "بروزاك", "فلوكستين", "زاناكس", "سيبرالكس", "مضاد اكتئاب", "مضادات الاكتئاب",
    "حبوب منومة", "جرعة", "sertraline", "zoloft", "prozac", "fluoxetine", "xanax", "cipralex", "ssri",
    "antidepressant", "sleeping pills", "dosage", "dose of",
]
METHOD_WORDS = ["جرعة زائدة", "overdose", "how to hang", "lethal", "قاتلة", "طريقة الانتحار"]
HARM_HELP_PATTERNS = [
    r"(اخفي|أخفي|تخفي|تخبي|اخبي|أخبي|امسح|تمسح)\s*(الجثة|الجثه|الدليل|الأدلة|الادلة|الآثار|الاثار|الدم)",
    r"(بدون ما|بدون لا|عشان ما)\s*(ينكشف|تنكشف|احد يدري|أحد يدري|يدرون|يمسكونك|تنمسك)",
    r"(تضربه|تأذيه|تأذيها|تضربها)\s*(بدون|عشان ما)\s*(اثر|أثر|علامات)",
    r"\b(how to|you could|you can|try to)\s+(hide|cover up|get away with|avoid getting caught|poison|hurt|stab|strangle)\b",
    r"\bwithout (getting caught|anyone (knowing|finding out)|leaving (marks|bruises|evidence))\b",
    r"\b(untraceable|leave no (marks|evidence|trace))\b",
]
SECRECY_PROMISES = [
    r"(سرك في بير|ما راح اقول لأحد|ما راح أقول لأحد|محد راح يعرف|بيبقى بيننا)",
    r"\b(i won'?t tell anyone|your secret is safe|no one will (ever )?know|this stays between us)\b",
]

PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s\-]{1,14}\d)(?!\d)")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

SAFE_FALLBACK = {
    "ar": "أنا هنا أسمعك. تقدر تحكيلي أكثر عن اللي تمر فيه؟",
    "en": "I'm here and listening. Can you tell me more about what you're going through?",
}


def splitSentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?؟。\n])\s*", text) if s.strip()]


class OversightAgent:
    name = "oversight"

    def __init__(self, verifiedPhones: set[str], logPath, extraGuard=None):
        self.verifiedPhones = verifiedPhones
        self.logPath = logPath
        self.extraGuard = extraGuard

    def log(self, policy: str, action: str):
        self.logPath.parent.mkdir(parents=True, exist_ok=True)
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "policy": policy, "action": action}
        with open(self.logPath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def violations(self, reply: str, permissions: dict) -> list[str]:
        found = []
        if any(re.search(p, reply, re.IGNORECASE) for p in DIAGNOSIS_PATTERNS):
            found.append("P1_diagnosis")
        if containsAny(reply, MEDICATION_WORDS):
            found.append("P1_medication")
        if containsAny(reply, METHOD_WORDS):
            found.append("P1_method")
        phones = {re.sub(r"\D", "", p) for p in PHONE_RE.findall(reply)}
        if phones - self.verifiedPhones or URL_RE.search(reply):
            found.append("P1_unverified_resource")
        if not permissions.get("faith") and containsAny(reply, FAITH_WORDS):
            found.append("P2_faith")
        if not permissions.get("family") and containsAny(reply, FAMILY_WORDS):
            found.append("P2_family")
        if any(re.search(p, reply, re.IGNORECASE) for p in HARM_HELP_PATTERNS):
            found.append("P8_harm_help")
        if any(re.search(p, reply, re.IGNORECASE) for p in SECRECY_PROMISES):
            found.append("P8_secrecy_promise")
        if self.extraGuard is not None and not self.extraGuard(reply):
            found.append("P1_guard")
        return found

    def cleanSentence(self, sentence: str, found: list[str], permissions: dict) -> bool:
        """True if the sentence can be kept."""
        if "P2_faith" in found and not permissions.get("faith") and containsAny(sentence, FAITH_WORDS):
            return False
        if "P2_family" in found and not permissions.get("family") and containsAny(sentence, FAMILY_WORDS):
            return False
        if "P1_unverified_resource" in found:
            phones = {re.sub(r"\D", "", p) for p in PHONE_RE.findall(sentence)}
            if phones - self.verifiedPhones or URL_RE.search(sentence):
                return False
        return True

    def review(self, reply: str, permissions: dict, lang: str, userText: str, routedTier: int,
               routedHarm: str = "none") -> dict:
        # Escalate serious harm to others the safety agent missed (P8).
        strictHarm = lexiconHarm(userText)
        if strictHarm in SERIOUS_HARM and routedHarm not in SERIOUS_HARM:
            self.log("P8", "escalate_harm_to_others")
            return {"action": "escalate", "kind": "harm", "harm": strictHarm, "reply": reply, "tier": routedTier}

        # Escalate risk the safety agent missed.
        strictTier = lexiconTier(userText)["tier"]
        if strictTier >= 2 and routedTier < 2:
            self.log("P6", "escalate_to_crisis_support")
            return {"action": "escalate", "kind": "self", "reply": reply, "tier": strictTier}

        found = self.violations(reply, permissions)
        if not found:
            return {"action": "pass", "reply": reply}

        blocking = {"P1_diagnosis", "P1_medication", "P1_method", "P1_guard", "P8_harm_help", "P8_secrecy_promise"}
        if blocking & set(found):
            for v in found:
                self.log(v.split("_")[0], "block")
            return {"action": "block", "reply": SAFE_FALLBACK[lang], "violations": found}

        kept = [s for s in splitSentences(reply) if self.cleanSentence(s, found, permissions)]
        for v in found:
            self.log(v.split("_")[0], "rewrite")
        rewritten = " ".join(kept).strip() or SAFE_FALLBACK[lang]
        return {"action": "rewrite", "reply": rewritten, "violations": found}
