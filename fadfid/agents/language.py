"""
Language agent.

Identifies the language variety of each message (Gulf Arabic, MSA, English,
Arabizi, code-switched), converts Arabizi to Arabic script so the crisis
classifier sees one script, and tells the listener which variety to reply in.
CAMeL Tools dialect identification is used when installed; otherwise a
lightweight marker-based detector runs.
"""

import re

ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIZI_WORD_RE = re.compile(r"\b[a-z]*[2356789][a-z]+\b|\b[a-z]+[2356789][a-z]*\b", re.IGNORECASE)

GULF_MARKERS = {
    "ابي", "ابغى", "أبغى", "شلون", "وايد", "عشان", "ليش", "جذي", "شنو", "وش", "زين",
    "الحين", "ودي", "مب", "مو", "هالشي", "طفش", "حيل", "يبه", "يمه", "محد", "اباك", "ابيك",
    "بموت", "باموت", "شي", "اللحين", "خلاص", "اللي", "برا", "حاس", "حاسة", "صرت", "بس", "ياليتني", "ابي", "أبي",
    "ذبحني", "قتلني", "موتتني", "الويكند", "ويكند", "بنتحر", "ما أقدر", "أقدر", "اقدر", "وناسة", "كلها", "ترى", "يعني", "هاذي", "هذي", "اشوف", "ابا", "عيال", "دوام", "تعبان", "تعبانه",
}
MSA_MARKERS = {
    "لماذا", "أريد", "اريد", "الذي", "التي", "سوف", "ليس", "هذا", "هذه", "لقد", "جدا", "جداً", "إنني", "انني",
    "أشعر", "اشعر", "أحاول", "احاول", "لكن", "زلت", "أستطيع", "استطيع", "لم", "لن", "أيضا", "أيضاً", "حيث", "ماذا",
    "عندما", "بسبب", "دائما", "دائماً", "أصبحت", "اصبحت", "كثيرا", "كثيراً", "فقط",
}
ARABIZI_LEXICON = {
    "ana", "enta", "enti", "wallah", "walla", "abi", "abee", "shlon", "wayed", "ya3ni", "msh", "mob",
    "mo", "ma", "aby", "khalas", "inshallah", "yalla", "7ayati", "3ad", "lesh", "laish", "zain",
}

ARABIZI_MAP = [
    ("sh", "ش"), ("kh", "خ"), ("th", "ث"), ("dh", "ذ"), ("gh", "غ"), ("ch", "تش"),
    ("3'", "غ"), ("7'", "خ"), ("2", "ء"), ("3", "ع"), ("5", "خ"), ("6", "ط"), ("7", "ح"), ("9", "ص"), ("8", "ق"),
    ("aa", "ا"), ("ee", "ي"), ("oo", "و"), ("ou", "و"), ("a", "ا"), ("b", "ب"), ("t", "ت"), ("j", "ج"),
    ("d", "د"), ("r", "ر"), ("z", "ز"), ("s", "س"), ("f", "ف"), ("q", "ق"), ("k", "ك"), ("l", "ل"),
    ("m", "م"), ("n", "ن"), ("h", "ه"), ("w", "و"), ("y", "ي"), ("i", "ي"), ("e", "ي"), ("o", "و"),
    ("u", "و"), ("g", "ق"), ("p", "ب"), ("v", "ف"), ("x", "كس"), ("c", "ك"),
]

REPLY_INSTRUCTION = {
    "gulf": ("Reply in Gulf Arabic (Khaleeji), the way a caring Gulf friend texts: use words like "
             "وش، شلون، الحين، واجد، أبي، تبي، ليش، ماعليه، عساك، يعني. Avoid formal words like "
             "قد، لكن، يمكننا، هل، إنّ. Example: \"حاس إنك تعبان واجد هالفترة. وش أكثر شي ضاغط عليك؟\""),
    "msa": "Reply in clear, simple Modern Standard Arabic, as the person writes.",
    "english": "Reply in English, matching the person's tone: casual if they are casual, formal if they are formal.",
    "arabizi": ("The person writes Gulf Arabic in Latin letters with numbers (Arabizi). Reply the same way, "
                "in Arabizi, e.g. \"7ases enk ta3ban wayed. sh9ar?\" (3 = ع, 7 = ح, 5 = خ, 9 = ص, 2 = ء)."),
    "code_switched": ("Reply mixing Gulf Arabic and English the way the user does, in about the same "
                      "proportion. Example: \"wallah that sounds exhausting، وش صار بالضبط؟\""),
}

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


SHORT_REMINDER = {
    "gulf": "Reply in Gulf Arabic (Khaleeji) like a caring friend texting, not formal Arabic.",
    "msa": "Reply in simple Modern Standard Arabic.",
    "english": "Reply in English.",
    "arabizi": "Reply in Arabizi (Gulf Arabic in Latin letters with numbers like 3, 7, 5), not Arabic script.",
    "code_switched": "Reply mixing Gulf Arabic and English in one message, like the person does.",
}


# Formal words that give away a reply that slipped out of Gulf dialect.
REPLY_FORMAL = MSA_MARKERS | {"قد", "هل", "يمكن", "يمكنك", "يمكننا", "حقاً", "حقا", "تماماً", "تماما", "دعنا", "إن", "أن", "ربما", "كيف", "تجرب"}


def replyMatches(variety: str, reply: str) -> bool:
    """Does the reply use the same variety the person wrote in? Used to catch slips into formal Arabic."""
    words = re.findall(r"\S+", reply)
    arabicWords = sum(bool(ARABIC_RE.search(w)) for w in words)
    latinWords = sum(bool(LATIN_RE.search(w)) and not ARABIC_RE.search(w) for w in words)
    tokens = tokenize(reply)
    if variety == "english":
        return latinWords > arabicWords
    if variety == "arabizi":
        return latinWords > arabicWords and (bool(ARABIZI_WORD_RE.search(reply)) or any(t in ARABIZI_LEXICON for t in tokens))
    if variety == "code_switched":
        return arabicWords > 0 and latinWords > 0
    if variety == "msa":
        return arabicWords > latinWords
    # Gulf: Arabic script, and Gulf markers at least as common as formal markers.
    gulfHits = sum(t in GULF_MARKERS for t in tokens)
    formalHits = sum(t in REPLY_FORMAL for t in tokens)
    return arabicWords > latinWords and gulfHits >= 1 and gulfHits >= formalHits


def styleHint(text: str) -> str:
    """Describe how the person writes so the reply can mirror it."""
    words = len(text.split())
    hints = []
    if words <= 6:
        hints.append("They write very short messages: reply in 1-2 short sentences.")
    elif words >= 40:
        hints.append("They write at length: you may use up to 4 sentences.")
    if EMOJI_RE.search(text):
        hints.append("They use emoji: one gentle emoji is fine, unless the topic is serious.")
    if re.search(r"(lol|lmao|haha|ههه|😂|🤣)", text, re.IGNORECASE):
        hints.append("Their tone is joking: keep it light, but still check in gently.")
    return " ".join(hints)


def transliterateArabizi(text: str) -> str:
    """Approximate Arabizi-to-Arabic conversion, good enough for classification."""
    out = text.lower()
    for latin, arabic in ARABIZI_MAP:
        out = out.replace(latin, arabic)
    return out


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[\w']+", text.lower())
    # Also check Arabic tokens without a leading و / ف and without the article ال.
    extra = []
    for t in tokens:
        base = t[1:] if len(t) > 3 and t[0] in "وف" else t
        extra.append(base)
        if base.startswith("ال") and len(base) > 4:
            extra.append(base[2:])
    return tokens + extra


class LanguageAgent:
    name = "language"

    def __init__(self):
        self.camelDid = None
        try:
            from camel_tools.dialectid import DialectIdentifier
            self.camelDid = DialectIdentifier.pretrained()
        except Exception:
            self.camelDid = None

    def detectVariety(self, text: str) -> str:
        arabicChars = len(ARABIC_RE.findall(text))
        latinChars = len(LATIN_RE.findall(text))
        tokens = tokenize(text)
        words = re.findall(r"\S+", text)
        arabicWords = sum(bool(ARABIC_RE.search(w)) for w in words)
        latinWords = sum(bool(LATIN_RE.search(w)) and not ARABIC_RE.search(w) for w in words)
        minority = min(arabicWords, latinWords) / max(arabicWords + latinWords, 1)

        if arabicWords and latinWords and (minority >= 0.15 or min(arabicWords, latinWords) >= 2):
            return "code_switched"
        if latinChars > arabicChars:
            arabiziHits = len(ARABIZI_WORD_RE.findall(text)) + sum(t in ARABIZI_LEXICON for t in tokens)
            return "arabizi" if arabiziHits >= 1 else "english"
        return self.detectArabicDialect(text, tokens)

    def detectArabicDialect(self, text: str, tokens: list[str]) -> str:
        if self.camelDid is not None:
            label = self.camelDid.predict([text])[0].top
            gulfCities = {"DOH", "RIY", "JED", "MUS", "KUW", "ABU", "DUB", "MAN"}
            if label == "MSA":
                return "msa"
            if label in gulfCities:
                return "gulf"
        gulfHits = sum(t in GULF_MARKERS for t in tokens)
        msaHits = sum(t in MSA_MARKERS for t in tokens)
        # Fadfid's users are mainly Gulf speakers: short or unmarked Arabic defaults to Gulf.
        return "msa" if msaHits > gulfHits else "gulf"

    def run(self, text: str) -> dict:
        variety = self.detectVariety(text)
        normalized = transliterateArabizi(text) if variety == "arabizi" else text
        return {
            "variety": variety,
            "normalized": normalized,
            "replyInstruction": REPLY_INSTRUCTION[variety],
            "replyLang": "en" if variety == "english" else "ar",
        }
