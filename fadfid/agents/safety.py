"""
Safety agent.

Assigns each user message a crisis tier (0 none, 1 distress, 2 passive ideation,
3 active ideation, 4 imminent), based on the C-SSRS. It combines three signals
and takes the highest: the fine-tuned classifier (or a lexicon fallback until it
is trained), its own rubric judgement through Fanar, and the tracker's trend.
Gulf hyperbole ("بموت من الطفش") is stripped before matching so it does not
raise the tier on its own.

Separately, it judges harm to OTHER people (P8): none, venting ("والله بذبحه"),
past harm, intent, ongoing harm to a child or dependant, or imminent harm. It
takes the more serious of the rules and a Fanar vote, like the tier.
"""

import logging
import re

log = logging.getLogger(__name__)

HYPERBOLE_PATTERNS = [
    r"(بموت|باموت|بأموت|راح اموت|راح أموت|ميت|ميته|ميتة|متت|اموت|أموت)\s*(من|م)\s*(ال)?(طفش|ضحك|الضحك|جوع|الجوع|تعب|التعب|برد|البرد|حر|الحر|نوم|النوم|ملل|الملل|فرح|الفرح|حماس|الحماس|خوف|الوناسة|وناسة)\S*",
    r"(ذبحني|قتلني|ذبحتني|قتلتني|موتني|موتتني)\s*(ال)?(ضحك|طفش|برد|حر|دوام|زحمة|جوع|ملل|حماس)\S*",
    r"(ميت|اموت|أموت)\s*(ضحك|جوع)",
    r"\bdying\s+(of|from)\s+(laughter|boredom|hunger|embarrassment|cringe|heat|cold)\b",
    r"\b(i'?m|im)\s+dead\s*(😂|🤣|lol|lmao)",
    r"\bkill(ing)?\s+me\s+(lol|lmao|😂|🤣)",
    r"\b(bored|starving|tired)\s+to\s+death\b",
    r"\b(bmoot|bamoot|ba?amoot|amoot|mayet|mayta|mayyet|mt)\s+(mn|min|men|m)\s*(el|al|l)?\s*(6afsh|6afash|ta3ab|jo3|ju3|7ar|bard|da7k|dhi7k|ma?lal|wanasa|nom|7amas)\w*",
    r"\b(tha7ani|th7ani|dha7ani|8atalni|qatalni|gatalni)\s+(el|al)?\s*(6afsh|da7k|dhi7k|7ar|bard|dawam)\w*",
]

TIER4_PATTERNS = [
    r"(قررت|خلاص قررت|حسمت).{0,25}(انهي|أنهي|انتحر|أنتحر|اموت|أموت|حياتي|اخلص)",
    r"(الليلة|الليله|اليوم|بكرة|بكره|هالليلة).{0,25}(انهي|أنهي|نتحر|اخلص على نفسي|أقتل نفسي|اقتل نفسي)",
    r"(انهي|أنهي|نتحر|اقتل نفسي|أقتل نفسي).{0,25}(الليلة|الليله|اليوم|بكرة|بكره)",
    r"(كتبت|كاتب|كاتبة)\s*(رسالة|رساله)\s*(وداع|الوداع)",
    r"(جهزت|جمعت|عندي)\s*(كل شي|الحبوب|الأدوات|الادوات)\s*.{0,15}(خلاص|عشان|للنهاية|انهي)",
    r"\b(tonight|today|tomorrow)\b.{0,30}\b(end it|end my life|kill myself)\b",
    r"\b(end it|end my life|kill myself)\b.{0,30}\b(tonight|today|tomorrow)\b",
    r"\b(i'?ve|i have)\s+(decided|made up my mind)\s+to\s+(die|end it|end my life|kill myself)\b",
    r"\bi have a plan\b",
    r"\b(end it|end my life|kill myself)\b.{0,30}\b(tonight|today|tomorrow)\b",
    r"\b(garart|qarart|9arart|gararat)\b.{0,25}\b(an7ar|ant7ar|aneh7ar|anta7ir|anhi|anhee|amoot)\b",
    r"\b(an7ar|ant7ar|aneh7ar|anta7ir)\b.{0,20}\b(el ?lela|al ?laila|el ?lail|elyom|el ?youm|bacher|bukra)\b",
    r"\b(goodbye forever|this is my goodbye|wrote my (suicide|goodbye) note)\b",
]

TIER3_PATTERNS = [
    r"(انهي|أنهي|انهاء|إنهاء)\s*(حياتي|حياتي\b)",
    r"(نتحر|الانتحار|انتحار)",
    r"(اقتل|أقتل|بقتل|باقتل|بأقتل|راح اقتل|راح أقتل|اذبح|أذبح|بذبح)\s*(نفسي|روحي)",
    r"(اخلص|أخلص)\s*(من حياتي|على نفسي|على روحي)",
    r"\b(kill(ing)? myself|end(ing)? my (own )?life|end(ing)? it all|suicid\w*|take my (own )?life)\b",
    r"\b(an7ar|ant7ar|aneh7ar|anta7ir|ente7ar|inti7ar)\b",
    r"\b(anhi|anhee|anhy)\s+(7ayati|7yati|7aiati)\b",
    r"\b(a8tel|aqtil|ag?tel|a2tel)\s+(nafsi|nefsi|roo7i|ro7i)\b",
]

TIER2_PATTERNS = [
    r"(ودي|ابي|أبي|ابغى|أبغى)\s*(اموت|أموت)",
    r"(ياليتني|يا ليتني|ليتني)\s*(ميت|ميته|ميتة|ما انولدت|ما جيت|ما خلقت)",
    r"(ودي|ابي|أبي)\s*(انام|أنام)\s*(وما|و ما)\s*(اصحى|أصحى|اقوم|أقوم)",
    r"(ما ابي|ما أبي|ما ابغى|مابي|ما ودي)\s*(اعيش|أعيش|اكمل|أكمل)",
    r"(تعبت|طفشت|مليت)\s*من\s*(الحياة|الحياه|العيشة|العيشه)",
    r"(الموت ارحم|الموت أرحم|الموت احسن|الموت أحسن)",
    r"\b(hurt(ing)?|harm(ing)?|cut(ting)?) myself\b",
    r"(اأذي|أأذي|اذي|أذي|أؤذي|اجرح|أجرح)\s*(نفسي|روحي)",
    r"\b(wish (i was|i were|i'd be) dead|don'?t want to (live|be alive|exist)|want to disappear forever|never wake up|better off dead)\b",
    r"\b(abi|aby|abee|ab8a|abgha|wdi|wedi|wdy)\s+(amoot|amot|amut)\b",
    r"\b(abi|aby|abee|wdi|wedi)\s+anam\s+w\s*(ma|mab)\s+(as7a|as7aa|asa7a|agoom|aqoom)\b",
    r"\b(ma|mab|mabi)\s*(abi|aby|ab8a|abgha)?\s+a3ee?sh\b",
    r"\b(laitni|laitany|ya ?laitni)\s+(mayet|mayta|ma ?anwaladt)\b",
]

TIER1_PATTERNS = [
    r"(تعبت|تعبانه|تعبانة|مخنوق|مخنوقة|ضايق|ضايقه|متضايق|متضايقة|حزين|حزينة|محد فاهمني|وحيد|وحيدة|ما انام|ما أنام|قلقان|قلقانة|خايف|خايفة|مكتئب|مالي خلق|منهار|منهارة|ضغط)",
    r"(حزن|بالحزن|الحزن|متعب|متعبة|ضاغط|ضاغطتني|ضاغطني|ما يفهموني|ما يفهمني|محد يهتم|ما احد يهتم)",
    r"\b(stressed|sad|lonely|anxious|hopeless|exhausted|can'?t sleep|overwhelmed|depressed|empty|worthless|tired of everything)\b",
    r"\b(no ?one|nobody|anyone) (to talk to|cares|understands)\b",
    r"\b(ta3ban|ta3bana|ta3baan|ma ?7ad (fahemni|fahmni|yehtam|yihtam)|wa7eed|wa7eeda|da9eg|dayeg|mdaye8|7azeen|7azeena|mo ?zain|ma yemshy|mkhnoog|makhnoog)\b",
]

AMBIGUOUS_DEATH = r"(بموت|باموت|اموت|أموت|ميت|ذبحني|قتلني|i'?m dying|kill me|\bbmoot\b|\bamoot\b)"

TIER_RUBRIC = """You are the safety agent of a wellbeing support system. Classify the user's LAST message
into one crisis tier, based on the Columbia-Suicide Severity Rating Scale:
0 = no risk, including everyday hyperbole (e.g. "بموت من الطفش" = I'm so bored).
1 = distress (sadness, stress, hopelessness) with no thoughts of death.
2 = passive ideation: wishes to be dead, no thought of acting.
3 = active ideation: thoughts of ending their life, no plan or timeframe.
4 = imminent: a plan, intent, means or timeframe.
Use the earlier messages as context. If unsure between two tiers, choose the higher.
Answer with exactly one line in the form TIER: <number>."""


def parseTier(answer: str) -> int:
    """Read a tier from the model's answer; -1 if there isn't one. Accepts "TIER: 2", "Tier 2", "TIER: <2>", "2"."""
    m = re.search(r"(?:tier|المستوى|مستوى|الفئة)\s*[:：=\-]?\s*[<(\[]?\s*([0-4])\b", answer, re.IGNORECASE)
    if not m:
        m = re.match(r"\s*[<(\[]?([0-4])[>)\]]?\s*(?:$|[.\n:=-])", answer)
    return int(m.group(1)) if m else -1


def parseHarm(answer: str) -> str | None:
    m = re.search(r"harm\s*[:：=\-]?\s*[<(\[]?\s*([A-Za-z]+)", answer, re.IGNORECASE)
    label = m.group(1).lower() if m else None
    if label not in HARM_LEVELS:
        found = [l for l in HARM_LEVELS if re.search(rf"\b{l}\b", answer, re.IGNORECASE)]
        label = found[0] if len(found) == 1 else None
    return label


def matchAny(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def stripHyperbole(text: str) -> tuple[str, bool]:
    found = False
    for p in HYPERBOLE_PATTERNS:
        text, n = re.subn(p, " ", text, flags=re.IGNORECASE)
        found = found or n > 0
    return text, found


def lexiconTier(text: str) -> dict:
    """Rule-based tier used before the classifier is trained, and by oversight as a second check."""
    stripped, hadHyperbole = stripHyperbole(text)
    if matchAny(TIER4_PATTERNS, stripped):
        tier = 4
    elif matchAny(TIER3_PATTERNS, stripped):
        tier = 3
    elif matchAny(TIER2_PATTERNS, stripped):
        tier = 2
    elif matchAny(TIER1_PATTERNS, stripped):
        tier = 1
    else:
        tier = 0
    ambiguous = tier < 2 and re.search(AMBIGUOUS_DEATH, stripped, re.IGNORECASE) is not None
    if ambiguous:
        tier = max(tier, 1)
    return {"tier": tier, "hyperbole": hadHyperbole, "ambiguous": ambiguous}


# Harm to other people (P8). Labels from least to most serious.
HARM_LEVELS = ["none", "venting", "past", "intent", "ongoing", "imminent"]
SERIOUS_HARM = {"intent", "ongoing", "imminent"}
SELF_OBJECTS = r"(نفسي|روحي|myself|nafsi|nefsi|roo7i|ro7i)"

VIOLENT_THREAT_PATTERNS = [
    r"(ب|با|بأ|راح ا|راح أ|لا|لأ|بـ)(ذبح|قتل|ضرب|طعن|كسر راس)(ه|ها|هم|ك|كم)\b",
    r"(ابي|أبي|ودي|ابغى|أبغى|افكر|أفكر)\s*(اذبح|أذبح|اقتل|أقتل|اضرب|أضرب|اطعن|أطعن|اذي|أذي|أأذي|اذبحه|اقتله|اضربه)(?!\s*" + SELF_OBJECTS + ")",
    r"\b(i'?ll|i will|i'?m going to|i am going to|gonna|want to|wanna|i could|thinking (about|of)|planning (to|on))\s+(kill|hurt|stab|shoot|beat( up)?|strangle|murder|smash|attack)(ing)?\s+(?!" + SELF_OBJECTS + r"\b)(him|her|them|you|my \w+|that \w+|this \w+|\w+)",
    r"(بقتل|بذبح|باقتل|باذبح|بأقتل|بأذبح|راح اقتل|راح أقتل|راح اذبح|راح أذبح|بغيت اذبح|بغيت أذبح|بغيت اقتل|بغيت أقتل)(ه|ها|هم)?\s+(?!نفسي|روحي)\S+",
    r"(بغيت)\s*(اذبح|أذبح|اقتل|أقتل)(ه|ها|هم|ك)",
    r"س(أ|ا)(قتل|ذبح|ضرب|طعن|ؤذي|آذي|اذي)(ه|ها|هم|ك|كم)",
    r"(اأذي|أأذي|أؤذي|اؤذي|آذي)(ه|ها|هم|ك)",
    r"(ايذاء|إيذاء|اذية|أذية)\s+(?!نفسي|روحي)\S+",
    r"\b(ba|b|ra7 a|abi a|wdi a|a)(th|4|dh|z|d)ba7(a|ah|ha|hom|ik)\b",
    r"\b(ba|b|ra7 a|abi a|wdi a|a)(8|g|q|k)tel(a|ah|ha|hom|ik)\b",
    r"\b(ba|b|ra7 a|abi a|wdi a|a)(9|s)rob(a|ah|ha|hom|ik)\b",
]
HARM_SERIOUS_CUES = [
    r"(افكر|أفكر|قاعد افكر|اخطط|أخطط|خطة|مو مزح|مب مزح|بجد|من جد|جدياً|جديا|ناوي|ناويه|ناوية|قررت)",
    r"\b(i'?ve been (thinking|planning)|i really (want|mean)|seriously|not joking|i mean it|planning to|decided to|i'?m going to)\b",
    r"\b(mo maz7|mb maz7|b ?jad|nawi|nawya|garart|qarart)\b",
]
HARM_IMMINENT_CUES = [
    r"(الليلة|الليله|اليوم|بكرة|بكره|الحين|هالحين|لين يرجع|اذا رجع|إذا رجع|ينتظرني|انتظره|أنتظره)",
    r"(سكين|سكينة|سلاح|مسدس|جهزت|اشتريت|عندي (سلاح|سكين|مسدس))",
    r"\b(tonight|today|tomorrow|right now|when (he|she) (gets|comes) (home|back)|waiting for (him|her|them))\b",
    r"\b(knife|gun|weapon|i bought|i have a (knife|gun|weapon)|i'?ve got a (knife|gun))\b",
    r"(اشتريت|جهزت)\s*(سلاح|سلاحاً|سكين|مسدس)",
    r"\b(el ?lela|al ?laila|elyom|el ?youm|bacher|bukra|al7een|el7in|sikeen|sikkeen|musaddas)\b",
]
IMMINENT_STANDALONE = [
    r"(سكين|السكين|سلاح|مسدس).{0,25}(انتظره|أنتظره|انتظرها|أنتظرها|ينتظر|لين يطلع|لين يرجع)",
    r"\b(knife|gun|weapon)\b.{0,40}\bwaiting for (him|her|them)\b",
]
ONGOING_PATTERNS = [
    r"(اضرب|أضرب|اذي|أذي|ضربت|اعاقب)\s*(ولدي|بنتي|عيالي|اطفالي|أطفالي|ولدي الصغير|الخدامة|الخدامه|الشغالة|الشغاله|امي|أمي|ابوي|أبوي)",
    r"(دايم|كل يوم|على طول|دايماً|دايما)\s*.{0,15}(اضرب|أضرب|ضرب)(ه|ها|هم)?",
    r"\b(a9rob|a9reb|adrub|a4rab|a6rob)\s+(wildi|weldi|binti|3yali|3iyali|atfali|el ?5adama|el ?shaghala)\b",
    r"(اضرب|أضرب)\s*(ولدي|بنتي|عيالي|اطفالي|أطفالي).{0,20}(باستمرار|دايم|كل يوم)",
    r"\b(i (keep )?(hit|hitting|beat|beating|hurt|hurting|slap|slapping|kick|kicking)|i'?ve been (hitting|beating|hurting|slapping)|i (always|often) (hit|beat|hurt))\s+(my )?(son|daughter|kids?|children|child|baby|maid|helper|nanny|mother|mom|mum|father|dad|grandma|grandpa)\b",
]
PAST_HARM_PATTERNS = [
    r"(ضربته|ضربتها|ضربتهم|اذيته|أذيته|آذيته|اذيتها|أذيتها|طعنته|طعنتها|قتلته|قتلتها|اعتديت)",
    r"(ضربت|اذيت|أذيت|آذيت)\s*(واحد|شخص|احد|أحد|ولد|بنت|ريال|حرمة)",
    r"\b(i (hit|hurt|beat( up)?|stabbed|attacked|killed|assaulted|punched)\s+(someone|somebody|a (guy|man|woman|girl|boy|person)|him|her|them|my \w+))\b",
    r"\b(9arabta|9rabta|th(a)?ba7ta|4ba7ta|8ataltah|8tlta)\b",
]

HARM_RUBRIC = """You are the safety agent of a wellbeing support system. Decide whether the user's LAST
message is about harming OTHER people (not themselves). Choose one label:
NONE = not about harming others.
VENTING = angry exaggeration with no real intent (e.g. "والله بذبحه" about a brother who took the car).
PAST = says they harmed someone before, no sign they intend to do it again now.
INTENT = wants or is thinking about hurting a specific person.
ONGOING = is currently or repeatedly hurting a child, partner, domestic worker or other dependant.
IMMINENT = intent plus a time, a plan or a weapon.
Use earlier messages as context. If unsure between two labels, choose the more serious.
Answer with exactly one line in the form HARM: <LABEL>."""


def moreSerious(a: str, b: str) -> str:
    return a if HARM_LEVELS.index(a) >= HARM_LEVELS.index(b) else b


def lexiconHarm(text: str) -> str:
    """Rule-based harm-to-others label (P8)."""
    if matchAny(IMMINENT_STANDALONE, text):
        return "imminent"
    # "بغيت أذبحه" in Gulf usually means "I nearly strangled him": anger, not a plan.
    if re.search(r"بغيت\s*(اذبح|أذبح|اقتل|أقتل)", text) and not matchAny(HARM_SERIOUS_CUES, text):
        return "venting"
    if matchAny(ONGOING_PATTERNS, text):
        return "ongoing"
    threat = matchAny(VIOLENT_THREAT_PATTERNS, text)
    if threat and matchAny(HARM_IMMINENT_CUES, text):
        return "imminent"
    if threat and matchAny(HARM_SERIOUS_CUES, text):
        return "intent"
    if matchAny(PAST_HARM_PATTERNS, text):
        return "past"
    if threat:
        return "venting"
    return "none"


class CrisisClassifier:
    """Fine-tuned MARBERT classifier (scripts/train_classifier.py). Optional until trained."""

    def __init__(self, modelDir):
        self.pipe = None
        if modelDir.exists():
            from transformers import pipeline
            self.pipe = pipeline("text-classification", model=str(modelDir), tokenizer=str(modelDir))
            log.info("Loaded crisis classifier from %s", modelDir)

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    def predict(self, text: str) -> int:
        label = self.pipe(text, truncation=True)[0]["label"]
        return int(label.split("_")[-1])


class SafetyAgent:
    name = "safety"

    def __init__(self, llm, classifier: CrisisClassifier, tracker):
        self.llm = llm
        self.classifier = classifier
        self.tracker = tracker

    @staticmethod
    def classifyTurn(history: list[dict], text: str, form: str) -> list[dict]:
        # Framed as a task so the model labels the message instead of replying to it.
        task = f"Classify this message from the user. Do not reply to it.\nMESSAGE: {text}\nAnswer only {form}"
        return history[-6:] + [{"role": "user", "content": task}]

    def llmTierRaw(self, history: list[dict], text: str) -> tuple[int, str]:
        answer = self.llm.generate(TIER_RUBRIC, self.classifyTurn(history, text, "TIER: <number 0-4>"), temperature=0)
        return parseTier(answer), answer

    def llmTier(self, history: list[dict], text: str) -> int:
        return self.llmTierRaw(history, text)[0]

    def llmHarm(self, history: list[dict], text: str) -> str | None:
        answer = self.llm.generate(HARM_RUBRIC, self.classifyTurn(history, text, "HARM: <LABEL>"), temperature=0)
        return parseHarm(answer)

    def run(self, sessionId: str, normalizedText: str, history: list[dict], rawText: str | None = None) -> dict:
        lex = lexiconTier(normalizedText)
        if rawText and rawText != normalizedText:
            rawLex = lexiconTier(rawText)
            if rawLex["tier"] > lex["tier"] or (rawLex["hyperbole"] and not lex["hyperbole"]):
                lex = rawLex
        classifierTier = self.classifier.predict(normalizedText) if self.classifier.ready else lex["tier"]
        llmVote = self.llmTier(history, normalizedText)
        tier = max(classifierTier, llmVote)

        needsCheckIn = lex["ambiguous"] and tier < 2
        rising = self.tracker.distressRising(sessionId)
        if rising and tier < 2:
            needsCheckIn = True
        # A user who reached tier 2+ earlier in the session stays in crisis support while distress continues.
        if self.tracker.recentMaxTier(sessionId) >= 2 and tier == 1:
            tier = 2

        harmRules = moreSerious(lexiconHarm(normalizedText), lexiconHarm(rawText or normalizedText))
        harmVote = self.llmHarm(history, normalizedText)
        harm = moreSerious(harmRules, harmVote) if harmVote else harmRules

        return {
            "tier": tier,
            "harm": harm,
            "needsCheckIn": needsCheckIn,
            "signals": {"classifier": classifierTier if self.classifier.ready else None, "llm": llmVote, "lexicon": lex["tier"],
                        "hyperbole": lex["hyperbole"], "distressRising": rising,
                        "harmRules": harmRules, "harmLlm": harmVote},
        }
