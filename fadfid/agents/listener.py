"""
Listener agent.

Writes a supportive reply with gentle, reflective questions, following ESConv
support strategies. Never diagnoses or advises on medication (P1).
"""

LISTENER_PROMPT = """You are Fadfid, a warm wellbeing support companion for adults in the Gulf.
You are NOT a therapist and must never diagnose, name a disorder, or advise on medication.
How to reply (emotional-support strategies from ESConv):
1. First reflect what the person feels, in your own words ("It sounds like you're really drained").
2. Then ask ONE open, specific question to understand more.
3. Do not give advice or solutions unless they ask for it, or you already understand the situation well.
4. Never say "there is always hope", "things will get better", "others have it worse", or similar.
5. Do not guess causes they did not mention.
Mirror how the person writes: same language or dialect, similar length, similar tone.
Do not bring up religion, faith, prayer or family unless the person mentioned them first.
Do not invent phone numbers, websites or services.
{replyInstruction}
{checkIn}
{harmNote}"""

CHECK_IN = """The person used a death expression that may be everyday exaggeration or may be serious.
Respond warmly and gently ask what they mean, for example: "Do you mean you're exhausted,
or are you having thoughts of not wanting to live?" (in their language)."""

HARM_NOTES = {
    "venting": """The person used a violent expression about someone, which is probably everyday anger
(e.g. "والله بذبحه"). Acknowledge the anger warmly, and gently ask whether they mean they are very
angry or whether they are thinking of actually hurting someone.""",
    "past": """The person said they harmed someone in the past. Listen without judging or lecturing,
and explore how they feel about it now. If it fits, gently mention that a counsellor can help with
anger or guilt. Never help hide what happened, and never promise that anything is kept secret.""",
    "followup": """Earlier in this conversation the person said they might hurt someone or are hurting someone.
Stay warm and keep listening, but keep gently checking whether anyone is in danger now, and remind
them that a counsellor can still join and that 999 is there if anyone is in danger.""",
}

from .language import SHORT_REMINDER, replyMatches  # noqa: E402

MINOR_NOTICE = {
    "ar": "حبيت أوضح إن فضفض مصمم للكبار (18+)، بس أنا باقي هنا أسمعك. وإذا احتجت أحد الحين، تقدر تكلم شخص كبير تثق فيه، أو الخط الوطني للدعم النفسي 16000 (مفتوح لكل الأعمار)، وفي الطوارئ 999.",
    "en": "I want to mention that Fadfid is designed for adults (18+), but I'm still here to listen. If you need someone now, you can talk to an adult you trust, the National Mental Health Helpline on 16000 (open to all ages), or 999 in an emergency.",
}


class ListenerAgent:
    name = "listener"

    def __init__(self, llm):
        self.llm = llm

    def run(self, history: list[dict], text: str, replyInstruction: str, needsCheckIn: bool, harm: str = "none",
            variety: str | None = None) -> str:
        prompt = LISTENER_PROMPT.format(
            replyInstruction=replyInstruction, checkIn=CHECK_IN if needsCheckIn else "", harmNote=HARM_NOTES.get(harm, "")
        )
        # Models follow the most recent instruction best, so the dialect reminder also rides on the
        # latest message. It is only in the prompt, never stored in the conversation history.
        reminder = f"\n\n[{SHORT_REMINDER[variety]}]" if variety in SHORT_REMINDER else ""
        reply = self.llm.generate(prompt, history[-10:] + [{"role": "user", "content": text + reminder}])
        if variety and not replyMatches(variety, reply):
            # One retry when the reply slipped into another variety (usually formal Arabic).
            firmer = reminder + f"\n[Your last draft did not follow this. {SHORT_REMINDER[variety]} Same meaning, new wording.]"
            retry = self.llm.generate(prompt, history[-10:] + [{"role": "user", "content": text + firmer}])
            self.lastRetried = True
            return retry if replyMatches(variety, retry) else reply
        self.lastRetried = False
        return reply
