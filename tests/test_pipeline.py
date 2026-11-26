"""
Pipeline tests (run with: python -m pytest tests -q). They use fake LLMs, so no GPU is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fadfid.agents.language import LanguageAgent  # noqa: E402
from fadfid.agents.safety import lexiconTier  # noqa: E402
from fadfid.config import Config  # noqa: E402
from fadfid.llm import LlmBackend, MockBackend  # noqa: E402
import pytest  # noqa: E402

from fadfid.memory_card import available as memoryCardAvailable  # noqa: E402
from fadfid.memory_card import loadCard, saveCard  # noqa: E402
from fadfid.pipeline import Fadfid  # noqa: E402


class FixedReplyLlm(LlmBackend):
    """Returns a fixed listener reply, to test what oversight does with it."""

    def __init__(self, reply: str):
        self.reply = reply
        self.mock = MockBackend()

    def generate(self, systemPrompt, messages, temperature=None):
        if "TIER:" in systemPrompt or "REWRITE" in systemPrompt:
            return self.mock.generate(systemPrompt, messages)
        return self.reply


def makeBot(tmp_path, llm=None, condition="responsible"):
    cfg = Config(backend="mock")
    cfg.escalationLog = tmp_path / "escalations.jsonl"
    cfg.interventionLog = tmp_path / "interventions.jsonl"
    return Fadfid(cfg, condition=condition, llm=llm)


def test_hyperbole_is_not_crisis():
    for text in ["بموت من الطفش", "ذبحني الضحك", "bmoot mn el6afsh", "I'm dying of boredom"]:
        assert lexiconTier(LanguageAgent().run(text)["normalized"])["tier"] < 2 or lexiconTier(text)["tier"] < 2


def test_crisis_tiers():
    assert lexiconTier("ودي أنام وما أصحى")["tier"] == 2
    assert lexiconTier("أفكر أنهي حياتي")["tier"] == 3
    assert lexiconTier("خلاص قررت أنهي حياتي الليلة")["tier"] == 4


def test_language_varieties():
    agent = LanguageAgent()
    assert agent.detectVariety("ودي أطلع بس محد فاضي") == "gulf"
    assert agent.detectVariety("أشعر بالحزن الشديد هذه الأيام") == "msa"
    assert agent.detectVariety("I feel lonely") == "english"
    assert agent.detectVariety("ana ta3ban wayed") == "arabizi"
    assert agent.detectVariety("I'm so tired و محد فاهمني") == "code_switched"


def test_crisis_never_ends_and_offers_counsellor(tmp_path):
    bot = makeBot(tmp_path)
    sid = bot.startSession()
    r = bot.handleMessage(sid, "ما ابي أعيش بعد")
    assert r["route"] == "crisis" and "16000" in r["reply"] and "(نعم / لا)" in r["reply"]
    r = bot.handleMessage(sid, "لا")
    assert r["reply"] and "القرار لك" in r["reply"]
    r = bot.handleMessage(sid, "خلاص قررت أنهي حياتي الليلة")
    assert r["tier"] == 4 and "999" in r["reply"]
    log = (tmp_path / "escalations.jsonl").read_text()
    assert "counsellor_alerted_no_consent_needed" in log and "ما ابي" not in log  # no message text stored


def test_oversight_blocks_diagnosis(tmp_path):
    bot = makeBot(tmp_path, FixedReplyLlm("It sounds like you have depression. Try sertraline."))
    sid = bot.startSession()
    r = bot.handleMessage(sid, "I feel sad")
    assert r["oversight"] == "block" and "depression" not in r["reply"]


def test_oversight_removes_unrequested_faith(tmp_path):
    reply = "I'm sorry you feel this way. Maybe pray and trust God. What happened today?"
    bot = makeBot(tmp_path, FixedReplyLlm(reply))
    sid = bot.startSession()
    r = bot.handleMessage(sid, "I feel lonely here")
    assert r["oversight"] == "rewrite" and "God" not in r["reply"] and "What happened today?" in r["reply"]

    # Same reply is allowed once the user raises faith themselves.
    sid2 = bot.startSession()
    r2 = bot.handleMessage(sid2, "I try to pray but I still feel lonely")
    assert "God" in r2["reply"]


def test_oversight_strips_invented_numbers(tmp_path):
    bot = makeBot(tmp_path, FixedReplyLlm("You can call 44556677 for help. How are you feeling now?"))
    sid = bot.startSession()
    r = bot.handleMessage(sid, "I feel stressed")
    assert "44556677" not in r["reply"]


def test_no_oversight_condition_lets_it_through(tmp_path):
    bot = makeBot(tmp_path, FixedReplyLlm("You have depression."), condition="no_oversight")
    sid = bot.startSession()
    assert "depression" in bot.handleMessage(sid, "I feel sad")["reply"]


def test_minor_gets_notice_and_support(tmp_path):
    bot = makeBot(tmp_path)
    sid = bot.startSession()
    r = bot.handleMessage(sid, "I'm 15 and school is too much")
    assert "18+" in r["reply"] and "16000" in r["reply"]


def test_session_deleted(tmp_path):
    bot = makeBot(tmp_path)
    sid = bot.startSession()
    bot.handleMessage(sid, "hello")
    bot.endSession(sid)
    assert sid not in bot.tracker.sessions


@pytest.mark.skipif(not memoryCardAvailable(), reason="cryptography not installed or blocked")
def test_memory_card_roundtrip(tmp_path):
    card = tmp_path / "card.bin"
    saveCard("You talked about work stress.", "secret", card)
    assert loadCard("secret", card) == "You talked about work stress."
    assert loadCard("wrong", card) is None
