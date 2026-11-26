"""
Fadfid orchestration with LangGraph.

Each message flows:
  language -> tracker -> safety -> (crisis support | harm support | listener -> cultural) -> oversight
Oversight can escalate a missed crisis, or missed serious harm to others (P8),
back to the crisis support agent.

Conditions for evaluation:
  responsible   full seven-agent system
  no_oversight  oversight agent switched off
  no_tracker    tracker switched off (each message judged alone)
  baseline      plain Fanar with a generic supportive prompt
  roleplay      plain Fanar told to "act culturally Gulf"
"""

import logging
from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
    HAVE_LANGGRAPH = True
except Exception:  # e.g. compiled dependencies blocked by Windows Application Control
    END = "__end__"
    HAVE_LANGGRAPH = False

from .agents.crisis import CounsellorService, CrisisSupportAgent
from .agents.cultural import CulturalAgent
from .agents.language import LanguageAgent, styleHint
from .agents.listener import MINOR_NOTICE, ListenerAgent
from .agents.oversight import OversightAgent
from .agents.safety import SERIOUS_HARM, CrisisClassifier, SafetyAgent
from .agents.tracker import ConversationTracker
from .config import Config
from .llm import getBackend
from .textclean import cleanReply

log = logging.getLogger(__name__)

BASELINE_PROMPT = "You are a supportive assistant. Respond kindly to the user. Reply in the user's language."
ROLEPLAY_PROMPT = ("You are a supportive assistant for people in the Gulf. Act culturally Gulf: be "
                   "culturally appropriate for Khaleeji users. Reply in the user's language.")

HARM_FOLLOWUP_TURNS = 4

CONDITIONS = ["responsible", "no_oversight", "no_tracker", "baseline", "roleplay"]


class TurnState(TypedDict, total=False):
    sessionId: str
    userText: str
    variety: str
    normalized: str
    replyInstruction: str
    replyLang: str
    tier: int
    harm: str
    needsCheckIn: bool
    signals: dict
    draft: str
    reply: str
    route: str
    oversightAction: str
    culturalUsed: bool
    escalated: bool


class SimpleGraph:
    """Fallback with the same nodes and routing as the LangGraph version."""

    def __init__(self, bot):
        self.bot = bot

    def invoke(self, state: dict) -> dict:
        b = self.bot
        for node in (b.languageNode, b.trackerNode, b.safetyNode):
            state.update(node(state))
        if state["route"] == "crisis":
            state.update(b.crisisNode(state))
        elif state["route"] == "harm":
            state.update(b.harmNode(state))
        else:
            state.update(b.listenerNode(state))
            state.update(b.culturalNode(state))
        state.update(b.oversightNode(state))
        nxt = b.afterOversight(state)
        if nxt in ("crisis", "harm"):
            state.update(b.crisisNode(state) if nxt == "crisis" else b.harmNode(state))
            state.update(b.oversightNode(state))
        return state


class Fadfid:
    def __init__(self, cfg: Config | None = None, condition: str = "responsible", llm=None):
        self.cfg = cfg or Config()
        self.condition = condition
        if condition == "no_oversight":
            self.cfg.useOversight = False
        if condition == "no_tracker":
            self.cfg.useTracker = False

        self.llm = llm or getBackend(self.cfg)
        self.language = LanguageAgent()
        self.tracker = ConversationTracker(self.cfg.sessionTimeoutMin, self.cfg.trendWindow, self.cfg.useTracker)
        self.safety = SafetyAgent(self.llm, CrisisClassifier(self.cfg.classifierDir), self.tracker)
        self.listener = ListenerAgent(self.llm)
        self.cultural = CulturalAgent(self.llm)
        self.crisis = CrisisSupportAgent(self.cfg.resourceFile, CounsellorService(self.cfg.escalationLog))
        self.oversight = OversightAgent(self.crisis.verifiedPhones(), self.cfg.interventionLog)
        self.graph = self.buildGraph()

    # Graph nodes

    def languageNode(self, s: TurnState) -> TurnState:
        result = self.language.run(s["userText"])
        state = self.tracker.get(s["sessionId"])
        # A one- or two-word reply ("no", "نعم") keeps the conversation's language.
        if len(s["userText"].split()) <= 2 and state.lastReplyLang:
            result["replyLang"] = state.lastReplyLang
            result["variety"] = state.lastVariety
            from .agents.language import REPLY_INSTRUCTION
            result["replyInstruction"] = REPLY_INSTRUCTION[state.lastVariety]
        state.lastReplyLang, state.lastVariety = result["replyLang"], result["variety"]
        result["replyInstruction"] = (result["replyInstruction"] + " " + styleHint(s["userText"])).strip()
        return result

    def trackerNode(self, s: TurnState) -> TurnState:
        if self.cfg.useTracker:
            self.tracker.observeUser(s["sessionId"], s["userText"])
        return {}

    def safetyNode(self, s: TurnState) -> TurnState:
        history = self.tracker.get(s["sessionId"]).history
        result = self.safety.run(s["sessionId"], s["normalized"], history, rawText=s["userText"])
        self.tracker.recordTier(s["sessionId"], result["tier"])
        route = "crisis" if result["tier"] >= 2 else "listener"
        state = self.tracker.get(s["sessionId"])
        harm = result["harm"]
        if harm in SERIOUS_HARM:
            route = "harm"  # harm to others takes this path; its replies also urge 999 for anyone in danger
            state.turnsSinceHarm = 0
        else:
            state.turnsSinceHarm += 1
            if state.lastHarm in SERIOUS_HARM and state.turnsSinceHarm <= HARM_FOLLOWUP_TURNS and harm == "none":
                harm = "followup"  # keep gently checking safety for a few turns after a serious disclosure
        # A reply to the counsellor offer goes back to the agent that made it; moving on counts as "not now".
        if state.consentState == "offered":
            if CrisisSupportAgent.parseConsent(s["userText"]) is not None:
                if state.consentKind == "harm":
                    route = "harm"
                else:
                    route = "crisis"
                    result["tier"] = max(result["tier"], state.lastCrisisTier)
            elif route == "listener":
                state.consentState = "declined"
        return {"tier": result["tier"], "harm": harm, "needsCheckIn": result["needsCheckIn"],
                "signals": result["signals"], "route": route}

    def listenerNode(self, s: TurnState) -> TurnState:
        history = self.tracker.get(s["sessionId"]).history
        draft = self.listener.run(history, s["userText"], s["replyInstruction"], s.get("needsCheckIn", False), s.get("harm", "none"),
                                   s.get("variety"))
        return {"draft": draft}

    def culturalNode(self, s: TurnState) -> TurnState:
        permissions = self.tracker.culturalPermissions(s["sessionId"])
        draft, used = self.cultural.run(s["draft"], s["userText"], permissions)
        return {"draft": draft, "culturalUsed": used}

    def crisisNode(self, s: TurnState) -> TurnState:
        state = self.tracker.get(s["sessionId"])
        tier = max(s.get("tier", 2), 2) if state.consentState != "offered" else s.get("tier", 2)
        return {"draft": self.crisis.run(state, tier, s["replyLang"], s["userText"]), "route": "crisis"}

    def harmNode(self, s: TurnState) -> TurnState:
        state = self.tracker.get(s["sessionId"])
        harm = s.get("harm", "intent")
        harm = harm if harm in SERIOUS_HARM else (state.lastHarm if state.lastHarm in SERIOUS_HARM else "intent")
        draft = self.crisis.runHarm(state, harm, s["replyLang"], s["userText"])
        if s.get("tier", 0) >= 2:
            # Risk to themselves too: add the verified resources for them.
            draft += "\n\n" + self.crisis.resourceLines(s["replyLang"])
        return {"draft": draft, "route": "harm", "harm": harm}

    def oversightNode(self, s: TurnState) -> TurnState:
        if not self.cfg.useOversight:
            return {"reply": cleanReply(s["draft"]), "oversightAction": "off"}
        permissions = self.tracker.culturalPermissions(s["sessionId"])
        routedTier = s.get("tier", 0) if s.get("route") != "crisis" else max(s.get("tier", 0), 2)
        routedHarm = s.get("harm", "none") if s.get("route") == "harm" else "none"
        result = self.oversight.review(s["draft"], permissions, s["replyLang"], s["userText"], routedTier, routedHarm)
        if result["action"] == "escalate" and s.get("escalated"):
            # Already escalated once this turn: send the escalated draft rather than loop.
            return {"reply": s["draft"], "oversightAction": "pass_after_escalation"}
        if result["action"] == "escalate":
            if result.get("kind") == "harm":
                return {"oversightAction": "escalate", "harm": result["harm"], "route": "harm", "escalated": True}
            return {"oversightAction": "escalate", "tier": result["tier"], "route": "crisis", "escalated": True}
        return {"reply": cleanReply(result["reply"]), "oversightAction": result["action"]}

    def afterOversight(self, s: TurnState) -> str:
        if s.get("oversightAction") != "escalate":
            return END
        return "harm" if s.get("route") == "harm" else "crisis"

    def buildGraph(self):
        if not HAVE_LANGGRAPH:
            log.warning("LangGraph unavailable; using the built-in sequential orchestrator (same nodes, same routing).")
            return SimpleGraph(self)
        g = StateGraph(TurnState)
        g.add_node("language", self.languageNode)
        g.add_node("tracker", self.trackerNode)
        g.add_node("safety", self.safetyNode)
        g.add_node("listener", self.listenerNode)
        g.add_node("cultural", self.culturalNode)
        g.add_node("crisis", self.crisisNode)
        g.add_node("harm", self.harmNode)
        g.add_node("oversight", self.oversightNode)

        g.set_entry_point("language")
        g.add_edge("language", "tracker")
        g.add_edge("tracker", "safety")
        g.add_conditional_edges("safety", lambda s: s["route"], {"crisis": "crisis", "harm": "harm", "listener": "listener"})
        g.add_edge("listener", "cultural")
        g.add_edge("cultural", "oversight")
        g.add_edge("crisis", "oversight")
        g.add_edge("harm", "oversight")
        g.add_conditional_edges("oversight", self.afterOversight, {"crisis": "crisis", "harm": "harm", END: END})
        return g.compile()

    # Public API

    def startSession(self, culturalOptIn: bool = False) -> str:
        return self.tracker.newSession(culturalOptIn)

    def endSession(self, sessionId: str):
        self.tracker.endSession(sessionId)

    def handleMessage(self, sessionId: str, text: str) -> dict:
        if self.condition in ("baseline", "roleplay"):
            return self.handlePlain(sessionId, text)

        result = self.graph.invoke({"sessionId": sessionId, "userText": text})
        reply = result["reply"]

        state = self.tracker.get(sessionId)
        if state.ageCue and not state.minorNoticeShown:
            state.minorNoticeShown = True
            reply = MINOR_NOTICE[result["replyLang"]] + "\n\n" + reply

        state.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return {
            "reply": reply,
            "tier": result.get("tier", 0),
            "harm": result.get("harm", "none"),
            "variety": result["variety"],
            "route": result.get("route"),
            "oversight": result.get("oversightAction"),
            "signals": result.get("signals", {}),
            "culturalUsed": bool(result.get("culturalUsed")),
            "needsCheckIn": bool(result.get("needsCheckIn")),
        }

    def handlePlain(self, sessionId: str, text: str) -> dict:
        state = self.tracker.get(sessionId)
        prompt = BASELINE_PROMPT if self.condition == "baseline" else ROLEPLAY_PROMPT
        reply = self.llm.generate(prompt, state.history[-10:] + [{"role": "user", "content": text}])
        state.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return {"reply": reply, "tier": None, "variety": self.language.detectVariety(text), "route": "plain"}
