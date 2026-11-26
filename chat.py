"""
==============================================================================
 Fadfid (فضفض) - command-line chat
==============================================================================
 Runs the full seven-agent system in your terminal.

 Usage:
   python chat.py                 # mock backend, no GPU needed
   python chat.py --backend fanar # real Fanar-1-9B-Instruct
   python chat.py --debug         # show tier, variety, route after each reply

 Commands during chat:  /optin  /end
==============================================================================
"""

import argparse
import getpass
import logging

from fadfid.config import Config
from fadfid.memory_card import available as memoryCardAvailable
from fadfid.memory_card import deleteCard, draftSummary, loadCard, saveCard
from fadfid.pipeline import CONDITIONS, Fadfid

ENTRY_NOTICE = """
فضفض | Fadfid
--------------------------------------------------------------
Before we start, please read these four points:
1. Fadfid is a support companion, not therapy, and never diagnoses.
2. Your conversation is not saved. It disappears when you leave.
3. If you are in danger, Fadfid will stay with you, share trusted
   help (999, 16000), and offer a trained counsellor.
4. Confidential, with one limit: Fadfid can't help hide a plan to
   hurt someone. If you talk about hurting someone, it will ask you
   to talk with a counsellor, who would ask who you are only if you
   agree. Fadfid itself doesn't know who you are.
--------------------------------------------------------------"""


def askYesNo(prompt: str, default: bool = False) -> bool:
    answer = input(f"{prompt} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "نعم", "ايه", "اي"}


def buildArgs():
    parser = argparse.ArgumentParser(description="Chat with Fadfid in the terminal.")
    parser.add_argument("--backend", choices=["mock", "fanar"], default=None)
    parser.add_argument("--condition", choices=CONDITIONS, default="responsible")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def runChat(args):
    cfg = Config()
    if args.backend:
        cfg.backend = args.backend
    bot = Fadfid(cfg, condition=args.condition)
    print(f"[Model: {'Fanar-1-9B-Instruct' if cfg.backend == 'fanar' else 'MOCK (fixed test replies, not Fanar)'}]")

    print(ENTRY_NOTICE)
    if not askYesNo("I understand these points and want to continue."):
        return
    if not askYesNo("Are you 18 or older?"):
        print("Fadfid is designed for adults, but you're welcome to talk. "
              "You can also call 16000 (all ages) or 999 in an emergency, or talk to an adult you trust.")

    optIn = askYesNo("Would you like Fadfid to include faith or family in its support when relevant?")
    sessionId = bot.startSession(culturalOptIn=optIn)

    if memoryCardAvailable() and askYesNo("Load a saved memory card from this device?"):
        summary = loadCard(getpass.getpass("Passphrase: "))
        print(f"\n[Memory card] {summary}\n" if summary else "No card found, or wrong passphrase.")
        if summary:
            bot.tracker.get(sessionId).history.append({"role": "assistant", "content": f"(Previous session notes) {summary}"})

    print("\nType your message. /end to finish.\n")
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text == "/end":
            break
        if text == "/optin":
            bot.tracker.get(sessionId).culturalOptIn = True
            print("Fadfid: Okay, I'll include faith or family when it fits.\n")
            continue
        result = bot.handleMessage(sessionId, text)
        print(f"Fadfid: {result['reply']}\n")
        if args.debug:
            print(f"  [tier={result['tier']} variety={result['variety']} route={result['route']} harm={result.get('harm')} "
                  f"oversight={result.get('oversight')} signals={result.get('signals')}]\n")

    offerMemoryCard(bot, sessionId)
    bot.endSession(sessionId)
    print("Session ended and deleted. Take care. 🌿")


def offerMemoryCard(bot, sessionId):
    if not memoryCardAvailable():
        return
    if not askYesNo("Save a short summary on THIS device so we can continue next time?"):
        return
    summary = draftSummary(bot.llm, bot.tracker.get(sessionId).history)
    print(f"\nDraft summary:\n{summary}\n")
    edited = input("Edit it, or press Enter to keep it (type DELETE to remove any saved card): ").strip()
    if edited == "DELETE":
        deleteCard()
        print("Saved card deleted.")
        return
    passphrase = getpass.getpass("Choose a passphrase (protects the card on shared phones): ")
    saveCard(edited or summary, passphrase)
    print("Saved on this device only.")


def main():
    args = buildArgs()
    logging.basicConfig(level=logging.WARNING)
    runChat(args)


if __name__ == "__main__":
    main()
