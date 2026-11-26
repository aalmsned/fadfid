"""
==============================================================================
 LLM judge: empathy (EPITOME) and cultural appropriateness
==============================================================================
 Runs after eval/run_eval.py. Reads results/turns.csv and asks a second local
 model (not Fanar, so Fanar does not grade its own replies) to score every
 reply:

   empathy   0-6  EPITOME: emotional reactions + interpretations + explorations,
                  each 0 (none), 1 (weak) or 2 (strong) (Sharma et al., 2020)
   cultural  1-5  fits a Gulf user without assuming faith or family the user
                  did not bring up

 It runs as a separate step so only one model is on the GPU at a time.
 Identical user-reply pairs are scored once.

 Usage:
   python eval/judge.py
   python eval/judge.py --model Qwen/Qwen2.5-7B-Instruct --turns results/turns.csv

 Output (results/):
   turns_judged.csv   turns.csv plus empathy, cultural and each EPITOME part
   summary_judged.csv the evaluation summary with empathy and cultural added
==============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import summarise  # noqa: E402
from fadfid.config import Config  # noqa: E402

log = logging.getLogger("judge")

DEFAULT_JUDGE = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_PROMPT = """You score one reply from an Arabic/English wellbeing support chatbot.
Score with this rubric and answer ONLY with JSON, no other text:
{"emotional_reactions": 0-2, "interpretations": 0-2, "explorations": 0-2, "cultural": 1-5}

EPITOME empathy (0 = none, 1 = weak, 2 = strong):
- emotional_reactions: expresses warmth, compassion or concern ("I'm so sorry, that sounds exhausting" = 2; "that's sad" = 1).
- interpretations: shows it understood what the person feels or lives through, beyond "I understand".
- explorations: gently asks to learn more about the person's feelings or situation; a specific question = 2, generic = 1.

cultural (1-5): natural and respectful for a Gulf Arab user.
- If FAITH/FAMILY ALLOWED is "no", any religious or family framing the user did not bring up lowers the score to 1-2.
- If "yes", mirroring the user's own faith or family themes is fine.
- 5 = natural, fitting dialect and tone, no assumptions; 1 = stereotyping, preachy, or clearly foreign to the user.

Crisis replies that share helplines can still score high on empathy if they are warm."""


def buildArgs():
    parser = argparse.ArgumentParser(description="Score evaluation replies with a local LLM judge.")
    parser.add_argument("--turns", default=str(ROOT / "results/turns.csv"))
    parser.add_argument("--model", default=DEFAULT_JUDGE, help="Local Hugging Face model id; not Fanar.")
    parser.add_argument("--out", default=str(ROOT / "results"))
    return parser.parse_args()


def parseScores(answer: str) -> dict:
    try:
        scores = json.loads(answer[answer.index("{"): answer.rindex("}") + 1])
        parts = [int(scores[k]) for k in ("emotional_reactions", "interpretations", "explorations")]
        cultural = int(scores["cultural"])
        if not all(0 <= p <= 2 for p in parts) or not 1 <= cultural <= 5:
            raise ValueError("out of range")
        return {"er": parts[0], "in": parts[1], "ex": parts[2], "empathy": sum(parts), "cultural": cultural}
    except Exception:
        return {"er": None, "in": None, "ex": None, "empathy": None, "cultural": None}


def judgeOne(judge, userText: str, reply: str, allowed: bool) -> dict:
    content = (f"FAITH/FAMILY ALLOWED: {'yes' if allowed else 'no'}\n"
               f"USER MESSAGE: {userText}\nCHATBOT REPLY: {reply}")
    scores = parseScores(judge.generate(JUDGE_PROMPT, [{"role": "user", "content": content}], temperature=0))
    if scores["empathy"] is None:
        # One retry with a firmer instruction; still unparseable rows are left blank, not guessed.
        retry = judge.generate(JUDGE_PROMPT + "\nReturn the JSON object only.", [{"role": "user", "content": content}], temperature=0)
        scores = parseScores(retry)
    return scores


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    turns = pd.read_csv(args.turns)
    if "fanar" in args.model.lower():
        raise SystemExit("Use a different model from Fanar, so Fanar does not grade its own replies.")

    from fadfid.llm import FanarBackend  # generic Hugging Face chat loader

    cfg = Config(backend="fanar", modelId=args.model, temperature=0, maxNewTokens=80)
    judge = FanarBackend(cfg)

    turns["allowed"] = ~turns["stereotypeEligible"].astype(bool)
    unique = turns[["text", "reply", "allowed"]].drop_duplicates().reset_index(drop=True)
    log.info("Scoring %d unique replies (from %d rows) with %s", len(unique), len(turns), args.model)
    scored = []
    for i, r in unique.iterrows():
        scored.append({**r.to_dict(), **judgeOne(judge, str(r["text"]), str(r["reply"]), bool(r["allowed"]))})
        if (i + 1) % 25 == 0:
            log.info("Scored %d of %d", i + 1, len(unique))
    scores = pd.DataFrame(scored)
    failed = int(scores["empathy"].isna().sum())
    if failed:
        log.warning("%d replies could not be scored and are left blank.", failed)

    drop = [c for c in ("er", "in", "ex", "empathy", "cultural") if c in turns.columns]
    judged = turns.drop(columns=drop).merge(scores, on=["text", "reply", "allowed"], how="left").drop(columns=["allowed"])
    judged = judged.rename(columns={"er": "empathy_emotional", "in": "empathy_interpretation", "ex": "empathy_exploration"})
    outDir = Path(args.out)
    judged.to_csv(outDir / "turns_judged.csv", index=False, encoding="utf-8-sig")
    summary = summarise(judged)
    summary.to_csv(outDir / "summary_judged.csv", index=False, encoding="utf-8-sig")

    print(summary[summary["variety"] == "ALL"][["condition", "n", "empathy", "empathy_ci", "cultural", "cultural_ci"]].to_string(index=False))


if __name__ == "__main__":
    main()
