"""
==============================================================================
 Compare crisis detectors (proposal objective 3)
==============================================================================
 Scores five ways of spotting a crisis on the same labelled messages:

   keyword     a naive filter: any death or suicide word means crisis
   rules       our hand-written lexicon, with Gulf hyperbole stripped first
   fanar       Fanar alone, with the safety agent's rubric (LLM-only detection)
   classifier  the fine-tuned MARBERT classifier alone
   deployed    what the safety agent actually does: the higher of the
               classifier (or the rules, if no classifier) and Fanar

 Every metric is reported overall and per language variety, with 95%
 bootstrap CIs. McNemar's test compares the detectors pairwise on whether
 each crisis decision (tier >= 2 or not) was right.

 Use test sets the rules were NOT written from: the team's held-out set and
 the classifier's test split. Rows from contrast_set are dropped by default.

 Usage:
   python eval/compare_detectors.py --sets data/heldout_set.csv data/classifier_test.csv
   python eval/compare_detectors.py --sets data/heldout_set.csv --skip fanar

 Output (results/detectors/): summary.csv, rows.csv, mcnemar.csv
==============================================================================
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import bootstrapCi, mcnemar, precisionRecall, weightedKappa  # noqa: E402
from fadfid.agents.language import LanguageAgent  # noqa: E402
from fadfid.agents.safety import CrisisClassifier, SafetyAgent, lexiconTier  # noqa: E402
from fadfid.config import Config  # noqa: E402

log = logging.getLogger("compare_detectors")

DETECTORS = ["keyword", "rules", "fanar", "classifier", "deployed"]
KEYWORDS = re.compile(
    r"(موت|اموت|أموت|بموت|ميت|انتحار|انتحر|نتحر|اقتل نفسي|أقتل نفسي|ذبحني|قتلني|"
    r"\bdie\b|\bdying\b|\bdead\b|\bdeath\b|suicid|kill (myself|me)|end my life|"
    r"\bamoot\b|\bbmoot\b|\bmayet\b|an7ar|ent7ar|inti7ar)",
    re.IGNORECASE,
)
PAIRS = [("classifier", "fanar"), ("deployed", "classifier"), ("deployed", "fanar"),
         ("classifier", "keyword"), ("rules", "keyword"), ("fanar", "keyword")]


def buildArgs():
    parser = argparse.ArgumentParser(description="Compare crisis detectors on labelled messages.")
    parser.add_argument("--sets", nargs="+", required=True, help="CSVs with text, tier[, variety][, kind]")
    parser.add_argument("--skip", nargs="*", default=[], choices=DETECTORS)
    parser.add_argument("--keep-contrast", action="store_true", help="Keep rows from contrast_set (the rules were written from them).")
    parser.add_argument("--out", default=str(ROOT / "results/detectors"))
    return parser.parse_args()


def keywordTier(text: str) -> int:
    return 2 if KEYWORDS.search(text) else 0


def rulesTier(raw: str, normalized: str) -> int:
    return max(lexiconTier(normalized)["tier"], lexiconTier(raw)["tier"])


def loadSets(paths: list[str], keepContrast: bool) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["set"] = Path(p).stem
        if "source" in df.columns and not keepContrast:
            before = len(df)
            df = df[df["source"] != "contrast_set"]
            if before != len(df):
                log.info("%s: dropped %d contrast_set rows (the rules were written from them).", p, before - len(df))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).dropna(subset=["text", "tier"])
    data["tier"] = data["tier"].astype(int)
    for col in ("variety", "kind"):
        if col not in data.columns:
            data[col] = None
    return data.reset_index(drop=True)


def runDetectors(data: pd.DataFrame, skip: list[str]) -> pd.DataFrame:
    language = LanguageAgent()
    cfg = Config()
    classifier = CrisisClassifier(cfg.classifierDir)
    if not classifier.ready and "classifier" not in skip:
        log.warning("No trained classifier at %s; skipping it.", cfg.classifierDir)
        skip = skip + ["classifier"]

    fanar = None
    if "fanar" not in skip and "deployed" not in skip:
        from fadfid.llm import getBackend
        if cfg.backend != "fanar":
            log.warning("FADFID_BACKEND is '%s', so the 'fanar' column is not real Fanar output.", cfg.backend)
        fanar = SafetyAgent(getBackend(cfg), classifier=None, tracker=None)

    rows = []
    for i, r in data.iterrows():
        raw = str(r["text"])
        lang = language.run(raw)
        norm = lang["normalized"]
        row = {"set": r["set"], "text": raw, "gold": int(r["tier"]), "kind": r["kind"],
               "variety": r["variety"] if isinstance(r["variety"], str) else lang["variety"]}
        row["keyword"] = keywordTier(raw)
        row["rules"] = rulesTier(raw, norm)
        if fanar is not None:
            vote, raw = fanar.llmTierRaw([], norm)
            row["fanar"] = vote if vote >= 0 else None
            row["fanar_raw"] = None if vote >= 0 else raw[:200]  # kept to show what Fanar said instead
        if classifier.ready:
            row["classifier"] = classifier.predict(norm)
        if fanar is not None:
            base = row.get("classifier", row["rules"])
            row["deployed"] = max(base, row["fanar"] if row["fanar"] is not None else -1)
        rows.append(row)
        if (i + 1) % 50 == 0:
            log.info("Scored %d of %d", i + 1, len(data))
    return pd.DataFrame(rows)


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    detectors = [d for d in DETECTORS if d in rows.columns]
    groups = list(rows.groupby("variety")) + [("ALL", rows)]
    for variety, g in groups:
        for d in detectors:
            pred = [None if pd.isna(x) else int(x) for x in g[d]]
            gold = g["gold"].tolist()
            pr = precisionRecall(gold, pred)
            crisis = g[g["gold"] >= 2]
            calm = g[g["gold"] < 2]
            hyper = g[g["kind"] == "hyperbole"]
            recall, rLo, rHi = bootstrapCi([0.0 if pd.isna(x) else float(x >= 2) for x in crisis[d]])
            falseAlarm, fLo, fHi = bootstrapCi([0.0 if pd.isna(x) else float(x >= 2) for x in calm[d]])
            hyperFp, hLo, hHi = bootstrapCi([0.0 if pd.isna(x) else float(x >= 2) for x in hyper[d]])
            valid = [(a, b) for a, b in zip(gold, pred) if b is not None]
            # A detector that gives no answer has not flagged the message: it counts as a miss.
            out.append({
                "variety": variety, "detector": d, "n": len(g), "n_crisis": len(crisis),
                "crisis_recall": recall, "crisis_recall_ci": fmtCi(rLo, rHi),
                "crisis_precision": pr["precision"], "crisis_f1": pr["f1"],
                "false_alarm_rate": falseAlarm, "false_alarm_ci": fmtCi(fLo, fHi),
                "hyperbole_fp_rate": hyperFp, "hyperbole_fp_ci": fmtCi(hLo, hHi),
                "tier_weighted_kappa": weightedKappa([a for a, _ in valid], [b for _, b in valid], 0, 4) if valid else float("nan"),
                "no_answer": sum(p is None for p in pred),
            })
    return pd.DataFrame(out)


def fmtCi(lo: float, hi: float) -> str:
    return "" if lo != lo else f"[{lo:.2f}, {hi:.2f}]"


def pairedTests(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a, b in PAIRS:
        if a not in rows.columns or b not in rows.columns:
            continue
        g = rows.dropna(subset=[a, b], how="all").fillna({a: -1, b: -1})  # no answer counts as "not flagged"
        rightA = ((g[a] >= 2) == (g["gold"] >= 2)).tolist()
        rightB = ((g[b] >= 2) == (g["gold"] >= 2)).tolist()
        test = mcnemar(rightA, rightB)
        out.append({"comparison": f"{a} vs {b}", "n": len(g), f"only {a} right": test["onlyA"],
                    f"only {b} right": test["onlyB"], "p": test["p"]})
    return pd.DataFrame(out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    data = loadSets(args.sets, args.keep_contrast)
    log.info("Loaded %d messages (%d crisis, tier >= 2)", len(data), int((data["tier"] >= 2).sum()))

    rows = runDetectors(data, args.skip)
    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(outDir / "rows.csv", index=False, encoding="utf-8-sig")
    summary = summarise(rows)
    summary.to_csv(outDir / "summary.csv", index=False, encoding="utf-8-sig")
    tests = pairedTests(rows)
    tests.to_csv(outDir / "mcnemar.csv", index=False, encoding="utf-8-sig")

    cols = ["detector", "n", "crisis_recall", "crisis_precision", "false_alarm_rate", "hyperbole_fp_rate", "tier_weighted_kappa"]
    print(summary[summary["variety"] == "ALL"][cols].to_string(index=False))
    if not tests.empty:
        print()
        print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
