"""
==============================================================================
 Agreement between raters, and between raters and the automatic scores
==============================================================================
 Reads the two filled hand-scoring sheets and the key from eval/scoring_sheet.py
 and reports:

   1. Rater 1 vs rater 2        is the rubric reliable?
   2. Each rater vs the judge   can we trust the LLM judge's empathy and cultural scores?
   3. Each rater vs the checks  can we trust the automatic unsafe and stereotyping
                                checks (the same rules the oversight agent uses)?
   4. Human scores per condition, unblinded only at this step

 Ordered scores (empathy 0-6, cultural 1-5) use quadratic-weighted kappa;
 yes/no scores use Cohen's kappa.

 Usage:
   python eval/agreement.py
   python eval/agreement.py --dir results/scoring --turns results/turns_judged.csv

 Output (results/scoring/): agreement.csv, human_by_condition.csv
==============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import bootstrapCi, cohenKappa, weightedKappa  # noqa: E402

log = logging.getLogger("agreement")
EMPATHY_PARTS = ["empathy_emotional", "empathy_interpretation", "empathy_exploration"]


def buildArgs():
    parser = argparse.ArgumentParser(description="Report agreement for the hand-scored sample.")
    parser.add_argument("--dir", default=str(ROOT / "results/scoring"))
    parser.add_argument("--turns", default=None, help="Default: results/turns_judged.csv if present, else results/turns.csv")
    return parser.parse_args()


def readRater(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in EMPATHY_PARTS + ["cultural", "unsafe", "stereotyped"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["empathy"] = df[EMPATHY_PARTS].sum(axis=1, min_count=3)
    keep = ["blind_id", "empathy", "cultural", "unsafe", "stereotyped"]
    return df[keep].rename(columns={c: f"{c}_{tag}" for c in keep if c != "blind_id"})


def kappaRow(label: str, a: pd.Series, b: pd.Series, kind: str, lo: int = 0, hi: int = 1) -> dict:
    both = pd.concat([a, b], axis=1).dropna()
    x, y = both.iloc[:, 0].astype(int).tolist(), both.iloc[:, 1].astype(int).tolist()
    if not x:
        return {"comparison": label, "n": 0, "kappa": float("nan"), "type": kind}
    k = weightedKappa(x, y, lo, hi) if kind == "weighted" else cohenKappa(x, y)
    exact = sum(i == j for i, j in zip(x, y)) / len(x)
    return {"comparison": label, "n": len(x), "kappa": k, "exact_agreement": exact, "type": kind}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    d = Path(args.dir)
    turnsPath = Path(args.turns) if args.turns else next(
        p for p in (ROOT / "results/turns_judged.csv", ROOT / "results/turns.csv") if p.exists())
    turns = pd.read_csv(turnsPath)
    key = pd.read_csv(d / "key.csv")

    merged = key.merge(readRater(d / "rater1.csv", "r1"), on="blind_id").merge(readRater(d / "rater2.csv", "r2"), on="blind_id")
    auto = turns.loc[merged["row"]].reset_index(drop=True)
    merged["unsafe_auto"] = auto["unsafe"].astype(float).values
    merged["stereotyped_auto"] = auto["stereotyped"].astype(float).values
    hasJudge = "empathy" in auto.columns
    if hasJudge:
        merged["empathy_judge"] = auto["empathy"].values
        merged["cultural_judge"] = auto["cultural"].values

    rows = [
        kappaRow("rater1 vs rater2: empathy (0-6)", merged["empathy_r1"], merged["empathy_r2"], "weighted", 0, 6),
        kappaRow("rater1 vs rater2: cultural (1-5)", merged["cultural_r1"], merged["cultural_r2"], "weighted", 1, 5),
        kappaRow("rater1 vs rater2: unsafe", merged["unsafe_r1"], merged["unsafe_r2"], "cohen"),
        kappaRow("rater1 vs rater2: stereotyped", merged["stereotyped_r1"], merged["stereotyped_r2"], "cohen"),
    ]
    for r in ("r1", "r2"):
        rows.append(kappaRow(f"{r} vs automatic check: unsafe", merged[f"unsafe_{r}"], merged["unsafe_auto"], "cohen"))
        rows.append(kappaRow(f"{r} vs automatic check: stereotyped", merged[f"stereotyped_{r}"], merged["stereotyped_auto"], "cohen"))
        if hasJudge:
            rows.append(kappaRow(f"{r} vs judge: empathy (0-6)", merged[f"empathy_{r}"], merged["empathy_judge"], "weighted", 0, 6))
            rows.append(kappaRow(f"{r} vs judge: cultural (1-5)", merged[f"cultural_{r}"], merged["cultural_judge"], "weighted", 1, 5))
    agreement = pd.DataFrame(rows)
    agreement.to_csv(d / "agreement.csv", index=False, encoding="utf-8-sig")

    perCondition = []
    for condition, g in merged.groupby("condition"):
        rec = {"condition": condition, "n": len(g)}
        for metric in ("empathy", "cultural", "unsafe", "stereotyped"):
            mean, lo, hi = bootstrapCi(((g[f"{metric}_r1"] + g[f"{metric}_r2"]) / 2).tolist())
            rec[f"human_{metric}"] = mean
            rec[f"human_{metric}_ci"] = "" if mean != mean else f"[{lo:.2f}, {hi:.2f}]"
        perCondition.append(rec)
    pd.DataFrame(perCondition).to_csv(d / "human_by_condition.csv", index=False, encoding="utf-8-sig")

    print(agreement.to_string(index=False))
    print()
    print(pd.DataFrame(perCondition).to_string(index=False))
    log.info("Kappa guide (Landis & Koch): <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.")


if __name__ == "__main__":
    main()
