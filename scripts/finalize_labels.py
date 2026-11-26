"""
==============================================================================
 Combine two raters' labels, report agreement, and write the final labels
==============================================================================
 Reads data/labelling/rater1.csv and rater2.csv, reports Cohen's kappa
 (exact tier, quadratic-weighted, and crisis yes/no), and writes:

   disagreements.csv   rows where the raters differ, with a blank final_tier
                       column (and Fanar's suggestion, if it exists)
   agreement.txt       the agreement figures for the report

 Run it again after the team fills final_tier in disagreements.csv: agreed
 rows and resolved rows go to the final labelled file. Unresolved rows are
 left out, never guessed.

 Usage:
   python scripts/finalize_labels.py
   python scripts/finalize_labels.py --out data/shifaa_labelled.csv --variety msa
==============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import cohenKappa, weightedKappa  # noqa: E402

log = logging.getLogger("finalize_labels")


def buildArgs():
    parser = argparse.ArgumentParser(description="Merge rater labels and report agreement.")
    parser.add_argument("--dir", default=str(ROOT / "data/labelling"))
    parser.add_argument("--rater-a", default="rater1.csv")
    parser.add_argument("--rater-b", default="rater2.csv")
    parser.add_argument("--out", default=str(ROOT / "data/shifaa_labelled.csv"))
    parser.add_argument("--variety", default="msa", help="Language variety for these rows (Shifaa is mostly MSA).")
    return parser.parse_args()


def readSheet(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    df = pd.read_csv(path)
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce")
    bad = df["tier"].notna() & ~df["tier"].isin([0, 1, 2, 3, 4])
    if bad.any():
        raise SystemExit(f"{path}: tiers must be 0-4. Check ids {df.loc[bad, 'id'].tolist()[:10]}")
    missing = df["tier"].isna().sum()
    if missing:
        log.warning("%s: %d rows have no tier yet; they are skipped.", path.name, missing)
    return df.rename(columns={"tier": f"tier_{name}", "notes": f"notes_{name}"})


def agreementReport(merged: pd.DataFrame) -> dict:
    a = merged["tier_a"].astype(int).tolist()
    b = merged["tier_b"].astype(int).tolist()
    return {
        "rows labelled by both": len(merged),
        "exact agreement": sum(x == y for x, y in zip(a, b)) / len(a),
        "Cohen's kappa (exact tier)": cohenKappa(a, b),
        "Quadratic-weighted kappa (tier)": weightedKappa(a, b, 0, 4),
        "Cohen's kappa (crisis yes/no, tier >= 2)": cohenKappa([x >= 2 for x in a], [y >= 2 for y in b]),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    d = Path(args.dir)
    a = readSheet(d / args.rater_a, "a")
    b = readSheet(d / args.rater_b, "b")
    merged = a.merge(b[["id", "tier_b", "notes_b"]], on="id").dropna(subset=["tier_a", "tier_b"])
    if merged.empty:
        raise SystemExit("No rows are labelled by both raters yet.")

    report = agreementReport(merged)
    lines = [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in report.items()]
    (d / "agreement.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        log.info(line)

    agreed = merged[merged["tier_a"] == merged["tier_b"]].assign(tier=lambda x: x["tier_a"].astype(int))
    differ = merged[merged["tier_a"] != merged["tier_b"]]

    disPath = d / "disagreements.csv"
    resolved = pd.DataFrame(columns=["id", "tier"])
    if disPath.exists():
        old = pd.read_csv(disPath)
        old["final_tier"] = pd.to_numeric(old.get("final_tier"), errors="coerce")
        resolved = old.dropna(subset=["final_tier"])[["id", "final_tier"]].rename(columns={"final_tier": "tier"})
        resolved = resolved[resolved["id"].isin(differ["id"])]
    else:
        out = differ[["id", "text", "tier_a", "tier_b", "notes_a", "notes_b"]].copy()
        sugg = d / "fanar_suggestions.csv"
        if sugg.exists():
            out = out.merge(pd.read_csv(sugg), on="id", how="left")
        out["final_tier"] = ""
        out.to_csv(disPath, index=False, encoding="utf-8-sig")
        log.info("Wrote %d disagreements to %s. Discuss them together and fill final_tier.", len(out), disPath)

    resolvedRows = differ.merge(resolved, on="id")
    final = pd.concat([agreed[["id", "text", "tier"]], resolvedRows[["id", "text", "tier"]]], ignore_index=True)
    final["tier"] = final["tier"].astype(int)
    final["variety"] = args.variety
    final.to_csv(args.out, index=False, encoding="utf-8-sig")

    unresolved = len(differ) - len(resolvedRows)
    log.info("Wrote %d final labels to %s (%d agreed, %d resolved, %d still unresolved).",
             len(final), args.out, len(agreed), len(resolvedRows), unresolved)
    log.info("Tiers:\n%s", final["tier"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
