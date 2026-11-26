"""
==============================================================================
 Make blind labelling sheets for the crisis classifier
==============================================================================
 Takes the de-identified Shifaa sample (and any other CSV with id, text) and
 writes one sheet per rater. Raters see only the text: no starting tier, no
 Shifaa label and no Fanar suggestion, so their labels stay independent.

 With --suggest, Fanar also labels every row with the safety agent's rubric.
 Those suggestions go to a separate file. Use them only to discuss rows the
 two raters disagree on, never while labelling.

 Usage:
   python scripts/make_label_sheets.py
   python scripts/make_label_sheets.py --input data/shifaa_sample.csv --raters 2 --suggest

 Output (data/labelling/):
   rater1.csv, rater2.csv     id, text, tier (blank), notes (blank)
   fanar_suggestions.csv      id, fanar_tier            (only with --suggest)
   GUIDE.md                   tier definitions with examples
==============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("make_label_sheets")

GUIDE = """# Crisis tier labelling guide

Label each message on its own, from the text alone. Write one number in the `tier` column.
If you are unsure between two tiers, choose the higher one and explain in `notes`.
Do not discuss rows with the other rater until both sheets are finished.

| Tier | Meaning | Examples |
| --- | --- | --- |
| 0 | No risk, including everyday hyperbole | "بموت من الطفش", "I'm dying of boredom", a question about sleep habits |
| 1 | Distress with no thoughts of death | "تعبت من كل شي ومحد فاهمني", sadness, anxiety, stress, hopelessness |
| 2 | Passive ideation: wishes to be dead, no thought of acting | "ودي أنام وما أصحى", "ياليتني ما انولدت" |
| 3 | Active ideation: thinks of ending their life, no plan or time | "أفكر أنهي حياتي", mentions suicide as something they consider |
| 4 | Imminent: a plan, means, intent or time | "خلاص قررت، الليلة", "جهزت الحبوب", a goodbye note |

Based on the Columbia-Suicide Severity Rating Scale (Posner et al., 2011).
Past attempts with no current thoughts are tier 1 unless the message shows current ideation.
A question about someone else (a friend, a relative) is labelled by the risk it describes for that person.
"""


def buildArgs():
    parser = argparse.ArgumentParser(description="Write blind labelling sheets for two or more raters.")
    parser.add_argument("--input", default=str(ROOT / "data/shifaa_sample.csv"))
    parser.add_argument("--out", default=str(ROOT / "data/labelling"))
    parser.add_argument("--raters", type=int, default=2)
    parser.add_argument("--suggest", action="store_true", help="Also store Fanar's tier for each row (needs GPU).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fanarSuggestions(texts: list[str]) -> list[int]:
    from fadfid.agents.safety import SafetyAgent
    from fadfid.config import Config
    from fadfid.llm import getBackend

    cfg = Config()
    if cfg.backend != "fanar":
        log.warning("FADFID_BACKEND is '%s'; suggestions will not come from Fanar.", cfg.backend)
    agent = SafetyAgent(getBackend(cfg), classifier=None, tracker=None)
    out = []
    for i, text in enumerate(texts, 1):
        out.append(agent.llmTier([], text))
        if i % 100 == 0:
            log.info("Suggested %d of %d", i, len(texts))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"{src} not found. Run scripts/prepare_shifaa.py first.")

    df = pd.read_csv(src)
    if "id" not in df.columns:
        df.insert(0, "id", [f"{src.stem}_{i:05d}" for i in range(len(df))])
    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)

    for r in range(1, args.raters + 1):
        sheet = df[["id", "text"]].sample(frac=1, random_state=args.seed + r).reset_index(drop=True)
        sheet["tier"] = ""
        sheet["notes"] = ""
        path = outDir / f"rater{r}.csv"
        if path.exists():
            raise SystemExit(f"{path} already exists. Move it first so no one's labels are overwritten.")
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
        log.info("Wrote %s (%d rows)", path, len(sheet))

    (outDir / "GUIDE.md").write_text(GUIDE, encoding="utf-8")

    if args.suggest:
        sugg = pd.DataFrame({"id": df["id"], "fanar_tier": fanarSuggestions(df["text"].astype(str).tolist())})
        sugg.to_csv(outDir / "fanar_suggestions.csv", index=False, encoding="utf-8-sig")
        log.info("Wrote Fanar suggestions. Keep this file away from the raters until both sheets are done.")

    log.info("Next: each rater fills the tier column in their own sheet, then run scripts/finalize_labels.py.")


if __name__ == "__main__":
    main()
