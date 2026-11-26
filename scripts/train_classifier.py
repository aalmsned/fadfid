"""
==============================================================================
 Train the crisis-tier classifier (MARBERT)
==============================================================================
 Fine-tunes UBC-NLP/MARBERTv2 to predict tiers 0-4 from:
   - data/shifaa_labelled.csv  (two-rater labels from scripts/finalize_labels.py)
   - data/contrast_set.csv     (Gulf hyperbole vs. crisis; gives tier-0 examples
                                that Shifaa does not have)
 Arabizi rows are transliterated to Arabic script first, the same way the
 language agent does at run time.

 The data is split 70/15/15. The 15% test split is saved to
 data/classifier_test.csv and scored once at the end; it is never used for
 training or for choosing a checkpoint. Tiers are weighted by how rare they
 are, because crisis rows are far fewer than distress rows.

 The held-out evaluation set (data/heldout_set.csv) is refused here: it is
 for testing only.

 Usage:
   python scripts/train_classifier.py --epochs 4
   python scripts/train_classifier.py --data data/shifaa_labelled.csv data/contrast_set.csv data/extra.csv

 Output:
   models/crisis_classifier/     the safety agent loads it automatically
   data/classifier_test.csv      the untouched test split
   results/classifier_test.json  test metrics
==============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fadfid.agents.language import LanguageAgent  # noqa: E402

log = logging.getLogger("train_classifier")
MODEL_ID = "UBC-NLP/MARBERTv2"
NUM_TIERS = 5
FORBIDDEN = ("heldout",)


def buildArgs():
    parser = argparse.ArgumentParser(description="Fine-tune MARBERT on crisis tiers.")
    parser.add_argument("--data", nargs="+", default=[str(ROOT / "data/shifaa_labelled.csv"), str(ROOT / "data/contrast_set.csv")])
    parser.add_argument("--out", default=str(ROOT / "models/crisis_classifier"))
    parser.add_argument("--test-out", default=str(ROOT / "data/classifier_test.csv"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def loadData(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        name = Path(p).stem
        if any(f in name for f in FORBIDDEN):
            raise SystemExit(f"{p} is a held-out test file and must not be used for training.")
        if name == "shifaa_sample":
            log.warning("shifaa_sample.csv has automatic starting tiers, not reviewed labels. Prefer shifaa_labelled.csv.")
        if not Path(p).exists():
            log.warning("Missing %s, skipping", p)
            continue
        df = pd.read_csv(p)
        keep = [c for c in ("id", "text", "tier", "variety", "kind") if c in df.columns]
        df = df[keep].dropna(subset=["text", "tier"])
        df["source"] = name
        frames.append(df)
    if not frames:
        raise SystemExit("No training data found. Run scripts/finalize_labels.py first.")
    data = pd.concat(frames, ignore_index=True).drop_duplicates("text")
    data["tier"] = data["tier"].astype(int)
    for col in ("variety", "kind"):
        if col not in data.columns:
            data[col] = None
    language = LanguageAgent()
    data["rawText"] = data["text"].astype(str)
    data["text"] = data["rawText"].apply(lambda t: language.run(t)["normalized"])
    log.info("Rows per tier:\n%s", data["tier"].value_counts().sort_index().to_string())
    return data


def splitData(data: pd.DataFrame, seed: int):
    from sklearn.model_selection import train_test_split

    strat = data["tier"] if data["tier"].value_counts().min() >= 3 else (data["tier"] >= 2)
    trainDf, rest = train_test_split(data, test_size=0.30, stratify=strat, random_state=seed)
    restStrat = rest["tier"] if rest["tier"].value_counts().min() >= 2 else (rest["tier"] >= 2)
    valDf, testDf = train_test_split(rest, test_size=0.50, stratify=restStrat, random_state=seed)
    return trainDf, valDf, testDf


def crisisMetrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    crisisTrue, crisisPred = labels >= 2, preds >= 2
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=list(range(NUM_TIERS)), zero_division=0)),
        "crisis_precision": float(precision_score(crisisTrue, crisisPred, zero_division=0)),
        "crisis_recall": float(recall_score(crisisTrue, crisisPred, zero_division=0)),
        "crisis_f1": float(f1_score(crisisTrue, crisisPred, zero_division=0)),
    }


def computeMetrics(evalPred):
    logits, labels = evalPred
    return crisisMetrics(labels, np.argmax(logits, axis=-1))


def classWeights(tiers: pd.Series) -> list[float]:
    counts = tiers.value_counts().reindex(range(NUM_TIERS), fill_value=0)
    total = counts.sum()
    return [float(total / (NUM_TIERS * c)) if c else 0.0 for c in counts]


def train(args, data: pd.DataFrame):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)

    trainDf, valDf, testDf = splitData(data, args.seed)
    Path(args.test_out).parent.mkdir(parents=True, exist_ok=True)
    testDf[["text", "rawText", "tier", "variety", "kind", "source"]].rename(columns={"text": "normalized", "rawText": "text"}) \
        .to_csv(args.test_out, index=False, encoding="utf-8-sig")
    log.info("Split: %d train, %d validation, %d test (saved to %s)", len(trainDf), len(valDf), len(testDf), args.test_out)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    id2label = {i: f"tier_{i}" for i in range(NUM_TIERS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_TIERS, id2label=id2label, label2id={v: k for k, v in id2label.items()}
    )
    weights = torch.tensor(classWeights(trainDf["tier"]), dtype=torch.float)
    log.info("Tier weights: %s", [round(w, 2) for w in weights.tolist()])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def encode(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=256)
        enc["labels"] = batch["tier"]
        return enc

    toDs = lambda df: Dataset.from_pandas(df[["text", "tier"]].reset_index(drop=True)).map(encode, batched=True)
    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(ROOT / "models/checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="crisis_f1",
            seed=args.seed,
            report_to=[],
        ),
        train_dataset=toDs(trainDf),
        eval_dataset=toDs(valDf),
        processing_class=tokenizer,
        compute_metrics=computeMetrics,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    log.info("Saved classifier to %s", args.out)

    pred = trainer.predict(toDs(testDf))
    preds = np.argmax(pred.predictions, axis=-1)
    report = crisisMetrics(testDf["tier"].to_numpy(), preds)
    report["n_test"] = int(len(testDf))
    report["n_test_crisis"] = int((testDf["tier"] >= 2).sum())
    out = ROOT / "results/classifier_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Test split (scored once): %s", report)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    train(args, loadData(args.data))


if __name__ == "__main__":
    main()
