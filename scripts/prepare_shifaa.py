"""
==============================================================================
 Prepare Shifaa: sample, de-identify, count religious references
==============================================================================
 Downloads Shifaa Arabic Mental Health Consultations, keeps only the questions,
 samples about 2,000 (all suicidality-labelled ones first), removes identifying
 details, flags every row for manual review by a second member, and reports
 how many questions contain explicit religious references.

 Usage:
   python scripts/prepare_shifaa.py --sample 2000 --out data/shifaa_sample.csv
   python scripts/prepare_shifaa.py --paraphrase   # also paraphrase with Fanar (GPU)

 Output columns:
   id, text, label, suicidality, tier, has_religious_ref, needs_review
 The answers are dropped: they contain diagnoses, medication advice and
 religious framing that Fadfid must not copy.
==============================================================================
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("prepare_shifaa")

DATASET_ID = "Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations"

RELIGIOUS_LEXICON = [
    "الله", "ربي", "الرب", "الصلاة", "صلاة", "صلاتي", "اصلي", "أصلي", "القرآن", "قرآن", "الرقية", "رقية", "الدعاء",
    "دعاء", "حرام", "حلال", "الحسد", "حسد", "العين", "السحر", "سحر", "الجن", "جن", "الشيطان", "شيطان", "ذنب",
    "ذنوب", "الإيمان", "ايمان", "إيمان", "التوبة", "توبة", "المسجد", "مسجد", "الصيام", "صيام", "الحج", "استغفار",
    "وسوسة", "الوسواس القهري في العبادات", "الدين", "ديني", "عبادة", "العبادة",
]
SUICIDALITY_TERMS = ["انتحار", "انتحاري", "الانتحار", "إيذاء النفس", "ايذاء النفس", "suicid", "self-harm", "self harm"]

PLACES = [
    "قطر", "الدوحة", "الخور", "الوكرة", "الريان", "السعودية", "الرياض", "جدة", "مكة", "المدينة", "الدمام", "الكويت",
    "الإمارات", "الامارات", "دبي", "أبوظبي", "ابوظبي", "الشارقة", "البحرين", "المنامة", "عمان", "مسقط", "مصر", "القاهرة",
    "الإسكندرية", "الأردن", "عمّان", "سوريا", "دمشق", "لبنان", "بيروت", "العراق", "بغداد", "اليمن", "صنعاء", "ليبيا",
    "تونس", "الجزائر", "المغرب", "فلسطين", "غزة", "السودان", "الخرطوم", "تركيا", "أمريكا", "امريكا", "بريطانيا", "لندن",
]
NATIONALITIES = [
    "قطري", "قطرية", "سعودي", "سعودية", "كويتي", "كويتية", "إماراتي", "اماراتي", "بحريني", "عماني", "مصري", "مصرية",
    "أردني", "سوري", "سورية", "لبناني", "عراقي", "عراقية", "يمني", "ليبي", "تونسي", "تونسية", "جزائري", "مغربي",
    "مغربية", "فلسطيني", "فلسطينية", "سوداني", "سودانية",
]
OCCUPATIONS = [
    "طبيب", "طبيبة", "مهندس", "مهندسة", "معلم", "معلمة", "مدرس", "مدرسة", "ممرض", "ممرضة", "محامي", "محامية",
    "ضابط", "عسكري", "إمام", "امام", "خطيب", "طيار", "محاسب", "محاسبة", "موظف في", "أعمل في", "اعمل في", "أشتغل في",
]

DEID_RULES = [
    (r"\S+@\S+\.\S+", "[بريد]"),
    (r"(https?://\S+|www\.\S+)", "[رابط]"),
    (r"\+?\d[\d\s\-]{6,}\d", "[رقم]"),
    (r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b", "[تاريخ]"),
    (r"\b(19|20)\d{2}\b", "[سنة]"),
    (r"(عمري|عمرى|أبلغ من العمر|ابلغ من العمر|سني)\s*\d{1,2}", "عمري [عمر]"),
    (r"\b\d{1,2}\s*(سنة|سنه|عاما|عاماً|عام)\b", "[عمر]"),
    (r"(اسمي|إسمي|انا اسمي|أنا اسمي)\s+\S+(\s+\S+)?", "اسمي [اسم]"),
    (r"(أخي|اخي|أختي|اختي|زوجي|زوجتي|ابني|بنتي|ابنتي|صديقي|صديقتي)\s+(اسمه|اسمها)\s+\S+", r"\1"),
]


def wordPattern(words: list[str]) -> re.Pattern:
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)")


PLACE_RE = wordPattern(PLACES)
NATIONALITY_RE = wordPattern(NATIONALITIES)
OCCUPATION_RE = wordPattern(OCCUPATIONS)
RELIGIOUS_RE = wordPattern(RELIGIOUS_LEXICON)


def deidentify(text: str, ner=None) -> str:
    for pattern, repl in DEID_RULES:
        text = re.sub(pattern, repl, text)
    text = PLACE_RE.sub("[مكان]", text)
    text = NATIONALITY_RE.sub("[جنسية]", text)
    text = OCCUPATION_RE.sub("[مهنة]", text)
    if ner is not None:
        text = applyNer(text, ner)
    return re.sub(r"\s+", " ", text).strip()


def applyNer(text: str, ner) -> str:
    """CAMeL Tools NER: replace person, location and organisation entities."""
    tokens = text.split()
    labels = ner.predict_sentence(tokens)
    replaced = []
    for tok, lab in zip(tokens, labels):
        if lab.endswith("PERS"):
            replaced.append("[اسم]")
        elif lab.endswith("LOC"):
            replaced.append("[مكان]")
        elif lab.endswith("ORG"):
            replaced.append("[جهة]")
        else:
            replaced.append(tok)
    return " ".join(replaced)


def loadNer():
    try:
        from camel_tools.ner import NERecognizer
        return NERecognizer.pretrained()
    except Exception:
        log.warning("CAMeL Tools NER not available; using rule-based de-identification only.")
        return None


def pickColumn(columns, candidates, kind):
    for c in candidates:
        for col in columns:
            if col.lower() == c.lower():
                return col
    for c in candidates:
        for col in columns:
            if c.lower() in col.lower():
                return col
    raise SystemExit(f"Could not find a {kind} column. Columns are: {list(columns)}. Pass --{kind}-col.")


def isMinorPost(text: str) -> bool:
    return bool(re.search(r"(عمري|عمرى)\s*(1[0-7])\b|\b(1[0-7])\s*(سنة|سنه|عاما|عام)\b|(الصف|طالب في|طالبة في)\s*(ال)?(متوسط|ثانوي|إعدادي|اعدادي)", text))


def buildArgs():
    parser = argparse.ArgumentParser(description="Sample and de-identify Shifaa questions.")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--out", default="data/shifaa_sample.csv")
    parser.add_argument("--question-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paraphrase", action="store_true", help="Paraphrase each question with Fanar (needs GPU).")
    return parser.parse_args()


def loadShifaa(args) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    log.info("Loaded %d rows. Columns: %s", len(df), list(df.columns))

    qCol = args.question_col or pickColumn(df.columns, ["question", "السؤال", "query", "text", "title"], "question")
    labelCols = [args.label_col] if args.label_col else [
        c for c in df.columns if any(k in c.lower() for k in ["label", "diagnos", "category", "class", "تشخيص", "تصنيف"])
    ]
    log.info("Question column: %s | label columns: %s", qCol, labelCols)

    out = pd.DataFrame({"text": df[qCol].astype(str)})
    out["label"] = df[labelCols].astype(str).agg(" | ".join, axis=1) if labelCols else ""
    return out


def paraphraseAll(texts: list[str]) -> list[str]:
    from fadfid.config import Config
    from fadfid.llm import FanarBackend

    cfg = Config(backend="fanar")
    llm = FanarBackend(cfg)
    prompt = ("Paraphrase the following Arabic text so it keeps the same meaning and emotion but uses "
              "different wording. Do not add details. Return only the paraphrase.")
    return [llm.generate(prompt, [{"role": "user", "content": t}]) for t in texts]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()

    df = loadShifaa(args)
    df = df[df["text"].str.len() > 20].drop_duplicates("text")

    suicidalityRe = re.compile("|".join(SUICIDALITY_TERMS), re.IGNORECASE)
    df["suicidality"] = df["label"].str.contains(suicidalityRe) | df["text"].str.contains(suicidalityRe)
    log.info("Suicidality-labelled questions: %d of %d", int(df["suicidality"].sum()), len(df))

    df["has_religious_ref"] = df["text"].apply(lambda t: bool(RELIGIOUS_RE.search(t)))
    share = df["has_religious_ref"].mean() * 100
    log.info("Questions with an explicit religious reference (lexicon): %.1f%% of %d", share, len(df))

    df = df[~df["text"].apply(isMinorPost)]
    crisis = df[df["suicidality"]]
    rest = df[~df["suicidality"]].sample(n=max(args.sample - len(crisis), 0), random_state=args.seed)
    sample = pd.concat([crisis, rest]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    ner = loadNer()
    sample["text"] = sample["text"].apply(lambda t: deidentify(t, ner))
    if args.paraphrase:
        sample["text"] = paraphraseAll(sample["text"].tolist())

    # Starting tiers: suicidality -> 3 (refine to 2/3/4 by hand), everything else -> 1 (distress).
    sample["tier"] = sample["suicidality"].map({True: 3, False: 1})
    sample["needs_review"] = True
    sample.insert(0, "id", [f"shifaa_{i:05d}" for i in range(len(sample))])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False, encoding="utf-8-sig")

    religiousManual = sample.sample(n=min(200, len(sample)), random_state=args.seed)[["id", "text", "has_religious_ref"]]
    religiousManual.to_csv(Path(args.out).with_name("religious_check_200.csv"), index=False, encoding="utf-8-sig")

    log.info("Wrote %d rows to %s", len(sample), args.out)
    log.info("Sample religious share: %.1f%%", sample["has_religious_ref"].mean() * 100)
    log.info("Next: two members review every row (needs_review) and the 200-row religious check file.")


if __name__ == "__main__":
    main()
