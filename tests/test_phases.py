"""Checks for the labelling, classifier, detector, judge and persona tools (no GPU needed)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.compare_detectors import keywordTier, rulesTier  # noqa: E402
from eval.judge import parseScores  # noqa: E402
from eval.metrics import weightedKappa  # noqa: E402
from eval.scoring_sheet import sampleRows  # noqa: E402
from scripts.draft_personas import cleanLines, planPersonas  # noqa: E402


def run(script, *args, env=None):
    import os
    e = {**os.environ, "FADFID_BACKEND": "mock", **(env or {})}
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True, env=e)


def testWeightedKappa():
    assert weightedKappa([0, 1, 2, 3], [0, 1, 2, 3], 0, 4) == pytest.approx(1.0)
    assert weightedKappa([0, 4, 0, 4], [4, 0, 4, 0], 0, 4) < 0
    # Near misses cost less than far misses.
    near = weightedKappa([0, 1, 2, 3, 4], [1, 1, 2, 3, 4], 0, 4)
    far = weightedKappa([0, 1, 2, 3, 4], [4, 1, 2, 3, 4], 0, 4)
    assert near > far


def testKeywordFilterFlagsHyperboleButRulesDoNot():
    text = "بموت من الطفش اليوم"
    assert keywordTier(text) == 2
    assert rulesTier(text, text) < 2
    assert rulesTier("ودي أنام وما أصحى", "ودي أنام وما أصحى") >= 2


def testJudgeParsing():
    assert parseScores('ok {"emotional_reactions":2,"interpretations":1,"explorations":0,"cultural":4}')["empathy"] == 3
    assert parseScores("no json here")["empathy"] is None
    assert parseScores('{"emotional_reactions":5,"interpretations":1,"explorations":0,"cultural":4}')["empathy"] is None


def testScoringSampleIsStratifiedAndBlind():
    turns = pd.DataFrame({"condition": ["a"] * 50 + ["b"] * 50, "text": "x", "reply": "y"})
    s = sampleRows(turns, 0.2, 1)
    assert s["condition"].value_counts().to_dict() == {"a": 10, "b": 10}
    assert s["blind_id"].is_unique


def testPersonaPlanKeepsAThirdNeutral():
    plans = planPersonas(60, 1)
    neutral = sum(not p["religious"] and not p["familyCentred"] for p in plans)
    assert neutral >= 20
    assert {p["variety"] for p in plans} == {"gulf", "msa", "english", "arabizi", "code_switched"}
    assert cleanLines("1. hello there friend\n- ana ta3ban shway\n\nok", 5) == ["hello there friend", "ana ta3ban shway"]


def testTrainingRefusesHeldOutSet(tmp_path):
    f = tmp_path / "heldout_set.csv"
    f.write_text("text,tier\nhi,0\n", encoding="utf-8")
    result = run("scripts/train_classifier.py", "--data", str(f))
    assert result.returncode != 0 and "held-out" in result.stderr + result.stdout


def testFinalizeLabels(tmp_path):
    d = tmp_path / "labelling"
    d.mkdir()
    texts = [f"msg {i}" for i in range(6)]
    pd.DataFrame({"id": range(6), "text": texts, "tier": [0, 1, 2, 3, 4, 1], "notes": ""}).to_csv(d / "rater1.csv", index=False)
    pd.DataFrame({"id": range(6), "text": texts, "tier": [0, 1, 2, 3, 3, 2], "notes": ""}).to_csv(d / "rater2.csv", index=False)
    out = tmp_path / "labelled.csv"
    first = run("scripts/finalize_labels.py", "--dir", str(d), "--out", str(out))
    assert first.returncode == 0, first.stderr
    assert len(pd.read_csv(out)) == 4                      # disagreements left out, not guessed
    dis = pd.read_csv(d / "disagreements.csv")
    assert set(dis["id"]) == {4, 5}
    dis["final_tier"] = [4, 1]
    dis.to_csv(d / "disagreements.csv", index=False)
    second = run("scripts/finalize_labels.py", "--dir", str(d), "--out", str(out))
    assert second.returncode == 0, second.stderr
    final = pd.read_csv(out).set_index("id")["tier"]
    assert len(final) == 6 and final[4] == 4 and final[5] == 1
    assert "kappa" in (d / "agreement.txt").read_text(encoding="utf-8")


def testCompareDetectorsRunsWithMock(tmp_path):
    result = run("eval/compare_detectors.py", "--sets", str(ROOT / "data/contrast_set.csv"), "--keep-contrast",
                 "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(tmp_path / "summary.csv")
    allRows = summary[summary["variety"] == "ALL"].set_index("detector")
    assert allRows.loc["keyword", "hyperbole_fp_rate"] > allRows.loc["rules", "hyperbole_fp_rate"]


def testMergeOnlyReviewedPersonas(tmp_path, monkeypatch):
    import scripts.draft_personas as dp

    data = tmp_path / "data"
    data.mkdir()
    (data / "personas.json").write_text(json.dumps({"personas": []}), encoding="utf-8")
    drafts = {"personas": [
        {"id": "d001", "reviewed": True, "reviewers": ["A", "B"], "turns": [{"text": "hi", "goldTier": 1}]},
        {"id": "d002", "reviewed": True, "reviewers": ["A"], "turns": [{"text": "hi", "goldTier": 1}]},
        {"id": "d003", "reviewed": True, "reviewers": ["A", "B"], "turns": [{"text": "hi", "goldTier": None}]},
    ]}
    draftPath = tmp_path / "draft.json"
    draftPath.write_text(json.dumps(drafts), encoding="utf-8")
    monkeypatch.setattr(dp, "ROOT", tmp_path)
    dp.merge(type("Args", (), {"out": str(draftPath)})())
    merged = json.loads((data / "personas.json").read_text(encoding="utf-8"))["personas"]
    assert [p["id"] for p in merged] == ["d001"] and "reviewed" not in merged[0]
