# Fadfid (فضفض) prototype

A seven-agent wellbeing support system for Gulf users: culturally opt-in, confidential by design, and built never to end a conversation because of crisis language.

```
user message
   │
   ▼
Language agent ──► Conversation tracker ──► Safety agent (tier 0–4)
                                               │
                         tier 0–1 ◄────────────┴────────────► tier 2–4
                            │                                   │
                     Listener agent                   Crisis support agent
                            │                         (templates + verified
                  Cultural-adaptation agent             resources, consent,
                   (only if opted in / raised)          counsellor at tier 4)
                            │                                   │
                            └──────────► Oversight agent ◄──────┘
                                   (block / rewrite / escalate)
                                              │
                                            reply
```

Every agent uses the same locally hosted Fanar-1-9B-Instruct as its reasoning model. What makes each one different is its role prompt, its tools and its permissions (`fadfid/agents/`).

## Folder map

| Path | What it is |
| --- | --- |
| `chat.py` | Talk to Fadfid in the terminal |
| `fadfid/pipeline.py` | LangGraph orchestration and the five evaluation conditions |
| `fadfid/agents/` | The seven agents |
| `fadfid/memory_card.py` | Optional encrypted, on-device memory card |
| `data/resources.json` | Verified resource file (999, 16000) and crisis templates. **Re-verify monthly.** |
| `data/contrast_set.csv` | Starter Gulf hyperbole vs. crisis set. **Expand to 100+ of each.** |
| `data/personas.json` | Starter multi-turn personas. **Expand to ~100.** |
| `scripts/prepare_shifaa.py` | Sample, de-identify and religion-count Shifaa |
| `scripts/train_classifier.py` | Fine-tune MARBERT on crisis tiers |
| `eval/run_eval.py` | Five conditions, all metrics per language variety, CIs, McNemar |
| `tests/` | Automated checks (no GPU needed) |

## 1. Set up on Windows

1. Install **Python 3.12** from python.org. Tick "Add python.exe to PATH" during install.
2. Open **PowerShell** in this folder (in File Explorer, Shift + right-click the `Fadfid` folder, then "Open in Terminal"):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install langgraph cryptography pandas numpy pytest
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

3. Check that everything works:

```powershell
python -m pytest tests -q
python chat.py --debug
```

**If Windows says "An Application Control policy has blocked this file"**, Smart App Control or a work policy is blocking compiled packages (pydantic, cryptography, numpy, torch). Fadfid then falls back to a built-in orchestrator, and the memory card is switched off. Run the tests without auto-loaded plugins:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
py -m pytest tests -q
py chat.py --debug
```

The evaluation and the real model need numpy, pandas and torch, so run those on a machine without that policy.

This runs with the **mock backend**: fixed replies that test the plumbing (routing, tiers, oversight, crisis flow) without a GPU. Its replies are not real model output.

## 2. Run the real model (needs an NVIDIA GPU)

Fanar-1-9B in 4-bit needs about **7 GB of GPU memory**. Check yours with `nvidia-smi`. If your PC doesn't have an NVIDIA GPU, run this part on a university GPU machine instead. Don't switch to a hosted API, because that would break policy P4.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login          # only if the Fanar model page asks you to accept its terms
$env:FADFID_BACKEND = "fanar"
python chat.py --debug
```

## 3. Data, labelling and the classifier

Run these in Ubuntu with the venv active and `export FADFID_BACKEND=fanar FADFID_4BIT=0`.

**3a. Sample Shifaa** (computer)
```bash
python scripts/prepare_shifaa.py --sample 2000          # data/shifaa_sample.csv + religious_check_200.csv
```
It prints the **share of questions with a religious reference** (the 34% check). Its starting tiers are automatic guesses, not labels.

**3b. Label it with two raters** (people)
```bash
python scripts/make_label_sheets.py --suggest           # data/labelling/rater1.csv, rater2.csv, GUIDE.md
```
Two team members each fill the `tier` column in their own sheet, **without seeing each other's** or `fanar_suggestions.csv`. Then:
```bash
python scripts/finalize_labels.py                       # kappa + disagreements.csv
```
Discuss each row in `disagreements.csv`, fill `final_tier`, and run `finalize_labels.py` again. It writes `data/shifaa_labelled.csv` and `data/labelling/agreement.txt` (report the kappa).

**3c. Train MARBERT** (computer, a few minutes on the 5090)
```bash
python scripts/train_classifier.py --epochs 4
```
It splits 70/15/15 and keeps the 15% test split untouched in `data/classifier_test.csv`; the score is in `results/classifier_test.json`. The safety agent loads `models/crisis_classifier` automatically from then on.

## 4. Evaluation

**4a. Held-out test set** (people). Team members who have **not** read `safety.py` write new messages: see `data/HELDOUT_GUIDE.md`. This is the only fair test of the rules, because the starter contrast set was used while writing them.

**4b. More personas** (computer, then people)
```bash
python scripts/draft_personas.py --count 60             # data/personas_draft.json
```
Two native speakers edit each draft, fill every `goldTier`, add their names to `reviewers` and set `reviewed: true`. Then `python scripts/draft_personas.py --merge` adds them to `data/personas.json`.

**4c. Classifier vs. LLM-only detection** (objective 3; fills the "first measurements" slide)
```bash
python eval/compare_detectors.py --sets data/heldout_set.csv data/classifier_test.csv
```
Compares keyword filter, our rules, Fanar alone, MARBERT alone, and the deployed combination, per language variety, with McNemar tests. Output in `results/detectors/`.

**4d. Five conditions**
```bash
python eval/run_eval.py --backend fanar --extra data/heldout_set.csv
```
- `turns.csv`: one row per reply.
- `summary.csv`: every metric per condition and language variety, with 95% CIs.
- `mcnemar.csv`: paired tests against the responsible system.

**4e. Empathy and cultural scores** (stop the server first so the GPU is free)
```bash
python eval/judge.py                                    # Qwen2.5-7B-Instruct judges; not Fanar
```
Writes `results/turns_judged.csv` and `results/summary_judged.csv` (empathy 0–6, cultural 1–5).

**4f. Hand-score 20% with two raters** (people)
```bash
python eval/scoring_sheet.py                            # results/scoring/rater1.csv, rater2.csv, GUIDE.md
python eval/agreement.py                                # after both sheets are filled
```
Raters score blind (they don't know which condition wrote a reply). `agreement.py` reports kappa between raters, between raters and the judge, and between raters and the automatic unsafe/stereotyping checks, plus human scores per condition.

## 5. Web demo (open it from your laptop)

The model runs on the PC; the laptop only opens a web page. In Ubuntu on the PC, inside the Fadfid folder:

```bash
source ~/fadfid-venv/bin/activate
export FADFID_BACKEND=fanar FADFID_4BIT=0
python webapp/server.py --port 8000
```

Open `http://localhost:8000` on the PC to check it. To reach it from the laptop anywhere (home, campus), use **Tailscale**, a free private encrypted link between your own devices:

1. On the PC, in Ubuntu: `curl -fsSL https://tailscale.com/install.sh | sh`, then `sudo tailscale up`. Open the link it prints and sign in.
2. On the laptop: install the Tailscale app from tailscale.com and sign in with the **same account**.
3. On the PC, in Ubuntu: `tailscale ip -4` prints an address like `100.x.y.z`.
4. On the laptop's browser: `http://100.x.y.z:8000`.

Keep the PC on and awake during the demo, with the Ubuntu window running the server. Only devices signed in to your Tailscale account can open the page. The chat text travels encrypted between your devices and is never sent to a model provider.

**4g. Harm to others (P8)**
```bash
python eval/harm_eval.py                                # rules vs Fanar vs deployed on data/harm_set.csv
```
`harm_set.csv` is a starter set written alongside the rules, so it is not a fair test; add a held-out one written by teammates.

## Harm to other people (P8)

Fadfid separates four situations and treats each differently:

| What the user says | Label | What Fadfid does |
| --- | --- | --- |
| Angry exaggeration: "والله بذبحه" about a brother who took the car | `venting` | Listener acknowledges the anger and gently asks what they mean. Nothing recorded. |
| Harmed someone before, no current intent | `past` | Listener stays, without judging. Kept confidential. Nothing recorded. |
| Wants to hurt a specific person, or is hurting a child or dependant | `intent`, `ongoing` | De-escalates, asks whether anyone is in danger, urges 999, and offers a licensed counsellor. |
| Intent plus a time, plan or weapon | `imminent` | As above, and the on-call counsellor is alerted to join the chat. |

- **Identity is collected only by the counsellor, only with consent.** The offer says so plainly: the counsellor will ask for a name and contact and may need to act to keep someone safe.
- **A human decides on any warning**, never the AI (Lior, 2026).
- **If the user refuses, Fadfid can't identify them.** That's the cost of being confidential by design, accepted openly.
- **The limit is shown before the chat starts**, on the welcome screen and in the terminal.
- **Oversight blocks** any reply that helps hurt someone or hide harm, and any promise of secrecy.
- After a serious disclosure, the listener keeps gently checking safety for four more messages.
- The escalation log records `kind: harm_to_others` with the same text-free fields.

## What is stored, and where (P3)

| Data | Where | Kept? |
| --- | --- | --- |
| Conversation text | Memory only, during the session | Deleted at `/end` or after 30 minutes idle |
| Tracker flags (faith/family raised, tiers, age cue) | Memory only | Deleted with the session |
| `logs/escalations.jsonl` | Server | Time, tier, decision, kind (self or harm to others), counsellor ID. **No text.** |
| `logs/interventions.jsonl` | Server | Time, policy, action. **No text.** |
| Memory card | User's own device (`~/.fadfid/`), encrypted | Only if the user chooses, off by default |

## Not built yet

- **On-device age estimation.** The terminal prototype only asks users to confirm they are 18+ and watches for age cues. The face-based check belongs in a phone app.
- **The real counsellor rota.** `CounsellorService` in `fadfid/agents/crisis.py` simulates it, including service hours.
- **FanarGuard.** QCRI has published the FanarGuard paper and dataset on Hugging Face, but not the filter model itself. If they release it (or you get access), pass it to `OversightAgent(extraGuard=...)` as a second check.
