# Writing the held-out test set

This set is the fair test of crisis detection. It must be **new**: written by team members who
have **not** read `fadfid/agents/safety.py` or `data/contrast_set.csv`, and never used for training.

1. Copy `heldout_template.csv` to `heldout_set.csv`. Each writer adds rows with their own initials in `author`.
2. Aim for at least **100 hyperbole**, **100 crisis** (tiers 2-4) and **50 messages in each language variety**.
3. Write the way people really type: typos, emojis, short messages, slang, mixed languages.
4. Do not copy or lightly reword examples from the contrast set, the slides or the labelling guide.

| Column | Values |
| --- | --- |
| id | h0001, h0002, ... |
| text | the message |
| tier | 0-4, as in `data/labelling/GUIDE.md` |
| variety | gulf, msa, english, arabizi, code_switched |
| kind | hyperbole, crisis, distress, neutral, ambiguous |
| author | writer's initials |

A second team member checks every tier. Where you disagree, discuss and agree on one tier.

Then run:

    python eval/compare_detectors.py --sets data/heldout_set.csv data/classifier_test.csv
    python eval/run_eval.py --backend fanar --extra data/heldout_set.csv
