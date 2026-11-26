"""
Remove the model's working notes from a reply before the user sees it.

Fanar sometimes wraps a reply in labels or notes meant for us, not the user:
"Draft:", "Revised response:", "Redrafted (considering no specific themes
mentioned):", "Note: ...", "ملاحظة: ...". These are stripped; the reply itself
is left unchanged.
"""

import re

LABEL_WORDS = (r"(draft|redraft(ed)?|revis(ed|ion)|rewrit(ten|e)|edited|updated|improved|final|"
               r"response|reply|answer|version|output|مسودة|المسودة|الرد|الرد المعدل|الرد المعدّل|"
               r"النص المعدل|النسخة المعدلة|الإجابة|معدل)")
# A whole line that is only a label, e.g. "Revised response:" or "**Draft:**"
LABEL_LINE = re.compile(rf"^\W*(here('s| is) (the |my |a )?)?[\w ()'-]{{0,20}}?{LABEL_WORDS}[^\n:]{{0,70}}:\W*$",
                        re.IGNORECASE)
# A label at the start of a line, followed by the real text on the same line
LABEL_PREFIX = re.compile(rf"^\W*{LABEL_WORDS}\b[^\n:]{{0,70}}:[*_\s]*", re.IGNORECASE)
NOTE_LINE = re.compile(r"^\W*(note|notes|explanation|ملاحظة|ملاحظه|تنبيه)\b\W", re.IGNORECASE)


def cleanReply(text: str) -> str:
    if not text:
        return text
    out = []
    for line in text.strip().splitlines():
        if LABEL_LINE.match(line) or NOTE_LINE.match(line):
            continue
        out.append(LABEL_PREFIX.sub("", line, count=1))
    cleaned = "\n".join(out).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) > 2 and cleaned[0] in "\"“«" and cleaned[-1] in "\"”»":
        cleaned = cleaned[1:-1].strip()
    return cleaned or text.strip()
