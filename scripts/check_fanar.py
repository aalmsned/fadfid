"""
==============================================================================
 Check that Fanar loads and answers on this machine
==============================================================================
 Loads Fanar-1-9B-Instruct, reports GPU memory and load time, then sends a
 few test messages through the safety rubric and the listener prompt, and
 times each reply. Run this once before chatting or evaluating.

 Usage:
   FADFID_4BIT=0 python scripts/check_fanar.py      # full quality (needs ~18 GB GPU memory)
   python scripts/check_fanar.py                    # 4-bit (needs ~7 GB)
==============================================================================
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fadfid.agents.listener import LISTENER_PROMPT  # noqa: E402
from fadfid.agents.safety import TIER_RUBRIC  # noqa: E402
from fadfid.config import Config  # noqa: E402
from fadfid.llm import FanarBackend  # noqa: E402

log = logging.getLogger("check_fanar")

TEST_MESSAGES = [
    ("gulf", "بموت من الطفش اليوم"),
    ("gulf", "تعبت من كل شي ومحد فاهمني"),
    ("gulf", "ودي أنام وما أصحى"),
    ("english", "I feel so lonely since I moved here"),
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch. Run nvidia-smi and check the PyTorch install.")
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    cfg = Config(backend="fanar")
    start = time.time()
    llm = FanarBackend(cfg)
    log.info("Loaded in %.0f s (4-bit=%s). GPU memory in use: %.1f GB",
             time.time() - start, cfg.load4bit, torch.cuda.memory_allocated() / 1e9)

    for variety, text in TEST_MESSAGES:
        t0 = time.time()
        tier = llm.generate(TIER_RUBRIC, [{"role": "user", "content": text}], temperature=0)
        instruction = "Reply in English." if variety == "english" else "Reply in Gulf Arabic (Khaleeji), warm and simple."
        reply = llm.generate(LISTENER_PROMPT.format(replyInstruction=instruction, checkIn=""), [{"role": "user", "content": text}])
        print(f"\nUser:   {text}\nRubric: {tier}\nReply:  {reply}\n({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
