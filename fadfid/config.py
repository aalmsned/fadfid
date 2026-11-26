"""
Fadfid configuration.

All settings live here. The backend can be switched with the FADFID_BACKEND
environment variable ("fanar" for the real model, "mock" for testing without a GPU).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    backend: str = field(default_factory=lambda: os.environ.get("FADFID_BACKEND", "mock"))
    modelId: str = "QCRI/Fanar-1-9B-Instruct"
    # 4-bit fits ~7 GB GPUs; set FADFID_4BIT=0 for full quality on large GPUs (e.g. 32 GB).
    load4bit: bool = field(default_factory=lambda: os.environ.get("FADFID_4BIT", "1") != "0")
    maxNewTokens: int = 300
    temperature: float = 0.6

    classifierDir: Path = ROOT / "models" / "crisis_classifier"
    resourceFile: Path = ROOT / "data" / "resources.json"
    escalationLog: Path = ROOT / "logs" / "escalations.jsonl"
    interventionLog: Path = ROOT / "logs" / "interventions.jsonl"

    sessionTimeoutMin: int = 30
    trendWindow: int = 5

    # Evaluation conditions switch these off.
    useOversight: bool = True
    useTracker: bool = True
