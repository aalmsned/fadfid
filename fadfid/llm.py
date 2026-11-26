"""
LLM backends.

Every agent shares one Fanar instance and differs only in its role prompt,
tools and permissions. Fanar runs locally; nothing is sent to a hosted API.
"""

import logging

from .config import Config

log = logging.getLogger(__name__)


class LlmBackend:
    def generate(self, systemPrompt: str, messages: list[dict], temperature: float | None = None) -> str:
        raise NotImplementedError


class FanarBackend(LlmBackend):
    """Fanar-1-9B-Instruct via Hugging Face Transformers, full precision or 4-bit."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        kwargs = {"device_map": "auto"}
        if cfg.load4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            kwargs["dtype"] = torch.bfloat16

        log.info("Loading %s (4-bit=%s)", cfg.modelId, cfg.load4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.modelId)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.modelId, **kwargs)
        except TypeError:
            # Older Transformers versions call it torch_dtype.
            kwargs["torch_dtype"] = kwargs.pop("dtype", torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(cfg.modelId, **kwargs)
        self.model.eval()

    def generate(self, systemPrompt: str, messages: list[dict], temperature: float | None = None) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        turns = mergeRoles(messages)
        # Templates expect the first turn to be the user's: fold leading context into the system prompt.
        while turns and turns[0]["role"] == "assistant":
            systemPrompt += "\n\nContext from earlier: " + turns.pop(0)["content"]
        chat = [{"role": "system", "content": systemPrompt}] + turns
        inputs = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        genKwargs = {"max_new_tokens": self.cfg.maxNewTokens, "do_sample": temp > 0,
                     "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if temp > 0:
            genKwargs["temperature"] = temp
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **genKwargs)
        promptLen = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][promptLen:], skip_special_tokens=True).strip()


def mergeRoles(messages: list[dict]) -> list[dict]:
    """Chat templates require alternating user/assistant turns; merge consecutive same-role messages."""
    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


class MockBackend(LlmBackend):
    """Deterministic stand-in so the pipeline runs without a GPU. Not for results."""

    def generate(self, systemPrompt: str, messages: list[dict], temperature: float | None = None) -> str:
        if "TIER:" in systemPrompt:
            return "TIER: -1"
        if "HARM:" in systemPrompt:
            return "HARM: UNKNOWN"
        if "REWRITE" in systemPrompt:
            return messages[-1]["content"]
        if "SUMMARY" in systemPrompt:
            return "Talked about feeling under pressure lately; wanted to continue next time."
        if "Reply in English" in systemPrompt:
            return "I'm here with you. Can you tell me a bit more about what you're feeling?"
        return "أنا هنا معك. تقدر تحكيلي أكثر عن اللي تحس فيه؟"


def getBackend(cfg: Config) -> LlmBackend:
    if cfg.backend == "fanar":
        return FanarBackend(cfg)
    if cfg.backend == "mock":
        log.warning("Using the MOCK backend: fixed replies, not Fanar. Set FADFID_BACKEND=fanar for the real model.")
        return MockBackend()
    raise ValueError(f"Unknown FADFID_BACKEND '{cfg.backend}'. Use 'fanar' or 'mock'.")
