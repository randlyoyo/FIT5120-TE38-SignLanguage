"""The avatar's side of the conversation: history + what the user said -> one short English reply.

The reply is signed by the text->sign model, which learned from Auslan-Daily's everyday
communication clips and signs short sentences best (long ones stop signing before the end, names
and places have no sign to borrow; EXPERIMENTS.md E33-E34). So every backend is asked for one short
everyday sentence, and `clean_reply` enforces it whatever the model returns.

Backends (config `dialogue.backend`):
  hf         a local Hugging Face chat model (default Qwen2.5-1.5B-Instruct; free, runs on the same GPU)
  openai     any OpenAI-compatible /chat/completions server (vLLM, Ollama, LM Studio, ...)
  anthropic  the Claude API (needs ANTHROPIC_API_KEY)
  echo       no model: the avatar signs back what the user said (a translation mode, and for tests)
"""

from __future__ import annotations

import os
import re

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly avatar in a chat app who talks with Deaf and hard-of-hearing users in Auslan "
    "(Australian Sign Language). Your replies are turned into signing, so reply with ONE short, simple, "
    "everyday English sentence of at most 12 words. Use common words. No emojis, lists, markdown, "
    "numbers written as digits, or quotation marks. Do not mention that you are an AI unless asked. "
    "The user's message may come from sign recognition and contain mistakes; if it is unclear, "
    "ask them kindly to sign it again."
)


def clean_reply(text: str, max_words: int) -> str:
    """Plain words, whole sentences while they fit in max_words (at least the first, cut to fit),
    ending in punctuation."""
    t = re.sub(r"[*_`#>\"“”]|\[[^\]]*\]\([^)]*\)", "", text or "").strip()
    t = re.sub(r"\s+", " ", t)
    kept = []
    for s in re.findall(r"[^.!?]+[.!?]*", t):
        if kept and len(" ".join(kept + [s.strip()]).split()) > max_words:
            break
        kept.append(s.strip())
    t = " ".join(k for k in kept if k)
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(",;:") + "."
    if t and t[-1] not in ".!?":
        t += "."
    return t or "Sorry, can you sign that again?"


class Dialogue:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg["dialogue"]
        self.system = self.cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    def _complete(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def reply(self, history: list[dict], user_text: str) -> str:
        """history: [{'role': 'user'|'assistant', 'content': str}, ...] of earlier turns."""
        messages = list(history) + [{"role": "user", "content": user_text}]
        return clean_reply(self._complete(messages), self.cfg["max_words"])


class EchoDialogue(Dialogue):
    name = "echo"

    def reply(self, history, user_text):
        return clean_reply(user_text, 10 ** 6)


class HFDialogue(Dialogue):
    name = "hf"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = self.cfg.get("device") or cfg["device"]
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.tok = AutoTokenizer.from_pretrained(self.cfg["model"])
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg["model"], torch_dtype=dtype).to(dev).eval()

    def _complete(self, messages):
        import torch
        chat = [{"role": "system", "content": self.system}] + messages
        ids = self.tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.device)
        temp = float(self.cfg["temperature"])
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.cfg["max_new_tokens"],
                                      do_sample=temp > 0, temperature=temp if temp > 0 else None,
                                      top_p=0.9 if temp > 0 else None, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class OpenAICompatDialogue(Dialogue):
    """Plain HTTP to /chat/completions, so no client library is needed for local servers."""
    name = "openai"

    def _complete(self, messages):
        import requests
        key = os.environ.get(self.cfg["api_key_env"], "")
        r = requests.post(self.cfg["base_url"].rstrip("/") + "/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {key}"} if key else {},
                          json={"model": self.cfg["model"], "max_tokens": self.cfg["max_new_tokens"],
                                "temperature": self.cfg["temperature"],
                                "messages": [{"role": "system", "content": self.system}] + messages})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicDialogue(Dialogue):
    """Claude via the official SDK. Default model claude-opus-5 at low effort (a one-sentence chat
    reply needs little thinking), with server-side refusal fallbacks enabled."""
    name = "anthropic"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import anthropic
        self.client = anthropic.Anthropic()          # ANTHROPIC_API_KEY, or an `ant auth login` profile
        if not self.cfg["model"].startswith("claude"):
            self.cfg = dict(self.cfg, model="claude-opus-5")

    def _complete(self, messages):
        resp = self.client.beta.messages.create(
            model=self.cfg["model"], max_tokens=4096, system=self.system, messages=messages,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if resp.stop_reason == "refusal":
            return "Sorry, I cannot help with that."
        return "".join(b.text for b in resp.content if b.type == "text")


BACKENDS = {c.name: c for c in (EchoDialogue, HFDialogue, OpenAICompatDialogue, AnthropicDialogue)}


def build_dialogue(cfg: dict) -> Dialogue:
    name = cfg["dialogue"]["backend"]
    if name not in BACKENDS:
        raise ValueError(f"dialogue.backend={name!r}; choose one of {sorted(BACKENDS)}")
    return BACKENDS[name](cfg)
