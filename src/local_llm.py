"""Lazily loaded optional FLAN-T5 model for local rewriting/synthesis."""
from __future__ import annotations


class LocalTextModel:
    def __init__(self, model_id: str = "google/flan-t5-small"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def generate(self, instruction: str, max_new_tokens: int = 100) -> str:
        # Lazy loading keeps deterministic QA independent of downloads.
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._model.eval()
        import torch
        encoded = self._tokenizer(instruction, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            generated = self._model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
        return self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
