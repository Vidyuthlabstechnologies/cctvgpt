"""Indian-language query translation for footage search.

The vision-language model understands English best: Hindi queries scored at
chance level without translation (benchmarks/search_eval.py). Queries in
Indian scripts are therefore translated to English first.

Backends, in order of preference:
  1. IndicTrans2 (AI4Bharat, MIT licence): 22 scheduled Indian languages.
     The Hugging Face weights are gated: accept the terms once with an account,
     then either set HF_TOKEN on the server or download the model folder and
     point CCTVGPT_INDICTRANS2_PATH at it (air-gapped installs).
  2. Helsinki-NLP opus-mt-hi-en (Apache-2.0): Hindi only, ungated. Weaker:
     it can drop words, so the translated text is always shown to the officer.
  3. No translation available: the query is passed through and the search
     result carries a warning.
"""

from __future__ import annotations

import os
import threading
import unicodedata

# Unicode script -> IndicTrans2 FLORES-style language tag (best guess from script).
SCRIPT_LANG = {
    "DEVANAGARI": "hin_Deva", "BENGALI": "ben_Beng", "GURMUKHI": "pan_Guru", "GUJARATI": "guj_Gujr",
    "ORIYA": "ory_Orya", "TAMIL": "tam_Taml", "TELUGU": "tel_Telu", "KANNADA": "kan_Knda",
    "MALAYALAM": "mal_Mlym", "ARABIC": "urd_Arab",
}

_lock = threading.Lock()
_backend = None


def detect_script(text: str) -> str | None:
    counts: dict = {}
    for ch in text:
        if ord(ch) < 0x0250:
            continue
        name = unicodedata.name(ch, "")
        for script in SCRIPT_LANG:
            if name.startswith(script):
                counts[script] = counts.get(script, 0) + 1
                break
    return max(counts, key=counts.get) if counts else None


class IndicTrans2Backend:
    name = "indictrans2"

    def __init__(self, source: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:  # older toolkit layout
            from IndicTransToolkit import IndicProcessor
        self.tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, trust_remote_code=True).eval()
        self.ip = IndicProcessor(inference=True)

    def translate(self, text: str, src_lang: str) -> str:
        import torch

        batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang="eng_Latn")
        enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**enc, num_beams=4, max_length=64)
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(dec, lang="eng_Latn")[0]


class OpusBackend:
    name = "opus-mt-hi-en"

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer

        self.tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en").eval()

    def translate(self, text: str, src_lang: str) -> str:
        import torch

        if src_lang != "hin_Deva":
            raise ValueError("opus-mt fallback handles Hindi only")
        with torch.no_grad():
            out = self.model.generate(**self.tok([text], return_tensors="pt"), num_beams=4, max_new_tokens=64)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


def _load_backend():
    global _backend
    with _lock:
        if _backend is not None:
            return _backend
        src = os.environ.get("CCTVGPT_INDICTRANS2_PATH") or "ai4bharat/indictrans2-indic-en-dist-200M"
        for factory in (lambda: IndicTrans2Backend(src), OpusBackend):
            try:
                _backend = factory()
                return _backend
            except Exception:  # noqa: BLE001 - gated, missing package or offline: try the next backend
                continue
        _backend = False
        return _backend


def translate_query(text: str, backend=None) -> dict:
    """Return {"text": english_text, "translated": bool, "source_lang": tag|None, "backend": name|None, "error": str|None}."""
    script = detect_script(text or "")
    if script is None:
        return {"text": text, "translated": False, "source_lang": None, "backend": None, "error": None}
    lang = SCRIPT_LANG[script]
    be = backend if backend is not None else _load_backend()
    if not be:
        return {"text": text, "translated": False, "source_lang": lang, "backend": None, "error": "no translation model available"}
    try:
        return {"text": be.translate(text, lang).strip(), "translated": True, "source_lang": lang, "backend": be.name, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"text": text, "translated": False, "source_lang": lang, "backend": be.name, "error": str(e)}
