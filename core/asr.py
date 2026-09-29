"""Whisper speech-to-text on the Snapdragon NPU (Qualcomm AI Hub) with a CPU fallback.

Backends
--------
* ``AIHubWhisper``  - Qualcomm AI Hub Whisper, *precompiled QNN ONNX* (EPContext) encoder +
  KV-cache decoder, run on the Hexagon NPU through the ONNX Runtime QNN Execution Provider.
  The I/O contract mirrors ``qai_hub_models.models.templates.hf_whisper.app``: the encoder
  returns cross-attention K/V caches directly, the decoder is a fixed-shape (200 token)
  single-step graph with self-attention caches, an additive attention mask and position ids.
* ``CpuWhisper``    - portable ONNX export of the same OpenAI Whisper checkpoint
  (``onnx-community/whisper-*``) on the ONNX Runtime CPU EP. Used on non-Snapdragon PCs
  and as the baseline for NPU-vs-CPU benchmarks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from core.audio import SAMPLE_RATE, is_silent, smart_chunks
from core.device import make_session, qnn_status, run_options

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # tokenizer-only use; no PyTorch needed
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}
MASK_NEG = -100.0

LANGUAGES = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Bengali": "bn",
             "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Urdu": "ur",
             "Punjabi": "pa", "Auto-detect": "auto"}


@dataclass
class AsrStats:
    backend: str = ""
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    encoder_ms: list = field(default_factory=list)
    decoder_ms_per_token: list = field(default_factory=list)
    tokens: int = 0

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio duration (lower is better)."""
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else 0.0

    def as_dict(self) -> dict:
        enc = float(np.mean(self.encoder_ms)) if self.encoder_ms else 0.0
        dec = float(np.mean(self.decoder_ms_per_token)) if self.decoder_ms_per_token else 0.0
        return {
            "backend": self.backend,
            "audio_s": round(self.audio_seconds, 2),
            "processing_s": round(self.wall_seconds, 2),
            "real_time_factor": round(self.rtf, 3),
            "speed_x_realtime": round(1 / self.rtf, 1) if self.rtf else 0.0,
            "encoder_ms_avg": round(enc, 1),
            "decoder_ms_per_token": round(dec, 2),
            "tokens": self.tokens,
        }


class WhisperText:
    """Feature extraction (log-mel) + tokenizer, loaded from local files only."""

    def __init__(self, tok_dir: Path):
        from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

        self.fe = WhisperFeatureExtractor.from_pretrained(str(tok_dir))
        self.tok = WhisperTokenizerFast.from_pretrained(str(tok_dir))
        cfg = json.loads((tok_dir / "config.json").read_text(encoding="utf-8"))
        self.sot = cfg.get("decoder_start_token_id", 50258)
        self.eot = cfg.get("eos_token_id", 50257)
        tid = self.tok.convert_tokens_to_ids
        self.no_ts = tid("<|notimestamps|>")
        self.transcribe_id = tid("<|transcribe|>")
        self.translate_id = tid("<|translate|>")
        self.ts_begin = self.no_ts + 1
        self.lang_ids = {c: tid(f"<|{c}|>") for c in LANGUAGES.values() if c != "auto"}
        self.english_only = self.lang_ids.get("en") in (None, self.tok.unk_token_id)

    def features(self, audio: np.ndarray) -> np.ndarray:
        return self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"].astype(np.float32)

    def prompt(self, lang: str, task: str) -> list[int]:
        if self.english_only:
            return [self.sot, self.no_ts]
        task_id = self.translate_id if task == "translate" else self.transcribe_id
        return [self.sot, self.lang_ids.get(lang, self.lang_ids["en"]), task_id, self.no_ts]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()


def _repeating(ids: list[int]) -> bool:
    """Detect greedy-decoding loops (same n-gram repeated 3+ times at the end)."""
    for n in (2, 3, 4, 5, 6):
        if len(ids) >= 3 * n and ids[-n:] == ids[-2 * n:-n] == ids[-3 * n:-2 * n]:
            return True
    return len(ids) >= 6 and len(set(ids[-6:])) == 1


class _Base:
    label = "base"

    def __init__(self, text: WhisperText):
        self.text = text

    def _mask(self, logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32).reshape(-1)
        logits[self.text.ts_begin:] = -np.inf  # no timestamp tokens in no-timestamp mode
        return logits

    def detect_language(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    def transcribe_chunk(self, audio: np.ndarray, prompt: list[int], stats: AsrStats) -> list[int]:
        raise NotImplementedError


class AIHubWhisper(_Base):
    """Qualcomm AI Hub Whisper (precompiled QNN ONNX) on the Hexagon NPU."""

    def __init__(self, text: WhisperText, model_dir: Path):
        super().__init__(text)
        if not qnn_status().available:
            raise RuntimeError("Qualcomm NPU (QNN EP) not available: " + qnn_status().detail)
        self.enc, self.label = make_session(str(model_dir / "encoder.onnx"), prefer_npu=True)
        self.dec, _ = make_session(str(model_dir / "decoder.onnx"), prefer_npu=True)
        self.ro = run_options(True)
        self.enc_in = self.enc.get_inputs()[0]
        self.enc_out = [o.name for o in self.enc.get_outputs()]
        self.dec_in = {i.name: i for i in self.dec.get_inputs()}
        self.dec_out = [o.name for o in self.dec.get_outputs()]
        self.n_layers = sum(1 for n in self.dec_in if n.startswith("k_cache_self_"))
        self.max_len = self.dec_in["attention_mask"].shape[-1]  # 200
        self.label = f"AI Hub Whisper | {self.label}"

    def _dt(self, name: str):
        return ORT_TO_NP[self.dec_in[name].type]

    def _encode(self, audio: np.ndarray, stats: AsrStats | None) -> dict:
        feats = self.text.features(audio).astype(ORT_TO_NP[self.enc_in.type])
        t = time.perf_counter()
        outs = self.enc.run(self.enc_out, {self.enc_in.name: feats}, self.ro)
        if stats is not None:
            stats.encoder_ms.append((time.perf_counter() - t) * 1000)
        return dict(zip(self.enc_out, outs))

    def _decode(self, cross: dict, prompt: list[int], stats: AsrStats | None, max_new: int | None = None,
                first_logits_cb: Callable | None = None) -> list[int]:
        feed = {k: v.astype(self._dt(k)) for k, v in cross.items() if k in self.dec_in}
        for i in range(self.n_layers):
            for p in ("k", "v"):
                name = f"{p}_cache_self_{i}_in"
                feed[name] = np.zeros(self.dec_in[name].shape, dtype=self._dt(name))
        mask = np.full((1, 1, 1, self.max_len), MASK_NEG, dtype=self._dt("attention_mask"))
        ids = list(prompt)
        out_ids: list[int] = []
        t0 = time.perf_counter()
        steps = 0
        for n in range(self.max_len - 1):
            mask[..., self.max_len - n - 1] = 0.0
            feed["input_ids"] = np.array([[ids[n]]], dtype=self._dt("input_ids"))
            feed["position_ids"] = np.array([n], dtype=self._dt("position_ids"))
            feed["attention_mask"] = mask
            res = dict(zip(self.dec_out, self.dec.run(self.dec_out, feed, self.ro)))
            steps += 1
            for i in range(self.n_layers):
                feed[f"k_cache_self_{i}_in"] = res[f"k_cache_self_{i}_out"]
                feed[f"v_cache_self_{i}_in"] = res[f"v_cache_self_{i}_out"]
            if n < len(ids) - 1:
                continue  # still feeding the forced prompt
            if first_logits_cb is not None:
                return first_logits_cb(res["logits"].astype(np.float32).reshape(-1))
            nxt = int(np.argmax(self._mask(res["logits"])))
            if nxt == self.text.eot:
                break
            ids.append(nxt)
            out_ids.append(nxt)
            if _repeating(out_ids) or (max_new and len(out_ids) >= max_new):
                break
        if stats is not None and steps:
            stats.decoder_ms_per_token.append((time.perf_counter() - t0) * 1000 / steps)
            stats.tokens += len(out_ids)
        return out_ids

    def detect_language(self, audio: np.ndarray) -> str:
        cross = self._encode(audio, None)
        codes = list(self.text.lang_ids)

        def pick(logits):
            return codes[int(np.argmax([logits[self.text.lang_ids[c]] for c in codes]))]

        return self._decode(cross, [self.text.sot], None, first_logits_cb=pick)

    def transcribe_chunk(self, audio: np.ndarray, prompt: list[int], stats: AsrStats) -> list[int]:
        return self._decode(self._encode(audio, stats), prompt, stats)


class CpuWhisper(_Base):
    """Portable ONNX Whisper (encoder fp32 + merged int8 KV-cache decoder) on the CPU EP."""

    def __init__(self, text: WhisperText, model_dir: Path, max_len: int = 200):
        super().__init__(text)
        self.enc, _ = make_session(str(model_dir / "encoder_model.onnx"), prefer_npu=False)
        self.dec, _ = make_session(str(model_dir / "decoder_model_merged_quantized.onnx"), prefer_npu=False)
        self.label = "ONNX Whisper | CPU"
        self.max_len = max_len
        self.past_names = [i.name for i in self.dec.get_inputs() if i.name.startswith("past_key_values")]
        self.out_names = [o.name for o in self.dec.get_outputs()]
        self.has_branch = any(i.name == "use_cache_branch" for i in self.dec.get_inputs())
        sample = next(i for i in self.dec.get_inputs() if i.name.startswith("past_key_values"))
        self.heads, self.head_dim = sample.shape[1], sample.shape[3]

    def _encode(self, audio: np.ndarray, stats: AsrStats | None) -> np.ndarray:
        feats = self.text.features(audio)
        t = time.perf_counter()
        hidden = self.enc.run(None, {"input_features": feats})[0]
        if stats is not None:
            stats.encoder_ms.append((time.perf_counter() - t) * 1000)
        return hidden

    def _step(self, hidden, input_ids, past):
        feed = {"input_ids": np.array([input_ids], dtype=np.int64), "encoder_hidden_states": hidden}
        feed.update(past)
        if self.has_branch:
            feed["use_cache_branch"] = np.array([bool(past.get("_cached", False))])
        feed.pop("_cached", None)
        res = dict(zip(self.out_names, self.dec.run(self.out_names, {k: v for k, v in feed.items()})))
        return res

    def _decode(self, hidden, prompt, stats, first_logits_cb=None):
        empty = np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32)
        past = {n: empty for n in self.past_names}
        ids = list(prompt)
        out_ids: list[int] = []
        t0 = time.perf_counter()
        steps = 0
        feed_ids = ids  # first step: whole prompt at once
        enc_cache = None
        while len(ids) < self.max_len:
            res = self._step(hidden, feed_ids, dict(past, _cached=enc_cache is not None))
            steps += 1
            logits = res["logits"][0, -1]
            if first_logits_cb is not None:
                return first_logits_cb(logits.astype(np.float32))
            for n in self.past_names:
                pres = n.replace("past_key_values", "present")
                if ".encoder." in n:
                    if enc_cache is None:
                        past[n] = res[pres]
                else:
                    past[n] = res[pres]
            enc_cache = True
            nxt = int(np.argmax(self._mask(logits)))
            if nxt == self.text.eot:
                break
            ids.append(nxt)
            out_ids.append(nxt)
            feed_ids = [nxt]
            if _repeating(out_ids):
                break
        if stats is not None and steps:
            stats.decoder_ms_per_token.append((time.perf_counter() - t0) * 1000 / steps)
            stats.tokens += len(out_ids)
        return out_ids

    def detect_language(self, audio: np.ndarray) -> str:
        codes = list(self.text.lang_ids)

        def pick(logits):
            return codes[int(np.argmax([logits[self.text.lang_ids[c]] for c in codes]))]

        return self._decode(self._encode(audio, None), [self.text.sot], None, first_logits_cb=pick)

    def transcribe_chunk(self, audio: np.ndarray, prompt: list[int], stats: AsrStats) -> list[int]:
        return self._decode(self._encode(audio, stats), prompt, stats)


def _find(pattern: str) -> Path | None:
    hits = sorted(MODELS.glob(pattern)) if MODELS.exists() else []
    return hits[0] if hits else None


def available_backends(size: str = "base") -> dict[str, Path]:
    out = {}
    npu_dir = _find(f"whisper-{size}-aihub-*")
    if npu_dir and (npu_dir / "encoder.onnx").exists() and qnn_status().available:
        out["npu"] = npu_dir
    cpu_dir = MODELS / f"whisper-{size}-cpu"
    if (cpu_dir / "encoder_model.onnx").exists():
        out["cpu"] = cpu_dir
    return out


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber:
    """High-level API: picks NPU when possible, chunks long audio, streams segments."""

    def __init__(self, backend: str = "auto", size: str = "base"):
        tok_dir = MODELS / f"whisper-{size}-tokenizer"
        if not tok_dir.exists():
            raise FileNotFoundError("Whisper tokenizer missing - run: python scripts/download_models.py")
        self.text = WhisperText(tok_dir)
        avail = available_backends(size)
        if not avail:
            raise FileNotFoundError("No Whisper model found in ./models - run: python scripts/download_models.py")
        order = {"auto": ["npu", "cpu"], "npu": ["npu", "cpu"], "cpu": ["cpu", "npu"]}[backend]
        last_err = None
        self.engine: _Base | None = None
        for key in order:
            if key not in avail:
                continue
            try:
                self.engine = AIHubWhisper(self.text, avail[key]) if key == "npu" else CpuWhisper(self.text, avail[key])
                self.kind = key
                break
            except Exception as e:  # e.g. QNN driver mismatch -> fall back
                last_err = e
        if self.engine is None:
            raise RuntimeError(f"Could not start any Whisper backend: {last_err}")
        self.label = self.engine.label

    def transcribe(self, audio: np.ndarray, language: str = "en", task: str = "transcribe",
                   stats: AsrStats | None = None) -> Iterator[Segment]:
        stats = stats if stats is not None else AsrStats()
        stats.backend = self.label
        stats.audio_seconds += len(audio) / SAMPLE_RATE
        t0 = time.perf_counter()
        lang = language
        for start, chunk in smart_chunks(audio):
            if len(chunk) < SAMPLE_RATE * 0.3 or is_silent(chunk):
                continue
            if lang == "auto":
                lang = self.engine.detect_language(chunk)
            ids = self.engine.transcribe_chunk(chunk, self.text.prompt(lang, task), stats)
            text = self.text.decode(ids)
            stats.wall_seconds = time.perf_counter() - t0
            if text:
                yield Segment(start, start + len(chunk) / SAMPLE_RATE, text)
        stats.wall_seconds = time.perf_counter() - t0


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


if __name__ == "__main__":  # CLI: python -m core.asr lecture.wav [--backend npu|cpu] [--lang hi --task translate]
    import argparse

    from core.audio import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--backend", default="auto", choices=["auto", "npu", "cpu"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    a = ap.parse_args()
    tr = Transcriber(a.backend)
    st = AsrStats()
    print(f"Backend: {tr.label}")
    for seg in tr.transcribe(load_audio(a.audio), a.lang, a.task, st):
        print(f"[{fmt_ts(seg.start)}] {seg.text}")
    print(json.dumps(st.as_dict(), indent=2))
