"""Offline self-test for scripts/aihub_cloud_npu.py (no AI Hub account needed).

Builds cross-attention K/V in the exact AI Hub encoder output layout from the CPU model, then
decodes with `decode_with_npu_cross_kv`. If the transcript is correct, the layout conversion
and cache-branch decoding used for the real NPU outputs are correct.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aihub_cloud_npu import decode_with_npu_cross_kv  # noqa: E402

from core.asr import MODELS, CpuWhisper, WhisperText  # noqa: E402
from core.audio import load_audio, smart_chunks  # noqa: E402

text = WhisperText(MODELS / "whisper-base-tokenizer")
cpu = CpuWhisper(text, MODELS / "whisper-base-cpu")
_, chunk = next(smart_chunks(load_audio(ROOT / "samples" / "sample_lecture.wav")))
hidden = cpu._encode(chunk, None)
empty = np.zeros((1, cpu.heads, 0, cpu.head_dim), np.float32)
feed = {"input_ids": np.array([text.prompt("en", "transcribe")], np.int64), "encoder_hidden_states": hidden,
        "use_cache_branch": np.array([False]), **{n: empty for n in cpu.past_names}}
ref = dict(zip(cpu.out_names, cpu.dec.run(cpu.out_names, feed)))
fake_npu = {}
for i in range(6):
    fake_npu[f"k_cache_cross_{i}"] = np.transpose(ref[f"present.{i}.encoder.key"][0], (0, 2, 1))[:, None].astype(np.float16)
    fake_npu[f"v_cache_cross_{i}"] = ref[f"present.{i}.encoder.value"][0][:, None].astype(np.float16)
print("shapes:", fake_npu["k_cache_cross_0"].shape, fake_npu["v_cache_cross_0"].shape)
print("TRANSCRIPT:", decode_with_npu_cross_kv(cpu, text, fake_npu, 6))
