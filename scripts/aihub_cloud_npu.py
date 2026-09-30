r"""Run NotesNPU's Qualcomm AI Hub Whisper on a *real* Snapdragon X Elite NPU in Qualcomm AI Hub's cloud.

Use this when you are not on a Snapdragon PC. It:
  1. uploads the AI Hub precompiled QNN context binaries (Whisper-Base encoder + decoder, X Elite)
  2. profiles both on a hosted Snapdragon X Elite device (real Hexagon NPU timings)
  3. runs the NPU encoder on the sample lecture and checks its cross-attention K/V caches
     against the CPU reference model (cosine similarity)
  4. decodes text from the NPU-computed encoder output, proving the NPU path end to end

Setup (once):  .\.venv\Scripts\pip install qai-hub
               .\.venv\Scripts\qai-hub configure --api_token <YOUR_TOKEN>
Run:           .\.venv\Scripts\python scripts\aihub_cloud_npu.py
Results:       outputs/aihub_cloud_npu.json (+ printed Markdown table)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.asr import MODELS, CpuWhisper, WhisperText, _repeating  # noqa: E402
from core.audio import load_audio, smart_chunks  # noqa: E402

DEVICE = "Snapdragon X Elite CRD"
URL = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/whisper_base/releases/"
       "v0.63.0/whisper_base-qnn_context_binary-float-qualcomm_snapdragon_x_elite.zip")
BIN_DIR = MODELS / "whisper-base-aihub-ctxbin-x_elite"
OUT = ROOT / "outputs" / "aihub_cloud_npu.json"
SAMPLE = ROOT / "samples" / "sample_lecture.wav"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_binaries() -> tuple[Path, Path]:
    if not BIN_DIR.exists() or not list(BIN_DIR.glob("*.bin")):
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        z = BIN_DIR / "_dl.zip"
        log(f"Downloading AI Hub QNN context binaries (~200 MB)\n  {URL}")
        urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith(".bin") or m.endswith(".json"):
                    (BIN_DIR / Path(m).name).write_bytes(zf.read(m))
        z.unlink()
    bins = sorted(BIN_DIR.glob("*.bin"))
    enc = next(b for b in bins if "encoder" in b.name.lower())
    dec = next(b for b in bins if "decoder" in b.name.lower())
    return enc, dec


def summarize_profile(prof: dict) -> dict:
    s = prof.get("execution_summary", {})
    layers = prof.get("execution_detail", [])
    units: dict[str, int] = {}
    for l in layers:
        u = l.get("compute_unit", "?")
        units[u] = units.get(u, 0) + 1
    t = s.get("estimated_inference_time")
    return {
        "inference_ms": round(t / 1000, 3) if t else None,
        "peak_memory_mb": round(s.get("estimated_inference_peak_memory", 0) / 1e6, 1),
        "first_load_ms": round(s.get("first_load_time", 0) / 1000, 1) if s.get("first_load_time") else None,
        "compute_units": units or {"NPU": "context binary (all on HTP)"},
    }


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decode_with_npu_cross_kv(cpu: CpuWhisper, text: WhisperText, npu_out: dict, n_layers: int) -> str:
    """Greedy-decode using cross-attention K/V computed on the Snapdragon NPU.

    The CPU merged decoder accepts precomputed encoder K/V through its cache branch, so the
    audio understanding (the expensive encoder) comes entirely from the NPU result.
    """
    past = {}
    for i in range(n_layers):
        k = npu_out[f"k_cache_cross_{i}"].astype(np.float32)  # [8,1,64,1500]
        v = npu_out[f"v_cache_cross_{i}"].astype(np.float32)  # [8,1,1500,64]
        past[f"past_key_values.{i}.encoder.key"] = np.transpose(k, (1, 0, 3, 2))  # -> [1,8,1500,64]
        past[f"past_key_values.{i}.encoder.value"] = np.transpose(v, (1, 0, 2, 3))
    hidden_dummy = np.zeros((1, 1500, 512), dtype=np.float32)  # ignored on the cache branch
    ids = text.prompt("en", "transcribe")
    out: list[int] = []
    # decoder self-attention cache grows from 0
    for i in range(n_layers):
        for kv in ("key", "value"):
            past[f"past_key_values.{i}.decoder.{kv}"] = np.zeros((1, cpu.heads, 0, cpu.head_dim), np.float32)
    feed_ids = ids
    for _ in range(200):
        feed = {"input_ids": np.array([feed_ids], np.int64), "encoder_hidden_states": hidden_dummy,
                "use_cache_branch": np.array([True]), **past}
        res = dict(zip(cpu.out_names, cpu.dec.run(cpu.out_names, feed)))
        for i in range(n_layers):
            for kv in ("key", "value"):
                past[f"past_key_values.{i}.decoder.{kv}"] = res[f"present.{i}.decoder.{kv}"]
        nxt = int(np.argmax(cpu._mask(res["logits"][0, -1])))
        if nxt == text.eot:
            break
        out.append(nxt)
        feed_ids = [nxt]
        if _repeating(out):
            break
    return text.decode(out)


def submit(hub, device, text: WhisperText, chunks: list):
    enc_bin, dec_bin = get_binaries()
    log(f"Target: {DEVICE} (hosted by Qualcomm AI Hub)")
    log("Uploading encoder / decoder QNN context binaries ...")
    enc_m = hub.upload_model(str(enc_bin))
    dec_m = hub.upload_model(str(dec_bin))
    log("Submitting NPU profile jobs ...")
    pj_enc = hub.submit_profile_job(model=enc_m, device=device, name="NotesNPU whisper encoder")
    pj_dec = hub.submit_profile_job(model=dec_m, device=device, name="NotesNPU whisper decoder")
    feats = [text.features(c).astype(np.float16) for c in chunks]
    log(f"Submitting NPU inference job: {len(feats)} x 30 s chunks of the sample lecture ...")
    ij = hub.submit_inference_job(model=enc_m, device=device, inputs={"input_features": feats},
                                  name="NotesNPU whisper encoder - sample lecture")
    return pj_enc, pj_dec, ij


def main() -> None:
    import qai_hub as hub

    OUT.parent.mkdir(exist_ok=True)
    device = hub.Device(DEVICE)
    tok_dir = MODELS / "whisper-base-tokenizer"
    text = WhisperText(tok_dir)
    audio = load_audio(SAMPLE)
    chunks = [c for _, c in smart_chunks(audio)]

    if len(sys.argv) == 4:  # resume: aihub_cloud_npu.py <enc_profile_job> <dec_profile_job> <inference_job>
        pj_enc, pj_dec, ij = (hub.get_job(j) for j in sys.argv[1:4])
        log(f"Re-attaching to existing jobs {sys.argv[1:4]}")
    else:
        pj_enc, pj_dec, ij = submit(hub, device, text, chunks)

    for j in (pj_enc, pj_dec, ij):
        log(f"  job {j.job_id}: {j.url}")
    log("Waiting for Qualcomm AI Hub (queue + run usually 5-20 min) ...")

    res: dict = {"device": DEVICE, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "jobs": {"profile_encoder": pj_enc.url, "profile_decoder": pj_dec.url, "inference_encoder": ij.url}}
    enc_p = summarize_profile(pj_enc.download_profile())
    log(f"Encoder on NPU: {enc_p}")
    dec_p = summarize_profile(pj_dec.download_profile())
    log(f"Decoder on NPU: {dec_p}")
    res["profile"] = {"encoder": enc_p, "decoder": dec_p}

    npu = ij.download_output_data()
    n_layers = sum(1 for k in npu if k.startswith("k_cache_cross_"))
    cpu = CpuWhisper(text, MODELS / "whisper-base-cpu")

    checks, transcripts = [], []
    for ci, chunk in enumerate(chunks):
        npu_out = {k: np.asarray(v[ci]) for k, v in npu.items()}
        # CPU reference cross K/V: fp32 encoder + first decoder step's present.*.encoder.*
        hidden = cpu._encode(chunk, None)
        empty = np.zeros((1, cpu.heads, 0, cpu.head_dim), np.float32)
        feed = {"input_ids": np.array([text.prompt("en", "transcribe")], np.int64), "encoder_hidden_states": hidden,
                "use_cache_branch": np.array([False]), **{n: empty for n in cpu.past_names}}
        ref = dict(zip(cpu.out_names, cpu.dec.run(cpu.out_names, feed)))
        sims = []
        for i in range(n_layers):
            k_ref = np.transpose(ref[f"present.{i}.encoder.key"][0], (0, 2, 1))[:, None]  # [8,1,64,1500]
            v_ref = ref[f"present.{i}.encoder.value"][0][:, None]  # [8,1,1500,64]
            sims += [cos(npu_out[f"k_cache_cross_{i}"], k_ref), cos(npu_out[f"v_cache_cross_{i}"], v_ref)]
        checks.append(round(float(np.mean(sims)), 5))
        transcripts.append(decode_with_npu_cross_kv(cpu, text, npu_out, n_layers))
        log(f"  chunk {ci}: NPU vs CPU cross-KV cosine = {checks[-1]}")

    res["encoder_accuracy_cosine_vs_cpu"] = checks
    res["transcript_from_npu_encoder"] = " ".join(transcripts)
    audio_s = len(audio) / 16000
    enc_ms, dec_ms = enc_p["inference_ms"] or 0, dec_p["inference_ms"] or 0
    tokens = sum(len(text.tok.encode(t)) for t in transcripts)
    est = len(chunks) * enc_ms + tokens * dec_ms
    res["estimated_npu_processing_s"] = round(est / 1000, 3)
    res["estimated_speed_x_realtime"] = round(audio_s / (est / 1000), 1) if est else None
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n## Whisper-Base on a real Snapdragon X Elite NPU (Qualcomm AI Hub)\n")
    print("| Stage | NPU latency | Peak memory |\n|---|---|---|")
    print(f"| Encoder (30 s of audio) | {enc_ms} ms | {enc_p['peak_memory_mb']} MB |")
    print(f"| Decoder (per token) | {dec_ms} ms | {dec_p['peak_memory_mb']} MB |")
    print(f"\nSample lecture: {audio_s:.1f} s, {len(chunks)} chunks, {tokens} tokens -> "
          f"~{res['estimated_npu_processing_s']} s on the NPU (~{res['estimated_speed_x_realtime']}x real time)")
    print(f"Encoder accuracy (cosine vs CPU fp32): {checks}")
    print(f"\nTranscript decoded from NPU encoder output:\n{res['transcript_from_npu_encoder']}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
