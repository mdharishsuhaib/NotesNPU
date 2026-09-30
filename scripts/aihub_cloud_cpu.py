r"""Profile NotesNPU's CPU Whisper-Base baseline on the Oryon CPU of a hosted Snapdragon X Elite.

Gives a same-device CPU vs Hexagon NPU comparison (see scripts/aihub_cloud_npu.py for the NPU side).

Run:      .\.venv\Scripts\python scripts\aihub_cloud_cpu.py [encoder_job_id decoder_job_id]
Results:  docs/benchmarks/aihub_cloud_cpu.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import qai_hub as hub

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "Snapdragon X Elite CRD"
# Fixed-shape copies of models/whisper-base-cpu (AI Hub needs static shapes), made with
#   python -m onnxruntime.tools.make_dynamic_shape_fixed  (encoder 1x80x3000; decoder batch 1, past 100, enc 1500)
FIXED = ROOT / "outputs" / "cpu_fixed"
OUT = ROOT / "docs" / "benchmarks" / "aihub_cloud_cpu.json"
OPTS = "--compute_unit cpu"


def summary(job: hub.ProfileJob) -> dict:
    prof = job.download_profile()
    s = prof["execution_summary"]
    return {
        "job_id": job.job_id,
        "job_url": job.url,
        "inference_ms": s["estimated_inference_time"] / 1000.0,
        "peak_memory_mb": s["estimated_inference_peak_memory"] / 2**20,
    }


def main() -> None:
    device = hub.Device(DEVICE)
    if len(sys.argv) == 3:
        enc_job, dec_job = hub.get_job(sys.argv[1]), hub.get_job(sys.argv[2])
    else:
        enc_job = hub.submit_profile_job(
            model=str(FIXED / "encoder.onnx"), device=device, name="notesnpu-whisper-encoder-cpu", options=OPTS)
        dec_job = hub.submit_profile_job(
            model=str(FIXED / "decoder.onnx"), device=device, name="notesnpu-whisper-decoder-cpu", options=OPTS)
    print("encoder job:", enc_job.url, "\ndecoder job:", dec_job.url, flush=True)
    res = {"device": DEVICE, "compute_unit": "cpu", "encoder": summary(enc_job), "decoder": summary(dec_job)}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
