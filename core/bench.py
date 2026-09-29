"""NPU vs CPU benchmark for NotesNPU.

    python -m core.bench                      # uses samples/sample_lecture.wav
    python -m core.bench my_audio.wav --llm   # also time the LLM

Results are printed and saved to outputs/benchmarks.json (used for the README / pitch deck).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from core.asr import AsrStats, Transcriber, available_backends
from core.audio import load_audio
from core.device import system_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "benchmarks.json"
SAMPLE = ROOT / "samples" / "sample_lecture.wav"


def bench_asr(audio, backends=None, warmup: bool = True) -> list[dict]:
    rows = []
    for kind in backends or list(available_backends()):
        try:
            tr = Transcriber(kind)
            if tr.kind != kind:
                continue
            if warmup:  # first NPU run loads the context binary / graph; exclude from timing
                list(tr.transcribe(audio[: 16000 * 5]))
            st = AsrStats()
            text = " ".join(s.text for s in tr.transcribe(audio, stats=st))
            row = st.as_dict()
            row["words"] = len(text.split())
            rows.append(row)
        except Exception as e:
            rows.append({"backend": kind, "error": str(e)})
    return rows


def bench_llm(transcript: str) -> dict:
    from core.llm import StudyEngine

    eng = StudyEngine()
    st = {}
    for _, st in eng.run("summary", transcript):
        pass
    return st


def speedup(rows: list[dict]) -> float | None:
    npu = next((r for r in rows if "NPU" in r.get("backend", "") and "processing_s" in r), None)
    cpu = next((r for r in rows if "CPU" == r.get("backend", "").split("|")[-1].strip() and "processing_s" in r), None)
    if npu and cpu and npu["processing_s"]:
        return round(cpu["processing_s"] / npu["processing_s"], 2)
    return None


def run(audio_path: Path = SAMPLE, with_llm: bool = False) -> dict:
    audio = load_audio(audio_path)
    rows = bench_asr(audio)
    res = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "system": system_summary(),
           "audio_file": audio_path.name, "asr": rows, "npu_speedup_vs_cpu": speedup(rows)}
    if with_llm:
        tr = Transcriber()
        text = " ".join(s.text for s in tr.transcribe(audio))
        res["llm"] = bench_llm(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def to_markdown(res: dict) -> str:
    lines = ["| Backend | Audio (s) | Processing (s) | x Real-time | Encoder (ms) | Decoder (ms/token) |",
             "|---|---|---|---|---|---|"]
    for r in res["asr"]:
        if "error" in r:
            lines.append(f"| {r['backend']} | error: {r['error'][:60]} | | | | |")
        else:
            lines.append(f"| {r['backend']} | {r['audio_s']} | {r['processing_s']} | {r['speed_x_realtime']}x | "
                         f"{r['encoder_ms_avg']} | {r['decoder_ms_per_token']} |")
    if res.get("npu_speedup_vs_cpu"):
        lines.append(f"\n**NPU speed-up vs CPU: {res['npu_speedup_vs_cpu']}x**")
    if res.get("llm"):
        l = res["llm"]
        lines.append(f"\n**LLM** ({l.get('engine')}): {l.get('tokens_per_s')} tokens/s, "
                     f"first token {l.get('time_to_first_token_s')} s")
    s = res["system"]
    lines.append(f"\n_Device: {s['cpu']} | NPU available: {s['qnn_npu']} ({s['qnn_detail']})_")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default=str(SAMPLE))
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    r = run(Path(a.audio), a.llm)
    print(to_markdown(r))
    print(f"\nSaved {OUT}")
