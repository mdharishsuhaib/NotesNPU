"""Download the models NotesNPU needs.

Usage (from the project root):
    python scripts/download_models.py                 # auto: picks what fits this PC
    python scripts/download_models.py --asr aihub     # Qualcomm AI Hub Whisper (NPU)
    python scripts/download_models.py --asr cpu       # portable ONNX Whisper (CPU)
    python scripts/download_models.py --llm phi35     # Phi-3.5-mini INT4 (onnxruntime-genai)
    python scripts/download_models.py --llm none      # skip the LLM (extractive mode)

Everything lands in ./models and is never committed to git.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))

from core.device import detect_chipset, is_windows_arm64  # noqa: E402

AIHUB_VERSION = "v0.63.0"
AIHUB_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
    "whisper_{size}/releases/{ver}/whisper_{size}-precompiled_qnn_onnx-float-qualcomm_{chip}.zip"
)
AIHUB_CHIPSETS = {
    "x_elite": "snapdragon_x_elite",   # also used for Snapdragon X Plus (same Hexagon v73 NPU)
    "x2_elite": "snapdragon_x2_elite",  # Snapdragon X2 Elite / X2 Plus
}

WHISPER_HF_REPO = "onnx-community/whisper-{size}"
TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "normalizer.json",
]
CPU_ONNX_FILES = ["onnx/encoder_model.onnx", "onnx/decoder_model_merged_quantized.onnx"]

PHI35_REPO = "microsoft/Phi-3.5-mini-instruct-onnx"
PHI35_SUBDIR = "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"


def _progress(block_num: int, block_size: int, total: int) -> None:
    done = block_num * block_size
    if total > 0:
        pct = min(100.0, done * 100.0 / total)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:7.1f} / {total / 1e6:.1f} MB)")
        sys.stdout.flush()


def download_tokenizer(size: str) -> Path:
    from huggingface_hub import hf_hub_download

    out = MODELS / f"whisper-{size}-tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[tokenizer] openai/whisper-{size} tokenizer + feature extractor -> {out}")
    for f in TOKENIZER_FILES:
        if (out / f).exists():
            continue
        p = hf_hub_download(WHISPER_HF_REPO.format(size=size), f)
        shutil.copy(p, out / f)
    return out


def download_whisper_cpu(size: str) -> Path:
    from huggingface_hub import hf_hub_download

    out = MODELS / f"whisper-{size}-cpu"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[asr-cpu] ONNX Whisper-{size} (portable CPU fallback) -> {out}")
    for f in CPU_ONNX_FILES:
        dst = out / Path(f).name
        if dst.exists():
            print(f"  exists: {dst.name}")
            continue
        print(f"  downloading {f} ...")
        p = hf_hub_download(WHISPER_HF_REPO.format(size=size), f)
        shutil.copy(p, dst)
    return out


def download_whisper_aihub(size: str, chipset: str) -> Path:
    chip = AIHUB_CHIPSETS[chipset]
    out = MODELS / f"whisper-{size}-aihub-{chipset}"
    if (out / "encoder.onnx").exists() and (out / "decoder.onnx").exists():
        print(f"[asr-npu] already present: {out}")
        return out
    url = AIHUB_URL.format(size=size, ver=AIHUB_VERSION, chip=chip)
    MODELS.mkdir(parents=True, exist_ok=True)
    zpath = MODELS / f"_whisper_{size}_{chipset}.zip"
    print(f"[asr-npu] Qualcomm AI Hub Whisper-{size} (precompiled QNN ONNX, {chip})")
    print(f"  {url}")
    urllib.request.urlretrieve(url, zpath, _progress)
    print()
    tmp = MODELS / f"_extract_{chipset}"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(inner), str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    zpath.unlink(missing_ok=True)
    print(f"  extracted -> {out}")
    return out


def download_phi35() -> Path:
    from huggingface_hub import snapshot_download

    out = MODELS / "phi-3.5-mini-instruct-int4"
    if (out / "genai_config.json").exists():
        print(f"[llm] already present: {out}")
        return out
    print(f"[llm] {PHI35_REPO}/{PHI35_SUBDIR} (~2.6 GB, INT4) ...")
    tmp = MODELS / "_phi35"
    snapshot_download(PHI35_REPO, allow_patterns=[f"{PHI35_SUBDIR}/*"], local_dir=tmp)
    src = tmp / PHI35_SUBDIR
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(src), str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", choices=["auto", "aihub", "cpu", "both", "none"], default="auto")
    ap.add_argument("--size", choices=["tiny", "base", "small"], default="base", help="Whisper size")
    ap.add_argument("--chipset", choices=["auto", *AIHUB_CHIPSETS], default="auto")
    ap.add_argument("--llm", choices=["phi35", "none"], default="phi35")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    print(f"Host: {platform.platform()} | machine={platform.machine()} | ARM64={is_windows_arm64()}")

    asr = args.asr
    if asr == "auto":
        # On Snapdragon get the NPU model *and* the CPU model (used for NPU-vs-CPU benchmarks).
        asr = "both" if is_windows_arm64() else "cpu"

    download_tokenizer(args.size)
    if asr in ("cpu", "both"):
        download_whisper_cpu(args.size)
    if asr in ("aihub", "both"):
        chipset = args.chipset if args.chipset != "auto" else (detect_chipset() or "x_elite")
        download_whisper_aihub(args.size, chipset)
    if args.llm == "phi35":
        download_phi35()
    print("\nAll done. Start the app with:  .\\run.ps1   (or double-click run.bat)")


if __name__ == "__main__":
    main()
