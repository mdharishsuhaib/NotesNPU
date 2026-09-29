"""NotesNPU - private, offline AI lecture & meeting copilot for Snapdragon-powered HP PCs.

    python app.py [--asr auto|npu|cpu] [--no-llm] [--port 7860]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from core import bench  # noqa: E402
from core.asr import LANGUAGES, AsrStats, Transcriber, fmt_ts  # noqa: E402
from core.audio import SAMPLE_RATE, from_gradio, resample, to_mono_float  # noqa: E402
from core.device import system_summary  # noqa: E402
from core.llm import StudyEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SAMPLE = ROOT / "samples" / "sample_lecture.wav"
LIVE_WINDOW_S = 8.0

ASR: Transcriber | None = None
ASR_ERR = ""
STUDY: StudyEngine | None = None


def load_engines(asr_backend: str, use_llm: bool) -> None:
    global ASR, ASR_ERR, STUDY
    try:
        ASR = Transcriber(asr_backend)
    except Exception as e:
        ASR_ERR = str(e)
    STUDY = StudyEngine(use_llm=use_llm)


def status_md() -> str:
    s = system_summary()
    npu = "Hexagon NPU active" if (ASR and ASR.kind == "npu") else ("NPU available" if s["qnn_npu"] else "NPU not detected")
    asr = ASR.label if ASR else f"not loaded ({ASR_ERR})"
    return (f"**Device:** {s['cpu']} &nbsp;|&nbsp; **{npu}** &nbsp;|&nbsp; **Speech:** {asr} &nbsp;|&nbsp; "
            f"**Study AI:** {STUDY.label if STUDY else '-'} &nbsp;|&nbsp; 100% offline - audio never leaves this PC")


# ------------------------------------------------------------------ transcription
def transcribe(audio_value, language: str, translate: bool):
    if ASR is None:
        raise gr.Error(f"Speech model not loaded: {ASR_ERR}")
    audio = from_gradio(audio_value)
    if audio is None or len(audio) == 0:
        raise gr.Error("Record or upload some audio first (or click 'Load sample lecture').")
    stats = AsrStats()
    lines, plain = [], []
    task = "translate" if translate else "transcribe"
    for seg in ASR.transcribe(audio, LANGUAGES.get(language, "en"), task, stats):
        lines.append(f"[{fmt_ts(seg.start)}] {seg.text}")
        plain.append(seg.text)
        yield "\n".join(lines), " ".join(plain), stats.as_dict()
    if not lines:
        yield "(no speech detected)", "", stats.as_dict()
    else:
        yield "\n".join(lines), " ".join(plain), stats.as_dict()


def live_stream(chunk, buf, text):
    """Called ~every second while the mic is on; transcribes each ~8 s window on-device."""
    if chunk is None or ASR is None:
        return buf, text, text
    sr, data = chunk
    piece = resample(to_mono_float(data), sr)
    buf = piece if buf is None else np.concatenate([buf, piece])
    if len(buf) >= LIVE_WINDOW_S * SAMPLE_RATE:
        new = " ".join(s.text for s in ASR.transcribe(buf))
        text = (text + " " + new).strip()
        buf = None
    return buf, text, text


def live_flush(buf, text):
    if buf is not None and ASR is not None and len(buf) > SAMPLE_RATE:
        text = (text + " " + " ".join(s.text for s in ASR.transcribe(buf))).strip()
    return None, text, text


# ------------------------------------------------------------------ study tools
def generate(task: str, transcript: str, results: dict):
    if STUDY is None:
        raise gr.Error("Study engine not loaded")
    md, st = "", {}
    for md, st in STUDY.run(task, transcript):
        yield md, results, st or {"status": "generating..."}
    results = dict(results or {})
    results[task] = md
    yield md, results, st


def chat(question: str, history: list, transcript: str):
    history = list(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})
    for ans in STUDY.ask(question, transcript):
        history[-1]["content"] = ans
        yield "", history


def export(transcript_ts: str, results: dict) -> str:
    OUTPUTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = OUTPUTS / f"NotesNPU-{stamp}.md"
    titles = {"summary": "Summary", "notes": "Study Notes", "flashcards": "Flashcards", "quiz": "Quiz"}
    parts = [f"# NotesNPU session - {time.strftime('%d %b %Y, %H:%M')}\n"]
    for k, t in titles.items():
        if results and results.get(k):
            parts.append(f"## {t}\n\n{results[k]}\n")
    parts.append("## Transcript\n\n```\n" + (transcript_ts or "") + "\n```\n")
    parts.append("\n_Generated fully on-device with NotesNPU (Snapdragon NPU + ONNX Runtime)._\n")
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)


def run_bench(with_llm: bool):
    if not SAMPLE.exists():
        raise gr.Error("samples/sample_lecture.wav missing - run: python scripts/make_sample.py")
    res = bench.run(SAMPLE, with_llm)
    return bench.to_markdown(res), res


# ------------------------------------------------------------------ UI
CSS = """
#title {text-align:center}
#title h1 {font-size: 2.1em; margin-bottom: 0}
.status {font-size: 0.9em; opacity: 0.9}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NotesNPU") as demo:
        gr.Markdown("# NotesNPU\n**Private, offline AI lecture & meeting copilot - accelerated by the Snapdragon NPU**",
                    elem_id="title")
        gr.Markdown(status_md(), elem_classes="status")
        results = gr.State({})

        with gr.Tab("1. Transcribe"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Lecture / meeting audio")
                    language = gr.Dropdown(list(LANGUAGES), value="English", label="Spoken language")
                    translate = gr.Checkbox(False, label="Translate to English (e.g. Hindi lecture -> English notes)")
                    with gr.Row():
                        sample_btn = gr.Button("Load sample lecture")
                        go = gr.Button("Transcribe on-device", variant="primary")
                with gr.Column(scale=2):
                    transcript_ts = gr.Textbox(label="Transcript (timestamped)", lines=14, max_lines=30)
                    asr_stats = gr.JSON(label="Performance")
            transcript = gr.Textbox(label="Plain transcript (editable - feeds the study tools)", lines=4)

        with gr.Tab("2. Live mode"):
            gr.Markdown(f"Speak into the microphone - text appears every ~{int(LIVE_WINDOW_S)} s, fully on-device.")
            live_buf = gr.State(None)
            live_text_state = gr.State("")
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Live microphone")
            live_out = gr.Textbox(label="Live transcript", lines=10)
            use_live = gr.Button("Send live transcript to study tools")

        with gr.Tab("3. Study notes"):
            with gr.Row():
                b_sum = gr.Button("Summary", variant="primary")
                b_notes = gr.Button("Structured notes")
                b_cards = gr.Button("Flashcards")
                b_quiz = gr.Button("Quiz")
            study_out = gr.Markdown("_Pick a tool above._")
            llm_stats = gr.JSON(label="Study AI performance")
            with gr.Row():
                exp_btn = gr.Button("Export session (Markdown)")
                exp_file = gr.File(label="Download")

        with gr.Tab("4. Ask the lecture"):
            bot = gr.Chatbot(label="Q&A grounded in your transcript", height=380)
            q = gr.Textbox(placeholder="e.g. When is the exam? What does the Calvin cycle produce?", label="Question")

        with gr.Tab("5. NPU benchmark"):
            gr.Markdown("Runs the sample lecture through every available speech backend (NPU and CPU) and reports "
                        "real-time factor and per-stage latency. Results are saved to `outputs/benchmarks.json`.")
            with_llm = gr.Checkbox(False, label="Also benchmark the study LLM (slower)")
            b_bench = gr.Button("Run benchmark", variant="primary")
            bench_md = gr.Markdown()
            bench_json = gr.JSON(label="Raw results")
            gr.JSON(value=system_summary(), label="System")

        # wiring
        sample_btn.click(lambda: str(SAMPLE) if SAMPLE.exists() else None, outputs=audio)
        go.click(transcribe, [audio, language, translate], [transcript_ts, transcript, asr_stats])
        mic.stream(live_stream, [mic, live_buf, live_text_state], [live_buf, live_text_state, live_out],
                   stream_every=1.0, time_limit=3600)
        mic.stop_recording(live_flush, [live_buf, live_text_state], [live_buf, live_text_state, live_out])
        use_live.click(lambda t: (t, t), live_text_state, [transcript, transcript_ts])
        for btn, task in ((b_sum, "summary"), (b_notes, "notes"), (b_cards, "flashcards"), (b_quiz, "quiz")):
            btn.click(generate, [gr.State(task), transcript, results], [study_out, results, llm_stats])
        exp_btn.click(export, [transcript_ts, results], exp_file)
        q.submit(chat, [q, bot, transcript], [q, bot])
        b_bench.click(run_bench, with_llm, [bench_md, bench_json])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="auto", choices=["auto", "npu", "cpu"])
    ap.add_argument("--no-llm", action="store_true", help="use the fast extractive engine instead of Phi-3.5")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print("Loading on-device models...")
    load_engines(a.asr, not a.no_llm)
    print(json.dumps({"asr": ASR.label if ASR else ASR_ERR, "study": STUDY.label}, indent=2))
    demo = build_ui()
    kwargs = dict(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser, share=False)
    try:
        demo.queue().launch(theme=gr.themes.Soft(primary_hue="red"), css=CSS, **kwargs)
    except TypeError:  # Gradio 4/5 take theme/css on Blocks instead
        demo.theme, demo.css = gr.themes.Soft(primary_hue="red"), CSS
        demo.queue().launch(**kwargs)


if __name__ == "__main__":
    main()
