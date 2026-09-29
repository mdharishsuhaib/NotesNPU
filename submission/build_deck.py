"""Generate submission/NotesNPU_Pitch.pptx (and .pdf if PowerPoint is installed).

python submission/build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

RED = RGBColor(0xE4, 0x1E, 0x2B)      # Snapdragon-ish red
DARK = RGBColor(0x14, 0x17, 0x1F)
GREY = RGBColor(0x5A, 0x60, 0x6B)
LIGHT = RGBColor(0xF4, 0xF5, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, s, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT
    return tb


def bullets(slide, x, y, w, h, items, size=18, color=DARK, gap=10):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r1 = p.add_run()
        r1.text = "\u25A0  "
        r1.font.size, r1.font.color.rgb, r1.font.name = Pt(size - 6), RED, FONT
        r2 = p.add_run()
        r2.text = it
        r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size), color, FONT
    return tb


def header(slide, n, title, kicker=""):
    rect(slide, 0, 0, W, H, WHITE)
    rect(slide, 0, 0, Inches(0.18), H, RED)
    if kicker:
        text(slide, Inches(0.6), Inches(0.35), Inches(10), Inches(0.4), kicker.upper(), 12, True, RED)
    text(slide, Inches(0.6), Inches(0.62), Inches(12), Inches(0.9), title, 32, True, DARK)
    text(slide, Inches(11.3), Inches(7.0), Inches(1.8), Inches(0.35), f"NotesNPU  |  {n}", 10, False, GREY, PP_ALIGN.RIGHT)


def table(slide, x, y, w, rows, col_w, header_row=True, size=13, row_h=Inches(0.5)):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, row_h * len(rows))
    tbl = shp.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = cw
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size = Pt(size)
            para.runs[0].font.name = FONT
            is_h = header_row and i == 0
            para.runs[0].font.bold = is_h
            para.runs[0].font.color.rgb = WHITE if is_h else DARK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RED if is_h else (LIGHT if i % 2 else WHITE)
    return tbl


# 1. Title ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, DARK)
rect(s, 0, Inches(5.2), W, Inches(0.08), RED)
text(s, Inches(0.8), Inches(1.6), Inches(11), Inches(1.4), C.TITLE, 72, True, WHITE)
text(s, Inches(0.8), Inches(3.0), Inches(11.5), Inches(1.2), C.TAGLINE, 26, False, RGBColor(0xDD, 0xDD, 0xE3))
text(s, Inches(0.8), Inches(4.2), Inches(11.5), Inches(0.6),
     "Qualcomm AI Hub Whisper on the Hexagon NPU  +  on-device LLM  |  100% offline", 18, True, RED)
text(s, Inches(0.8), Inches(5.6), Inches(11.5), Inches(1.0),
     [f"{C.AUTHOR}  |  Snapdragon AI Lab Build & Present Challenge 2026", C.REPO], 16, False, RGBColor(0xBB, 0xBB, 0xC4))

# 2. Problem -------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 2, "Taking notes while learning is broken", "The problem")
bullets(s, Inches(0.6), Inches(1.7), Inches(7.6), Inches(5), C.PROBLEM, 19)
for i, (big, small) in enumerate([("25-30 h", "lectures / week per student"), ("0", "bytes of audio should leave the laptop"), ("1 day", "battery needed for back-to-back classes")]):
    y = Inches(1.8 + i * 1.7)
    rect(s, Inches(8.7), y, Inches(4.0), Inches(1.45), LIGHT)
    text(s, Inches(8.9), y + Inches(0.1), Inches(3.7), Inches(0.7), big, 34, True, RED)
    text(s, Inches(8.9), y + Inches(0.85), Inches(3.7), Inches(0.5), small, 14, False, GREY)

# 3. Solution ------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 3, "NotesNPU: your private AI note-taker", "The solution")
bullets(s, Inches(0.6), Inches(1.7), Inches(6.4), Inches(5), C.SOLUTION, 18)
for i, (name, desc) in enumerate(C.FEATURES):
    col, row = i % 2, i // 2
    x, y = Inches(7.3 + col * 2.95), Inches(1.75 + row * 1.3)
    rect(s, x, y, Inches(2.8), Inches(1.15), LIGHT)
    text(s, x + Inches(0.12), y + Inches(0.08), Inches(2.6), Inches(0.4), name, 15, True, RED)
    text(s, x + Inches(0.12), y + Inches(0.45), Inches(2.6), Inches(0.7), desc, 11, False, DARK)

# 4. Demo ----------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 4, "Live demo: lecture in, study notes out", "Product")
shots = [ROOT / "docs" / "screenshots" / n for n in ("transcribe.png", "notes.png")]
x = Inches(0.6)
for p in shots:
    if p.exists():
        pic = s.shapes.add_picture(str(p), x, Inches(1.5), height=Inches(4.9))
        x = pic.left + pic.width + Inches(0.3)
text(s, Inches(0.6), Inches(6.5), Inches(12.2), Inches(0.5),
     "1. Load or record a lecture   >   2. Transcribe on-device (NPU)   >   3. Summary / notes / flashcards / quiz   >   4. Ask & export",
     15, True, RED)

# 5. Architecture --------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 5, "Architecture: everything runs on the PC", "Technical implementation")
steps = [("Audio", "mic / file\n16 kHz mono"), ("Chunker", "pause-aware\n<=30 s + silence gate"),
         ("Whisper encoder", "AI Hub QNN ONNX\nHexagon NPU"), ("Whisper decoder", "KV-cache, 200 slots\nHexagon NPU"),
         ("Phi-3.5-mini", "INT4, onnxruntime-genai"), ("Study outputs", "summary, notes,\nquiz, Q&A, export")]
bw, gap = Inches(1.9), Inches(0.22)
for i, (t, d) in enumerate(steps):
    x = Inches(0.6) + i * (bw + gap)
    npu = "NPU" in d
    box = rect(s, x, Inches(2.0), bw, Inches(1.7), RED if npu else LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + Inches(0.1), Inches(2.1), bw - Inches(0.2), Inches(0.5), t, 15, True, WHITE if npu else DARK, PP_ALIGN.CENTER)
    text(s, x + Inches(0.1), Inches(2.65), bw - Inches(0.2), Inches(1.0), d.split("\n"), 12, False, WHITE if npu else GREY, PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + bw + Inches(0.02), Inches(2.72), gap - Inches(0.04), Inches(0.26))
        a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()
bullets(s, Inches(0.6), Inches(4.2), Inches(12.2), Inches(3), [
    "ONNX Runtime QNN Execution Provider (plugin EP, HTP backend, burst perf mode) runs AI Hub precompiled context binaries, so there is no on-device compile.",
    "Encoder emits cross-attention KV caches directly; the fixed-shape decoder loop feeds self-attention caches back, so every step stays on the NPU.",
    "Robust by design: NPU to CPU to extractive fallback chain, repetition guard, silence gating, head+tail context fitting for long lectures.",
], 15)

# 6. Models + Why Snapdragon --------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 6, "Why the Snapdragon NPU", "AI models & optimisation")
table(s, Inches(0.6), Inches(1.6), Inches(12.1), [("Task", "Model", "Source", "Runtime / hardware"), *C.MODELS],
      [Inches(2.4), Inches(3.4), Inches(2.6), Inches(3.7)], size=12, row_h=Inches(0.45))
bullets(s, Inches(0.6), Inches(3.75), Inches(12.2), Inches(3.4), C.WHY_SNAPDRAGON, 15, gap=6)

# 7. Benchmarks ----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 7, "Performance: NPU vs CPU", "Benchmarks (Whisper-Base, 85 s lecture)")
table(s, Inches(0.6), Inches(1.7), Inches(12.1), [("Device / backend", "Speed", "Encoder per 30 s", "Decoder per token"), *C.BENCH_ROWS],
      [Inches(5.2), Inches(2.4), Inches(2.2), Inches(2.3)], size=15, row_h=Inches(0.6))
rect(s, Inches(0.6), Inches(4.6), Inches(12.1), Inches(1.9), LIGHT)
text(s, Inches(0.9), Inches(4.75), Inches(11.6), Inches(1.7), [
    "AI Hub reference: ~28x faster encoder and ~12x faster decoding on the NPU than my laptop-CPU baseline",
    "A 1-hour lecture transcribes in well under a minute, on battery, with the CPU left free.",
    "Reproduce: python -m core.bench --llm   (built-in benchmark tab in the app)"], 17, False, DARK)

# 8. Impact --------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 8, "Who it helps", "Use case & innovation")
for i, (who, what) in enumerate(C.IMPACT):
    y = Inches(1.65 + i * 1.05)
    rect(s, Inches(0.6), y, Inches(3.2), Inches(0.9), RED)
    text(s, Inches(0.7), y + Inches(0.22), Inches(3.0), Inches(0.5), who, 17, True, WHITE, PP_ALIGN.CENTER)
    rect(s, Inches(3.8), y, Inches(8.9), Inches(0.9), LIGHT)
    text(s, Inches(4.0), y + Inches(0.2), Inches(8.6), Inches(0.6), what, 16, False, DARK)

# 9. Deployment ----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 9, "Deployment & accessibility", "Ready to use")
bullets(s, Inches(0.6), Inches(1.7), Inches(7.2), Inches(5), C.DEPLOYMENT, 17)
rect(s, Inches(8.2), Inches(1.7), Inches(4.5), Inches(4.3), DARK)
text(s, Inches(8.45), Inches(1.9), Inches(4.1), Inches(4.0), [
    "> git clone <repo>", "> cd notesnpu", "> setup.bat", "> run.bat", "", "Opens http://127.0.0.1:7860", "Works offline after setup"],
     16, False, RGBColor(0xE6, 0xE6, 0xEA))

# 10. Roadmap ------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
header(s, 10, "Roadmap", "What's next")
bullets(s, Inches(0.6), Inches(1.7), Inches(12), Inches(5), C.ROADMAP, 20)

# 11. Thank you ----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, DARK)
rect(s, 0, Inches(4.6), W, Inches(0.08), RED)
text(s, Inches(0.8), Inches(2.0), Inches(11.5), Inches(1.2), "Thank you", 60, True, WHITE)
text(s, Inches(0.8), Inches(3.3), Inches(11.5), Inches(1.0), "NotesNPU: learn more, write less, and keep it on your PC.", 24, False, RGBColor(0xDD, 0xDD, 0xE3))
text(s, Inches(0.8), Inches(5.0), Inches(11.5), Inches(1.2), [C.REPO, C.AUTHOR], 18, False, RGBColor(0xBB, 0xBB, 0xC4))

out = HERE / "NotesNPU_Pitch.pptx"
prs.save(out)
print("wrote", out)


def to_pdf(pptx: Path) -> None:
    try:
        import comtypes.client  # type: ignore

        app = comtypes.client.CreateObject("PowerPoint.Application")
        deck = app.Presentations.Open(str(pptx), WithWindow=False)
        deck.SaveAs(str(pptx.with_suffix(".pdf")), 32)
        deck.Close()
        app.Quit()
        print("wrote", pptx.with_suffix(".pdf"))
    except Exception as e:
        print(f"(PDF export skipped: {e}). Open the PPTX in PowerPoint > File > Export > PDF, "
              "or run: soffice --headless --convert-to pdf NotesNPU_Pitch.pptx")


if "--pdf" in sys.argv:
    to_pdf(out)
