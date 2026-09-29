# Unstop submission form: copy/paste values

Portal: *Snapdragon AI Lab Build & Present Challenge - Solution Submission Round*.
Deadline: **30 Sep 2026, 11:59 PM IST**. Submissions **cannot be edited** after submitting.

---

### Project Title  (limit 500 chars)

```
NotesNPU: Private, Offline AI Lecture & Meeting Copilot on the Snapdragon NPU
```

### Brief Project Description  (upload)

Upload `submission/NotesNPU_Description.pdf` (or `.docx`).

### GitHub Repository Link  (limit 500 chars)

```
https://github.com/<your-username>/notesnpu
```

### Short Pitch Presentation in PDF

Upload `submission/NotesNPU_Pitch.pdf`

### Short Pitch Presentation in PPT

Upload `submission/NotesNPU_Pitch.pptx`

### You have a Snapdragon laptop

Select **Yes** and tick the confirmation checkbox.

---

### Backup: 500-character description (in case a text field asks for one)

```
NotesNPU turns lectures and meetings into transcripts, summaries, study notes, flashcards, quizzes and Q&A, 100% offline on Snapdragon HP PCs. Qualcomm AI Hub Whisper runs on the Hexagon NPU via ONNX Runtime QNN EP (~100x real time); Phi-3.5-mini INT4 (onnxruntime-genai) writes the notes. It supports Hindi and other Indian languages with English translation. No audio leaves the PC: private, works without internet, battery-friendly. One-click setup, open source.
```

### Pre-submit checklist

- [ ] Replace `<your-username>` in `submission/content.py`, `README.md` and this file, then rebuild the deck and description
- [ ] Put your name in `AUTHOR` in `submission/content.py`
- [ ] (Recommended) Run `python -m core.bench --llm` on your Snapdragon PC and put the NPU numbers into `BENCH_ROWS`
- [ ] Rebuild: `python submission/build_deck.py --pdf` and `python submission/build_description.py --pdf`
- [ ] Open both PDFs and check them
- [ ] GitHub repo is **public** and the README renders
- [ ] Submit well before 11:59 PM IST
