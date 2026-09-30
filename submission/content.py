"""Shared content for the pitch deck and project description (single source of truth)."""

TITLE = "NotesNPU"
TAGLINE = "Private, offline AI lecture & meeting copilot - accelerated by the Snapdragon NPU on HP PCs"
FORM_TITLE = "NotesNPU: Private, Offline AI Lecture & Meeting Copilot on the Snapdragon NPU"
AUTHOR = "Mohammed Haris Suhaib M"
REPO = "https://github.com/mdharishsuhaib/NotesNPU"

PROBLEM = [
    "Indian students attend 25-30 hours of lectures a week, often in fast, code-mixed Hindi-English, and struggle to take notes and listen at the same time.",
    "Cloud transcription tools (Otter, Fireflies, cloud Copilots) upload sensitive classroom and meeting audio to third-party servers.",
    "They need reliable internet and paid subscriptions, which are hard to get in hostels, tier-2/3 colleges, trains and field work.",
    "Running AI on the CPU drains the battery and heats the laptop, so it can't run through a whole day of classes.",
]

SOLUTION = [
    "Record or upload a lecture and get a timestamped transcript, fully on-device, even in airplane mode.",
    "Qualcomm AI Hub Whisper runs on the Hexagon NPU via ONNX Runtime QNN EP, leaving the CPU free and saving battery.",
    "A local LLM (Phi-3.5-mini INT4) generates a summary, structured notes, flashcards, a quiz and answers questions.",
    "Auto-extracts action items and deadlines ('exam next Monday', 'read chapter 6').",
    "Supports 10+ Indian languages, with Hindi lecture to English notes translation in the same pass.",
]

FEATURES = [
    ("Transcribe", "Upload or record; pause-aware 30 s chunking; timestamps"),
    ("Live mode", "Mic streaming, text every ~8 s, on-device"),
    ("Translate", "Hindi / Tamil / Telugu / Bengali... to English"),
    ("Study notes", "Summary, key concepts, detailed notes, deadlines"),
    ("Revise", "Flashcards and 5-question MCQ quiz"),
    ("Ask the lecture", "Grounded Q&A chat on your transcript"),
    ("Export", "One-click Markdown notes"),
    ("Benchmark", "Built-in NPU vs CPU performance panel"),
]

MODELS = [
    ("Speech-to-text (NPU)", "Whisper-Base, precompiled QNN ONNX FP16", "Qualcomm AI Hub", "ONNX Runtime + QNN EP (HTP) on the Hexagon NPU"),
    ("Speech-to-text (fallback)", "Whisper-Base ONNX (INT8 decoder)", "Hugging Face onnx-community", "ONNX Runtime CPU EP"),
    ("Study assistant", "Phi-3.5-mini-instruct INT4 AWQ", "Microsoft (Hugging Face)", "onnxruntime-genai"),
]

WHY_SNAPDRAGON = [
    "Whisper encoder = dense transformer over 1500 frames, a good fit for the Hexagon HTP's FP16 tensor engines (measured: 45.5 ms per 30 s of audio on X Elite).",
    "AI Hub precompiled context binaries: no on-device compile, instant start-up.",
    "Fixed-shape KV-cache decoder (200 slots) keeps every step on the NPU (3.8 ms/token).",
    "NPU does the speech, so the Oryon CPU stays free for the LLM and UI and live mode never stutters.",
    "Low-power NPU inference means a full day of lectures on battery, quiet and cool.",
]

BENCH_ROWS = [
    ("Snapdragon X Elite, Oryon CPU (measured on Qualcomm AI Hub)", "~14x real time (~6.1 s)", "1286 ms", "9.1 ms"),
    ("Snapdragon X Elite, Hexagon NPU (measured on Qualcomm AI Hub)", "~80x real time (~1.1 s)", "45.5 ms", "3.8 ms"),
    ("NPU speed-up", "~5.7x end to end", "28.3x", "2.4x"),
]

NPU_PROOF = [
    "Measured on a real Snapdragon X Elite (Qualcomm AI Hub hosted device): encoder 45.5 ms per 30 s of audio, decoder 3.8 ms per token.",
    "100% of ops on the NPU: 556 encoder ops + 975 decoder ops on the Hexagon HTP, none on the CPU; peak memory 34 MB / 60 MB.",
    "Accuracy: NPU (FP16) encoder output matches the FP32 CPU reference with cosine similarity 0.999, and gives an identical, correct transcript of the sample lecture.",
    "The 85 s sample lecture takes ~1.1 s on the NPU vs ~6.1 s on the same Snapdragon X Elite's Oryon CPU.",
]

IMPACT = [
    ("Students", "Focus on understanding, not scribbling; revise with auto flashcards and quizzes."),
    ("Accessibility", "Live captions for deaf and hard-of-hearing learners; notes for students with dyslexia or ADHD."),
    ("Regional language learners", "Hindi / regional lectures turned into English notes, bridging the medium-of-instruction gap."),
    ("Professionals", "Confidential meetings summarised with action items, and nothing is uploaded."),
    ("Teachers", "Publish lecture notes and question banks from their own recordings."),
]

DEPLOYMENT = [
    "One-click setup.bat / run.bat: detects Snapdragon, installs native ARM64 deps, downloads AI Hub models.",
    "Runs as a local web app on 127.0.0.1, with offline mode enforced (HF_HUB_OFFLINE, no analytics).",
    "Automatic fallback: NPU to CPU to extractive engine, so it works on every Windows PC.",
    "Light install (-NoLLM, ~300 MB) for low-storage laptops.",
    "Open source (MIT) on GitHub, with CLI tools for transcription and benchmarking.",
]

ROADMAP = [
    "LLM on the NPU: AI Hub Llama-3.2-3B / Phi-3.5 QNN binaries via Genie",
    "Speaker diarisation for meetings (who said what)",
    "Slide / whiteboard capture with on-device OCR, linked to the timeline",
    "Semantic search across a semester of lectures (on-device embeddings)",
    "MSIX installer on the Microsoft Store; HP AI Companion integration",
]

ORIGINALITY = (
    "NotesNPU was designed and built from scratch by the participant during the Challenge Submission Period. "
    "It integrates open models from Qualcomm AI Hub (Whisper) and other open-source platforms (Phi-3.5-mini, "
    "ONNX Whisper) under their respective licenses. The application code is original and MIT-licensed."
)
