# NotesNPU: 2-3 minute demo video script

**Setup:** Snapdragon HP laptop, Wi-Fi **off** (show airplane mode), Task Manager open on the **NPU** graph, app running (`run.bat`).

| Time | Show on screen | Say |
|---|---|---|
| 0:00-0:15 | Title slide / app header | "Hi, I'm Mohammed Haris Suhaib. This is NotesNPU, a private AI note-taker that runs entirely on the Snapdragon NPU of an HP laptop. No internet, no cloud." |
| 0:15-0:30 | Airplane mode icon, status bar reading "Hexagon NPU active" | "Wi-Fi is off. Everything you'll see runs locally." |
| 0:30-1:00 | Load sample lecture, then Transcribe on-device. Split screen: Task Manager NPU graph spikes | "I load an 85-second biology lecture. Qualcomm AI Hub's Whisper runs on the Hexagon NPU, and the whole lecture is transcribed in a couple of seconds, with the CPU nearly idle." |
| 1:00-1:15 | Performance JSON: real-time factor, encoder ms | "That's roughly 100x faster than real time, so a one-hour class takes under a minute, on battery." |
| 1:15-1:45 | Study notes: Structured notes, then Quiz | "One click gives structured notes, key concepts, and even the deadlines the teacher mentioned: exam next Monday, read chapter 6. Then a quiz to revise." |
| 1:45-2:05 | Ask the lecture: "When is the exam?" | "I can ask questions, and answers come only from my lecture." |
| 2:05-2:20 | Transcribe tab: Hindi clip, Translate to English | "Hindi lecture? Tick translate and get English notes. That matters for millions of Indian students." |
| 2:20-2:35 | NPU benchmark tab, NPU vs CPU table | "The built-in benchmark shows the NPU speed-up over the CPU." |
| 2:35-2:50 | GitHub README | "It's open source, with one-click setup. NotesNPU: learn more, write less, and keep it on your PC. Thank you!" |

**Recording tips:** Win+Alt+R (Xbox Game Bar) or OBS; 1080p; plug in the headset mic; do one full dry run first so the models are warm.
