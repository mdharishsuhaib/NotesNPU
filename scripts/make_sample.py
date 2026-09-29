"""Generate samples/sample_lecture.wav with Windows' built-in text-to-speech (offline).

Usage: python scripts/make_sample.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "sample_lecture.wav"

LECTURE = (
    "Good morning everyone. Today we will talk about photosynthesis, the process plants use to make food. "
    "Photosynthesis happens mainly in the leaves, inside tiny structures called chloroplasts. "
    "Chloroplasts contain a green pigment called chlorophyll, which absorbs sunlight. "
    "The overall reaction is simple: carbon dioxide plus water, using light energy, produces glucose and oxygen. "
    "There are two main stages. The first stage is the light dependent reactions, which take place in the thylakoid membranes. "
    "Here, light energy splits water molecules, releasing oxygen, and produces A T P and N A D P H. "
    "The second stage is the Calvin cycle, which happens in the stroma. "
    "The Calvin cycle uses A T P and N A D P H to fix carbon dioxide into glucose. "
    "Three factors affect the rate of photosynthesis: light intensity, carbon dioxide concentration, and temperature. "
    "Remember, for your exam next Monday, you must be able to draw and label a chloroplast, "
    "and explain why photosynthesis is important for life on Earth. It produces the oxygen we breathe "
    "and is the base of almost every food chain. Please read chapter six before the next class. Thank you."
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = LECTURE.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 0; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{OUT}', $fmt); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
