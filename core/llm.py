"""On-device LLM study tools (summary, notes, flashcards, quiz, Q&A) via onnxruntime-genai.

The LLM is Phi-3.5-mini-instruct INT4 (ONNX). If it is not downloaded - or the machine
has too little RAM - NotesNPU falls back to a fast *extractive* engine (TF-IDF style
sentence ranking) so the app is always usable.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

SYSTEM = (
    "You are NotesNPU, a precise study assistant. Use ONLY facts from the transcript. "
    "Be concise and well structured. Write in English using Markdown."
)

TASKS = {
    "summary": (
        "Summarise this lecture/meeting transcript in 5-7 bullet points, then give a one-line TL;DR.\n\n"
        "Transcript:\n{t}"
    ),
    "notes": (
        "Turn this transcript into structured study notes in Markdown with: '## Key Concepts' "
        "(term: short definition), '## Detailed Notes' (headed sections with bullets), "
        "'## Action Items / Deadlines' (anything the speaker asked listeners to do; write 'None' if absent).\n\n"
        "Transcript:\n{t}"
    ),
    "flashcards": (
        "Create 6 flashcards from this transcript. Format each exactly as:\n"
        "**Q:** question\n**A:** answer\n\nTranscript:\n{t}"
    ),
    "quiz": (
        "Write a 5-question multiple-choice quiz on this transcript. For each question give options "
        "A-D on separate lines, then a line '**Answer:** <letter> - <one-line explanation>'.\n\n"
        "Transcript:\n{t}"
    ),
}

CHAT = "Answer the question using only the transcript. If the answer is not in it, say so.\n\nTranscript:\n{t}\n\nQuestion: {q}"


def _phi_dir() -> Path | None:
    for p in sorted(MODELS.glob("phi-*")) if MODELS.exists() else []:
        if (p / "genai_config.json").exists():
            return p
    return None


class LLM:
    """Thin streaming wrapper around onnxruntime-genai (supports both old and new generator APIs)."""

    def __init__(self, model_dir: Path, max_ctx_chars: int = 9000):
        import onnxruntime_genai as og

        self.og = og
        self.dir = model_dir
        self.model = og.Model(str(model_dir))
        self.tok = og.Tokenizer(self.model)
        self.max_ctx_chars = max_ctx_chars  # ~2.3k tokens of transcript keeps latency reasonable
        cfg = json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))
        prov = cfg.get("model", {}).get("decoder", {}).get("session_options", {}).get("provider_options", [])
        ep = next(iter(prov[0]), "cpu") if prov else "cpu"
        self.label = f"{model_dir.name} | onnxruntime-genai ({ep.upper() if ep != 'cpu' else 'CPU'})"

    def _prompt(self, user: str) -> str:
        return f"<|system|>\n{SYSTEM}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"

    def stream(self, user: str, max_new_tokens: int = 700) -> Iterator[str]:
        og = self.og
        ids = self.tok.encode(self._prompt(user))
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(ids) + max_new_tokens, do_sample=False, repetition_penalty=1.05)
        gen = og.Generator(self.model, params)
        if hasattr(gen, "append_tokens"):
            gen.append_tokens(ids)
        else:  # onnxruntime-genai < 0.6
            params.input_ids = ids
            gen = og.Generator(self.model, params)
        ts = self.tok.create_stream()
        while not gen.is_done():
            if hasattr(gen, "compute_logits"):
                gen.compute_logits()
            gen.generate_next_token()
            yield ts.decode(gen.get_next_tokens()[0])
        del gen

    def fit(self, transcript: str) -> str:
        if len(transcript) <= self.max_ctx_chars:
            return transcript
        # keep beginning and end (intros state topics; endings state homework/deadlines)
        half = self.max_ctx_chars // 2
        return transcript[:half] + "\n...\n" + transcript[-half:]


# ---------------------------------------------------------------- extractive fallback
STOP = set(
    "a an the and or but if then so of to in on at for from by with as is are was were be been being this that these "
    "those it its we you they he she i me my our your their them us will would can could should may might must do does "
    "did have has had not no yes very just also about into over under than too more most some any each all which who "
    "whom what when where why how there here up down out off again once today okay ok um uh like going thing things "
    "please thank thanks good morning everyone let lets".split()
)


def _sentences(text: str) -> list[str]:
    s = re.split(r"(?<=[.!?])\s+", text.strip())
    return [x.strip() for x in s if len(x.split()) >= 4]


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", s.lower()) if w not in STOP and len(w) > 2]


def _rank(text: str, k: int) -> list[str]:
    sents = _sentences(text)
    if not sents:
        return []
    tf = Counter(w for s in sents for w in _words(s))
    df = Counter(w for s in sents for w in set(_words(s)))
    n = len(sents)

    def score(s: str) -> float:
        ws = _words(s)
        return sum(tf[w] * math.log(1 + n / df[w]) for w in ws) / (len(ws) + 3) if ws else 0.0

    top = sorted(range(n), key=lambda i: -score(sents[i]))[:k]
    return [sents[i] for i in sorted(top)]


def _keywords(text: str, k: int = 8) -> list[str]:
    c = Counter(_words(text))
    return [w for w, _ in c.most_common(k)]


ACTION_RX = re.compile(r"\b(remember|must|homework|assignment|deadline|exam|test|submit|read|due|next (class|week|monday|tuesday|wednesday|thursday|friday)|make sure|don't forget)\b", re.I)


def extractive(task: str, transcript: str) -> str:
    sents = _sentences(transcript)
    if not sents:
        return "_Transcript too short._"
    if task == "summary":
        pts = _rank(transcript, 6)
        return "\n".join(f"- {p}" for p in pts) + f"\n\n**TL;DR:** {pts[0] if pts else sents[0]}"
    if task == "notes":
        kws = _keywords(transcript)
        concept_lines = []
        for kw in kws:
            s = next((x for x in sents if kw in x.lower()), "")
            concept_lines.append(f"- **{kw.capitalize()}**: {s}")
        actions = [s for s in sents if ACTION_RX.search(s)]
        body = "\n".join(f"- {s}" for s in _rank(transcript, 10))
        return ("## Key Concepts\n" + "\n".join(concept_lines) + "\n\n## Detailed Notes\n" + body +
                "\n\n## Action Items / Deadlines\n" + ("\n".join(f"- {a}" for a in actions) or "- None"))
    if task == "flashcards":
        cards = []
        for kw in _keywords(transcript, 6):
            s = next((x for x in sents if kw in x.lower()), None)
            if s:
                cards.append(f"**Q:** What does the lecture say about *{kw}*?\n**A:** {s}")
        return "\n\n".join(cards)
    if task == "quiz":
        out = []
        kws = _keywords(transcript, 12)
        for i, s in enumerate(_rank(transcript, 5), 1):
            present = [k for k in kws if k in s.lower()]
            if not present:
                continue
            ans = present[0]
            distract = [k for k in kws if k != ans and k not in s.lower()][:3]
            opts = sorted([ans, *distract])
            blank = re.sub(re.escape(ans), "_____", s, flags=re.I)
            letters = "ABCD"
            lines = [f"**{i}.** {blank}"] + [f"{letters[j]}) {o}" for j, o in enumerate(opts)]
            lines.append(f"**Answer:** {letters[opts.index(ans)]} - {ans}")
            out.append("\n".join(lines))
        return "\n\n".join(out) or "_Not enough content for a quiz._"
    return ""


def extractive_answer(question: str, transcript: str) -> str:
    q = set(_words(question))
    sents = _sentences(transcript)
    scored = sorted(sents, key=lambda s: -len(q & set(_words(s))))
    best = [s for s in scored[:3] if q & set(_words(s))]
    return ("From the transcript:\n" + "\n".join(f"> {s}" for s in best)) if best else "I couldn't find that in the transcript."


def clean_md(s: str) -> str:
    """LLMs often indent their whole answer (-> Markdown code block) or wrap it in ``` fences."""
    import textwrap

    s = s.replace("\r", "")
    s = re.sub(r"^\s*```(?:markdown|md)?\s*\n", "", s)
    s = re.sub(r"\n```\s*$", "", s)
    s = textwrap.dedent(s).strip()
    out = []
    for ln in s.split("\n"):
        st = ln.lstrip()
        # Keep indentation only for nested list items; anything else indented 4+ would render as code.
        if re.match(r"([-*+]|\d+[.)])\s", st) or not st:
            out.append(ln if len(ln) - len(st) <= 8 else "    " + st)
        else:
            out.append(st)
    return "\n".join(out)


# ---------------------------------------------------------------- public API
class StudyEngine:
    def __init__(self, use_llm: bool = True):
        self.llm: LLM | None = None
        self.error = ""
        d = _phi_dir() if use_llm else None
        if d is not None:
            try:
                self.llm = LLM(d)
            except Exception as e:  # missing onnxruntime-genai wheel, OOM, ...
                self.error = str(e)
        self.label = self.llm.label if self.llm else "Extractive engine (no LLM loaded)"

    def run(self, task: str, transcript: str) -> Iterator[tuple[str, dict]]:
        """Yields (markdown_so_far, stats)."""
        t0 = time.perf_counter()
        if not transcript.strip():
            yield "_Transcribe something first._", {}
            return
        if self.llm is None:
            md = extractive(task, transcript)
            yield md, {"engine": self.label, "seconds": round(time.perf_counter() - t0, 2)}
            return
        text, n, first = "", 0, None
        for piece in self.llm.stream(TASKS[task].format(t=self.llm.fit(transcript))):
            if first is None:
                first = time.perf_counter() - t0
            text += piece
            n += 1
            if n % 4 == 0:
                yield clean_md(text), {}
        dt = time.perf_counter() - t0
        gen_t = dt - (first or 0)
        yield clean_md(text), {"engine": self.label, "seconds": round(dt, 2), "time_to_first_token_s": round(first or 0, 2),
                     "tokens": n, "tokens_per_s": round(n / gen_t, 1) if gen_t > 0 else 0}

    def ask(self, question: str, transcript: str) -> Iterator[str]:
        if not transcript.strip():
            yield "_Transcribe something first._"
            return
        if self.llm is None:
            yield extractive_answer(question, transcript)
            return
        text = ""
        for piece in self.llm.stream(CHAT.format(t=self.llm.fit(transcript), q=question), max_new_tokens=300):
            text += piece
            yield clean_md(text)


if __name__ == "__main__":  # python -m core.llm transcript.txt notes
    import sys

    tr = Path(sys.argv[1]).read_text(encoding="utf-8")
    eng = StudyEngine(use_llm="--no-llm" not in sys.argv)
    print("Engine:", eng.label, eng.error)
    md, st = "", {}
    for md, st in eng.run(sys.argv[2] if len(sys.argv) > 2 else "summary", tr):
        pass
    print(md)
    print(st)
