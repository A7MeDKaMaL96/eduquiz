"""
gemini_question_extractor.py
EduQuiz — AI-assisted question extraction using Gemini Flash.

v3 — section-aware extraction.

WHY THIS VERSION EXISTS
Real combined question banks (e.g. NASS_ACADEMY...Collective_Bank) mix many
question formats in one PDF: open Q&A pairs, fill-in-the-blank, matching,
code-debugging, scientific-term, PLUS the MCQ/True-False blocks this system
actually supports — PLUS trailing TPK/lesson-code index tables that aren't
questions at all. Feeding the whole raw text to one LLM call in one shot
caused two failures:
  1. The model lost track of which line was a question vs. an option in
     dense MCQ tables (a stray row-number from a multi-line table cell
     landing on an option line, e.g. "44 b) Hypertext Preprocessor"),
     producing single-fragment "questions" with no stem and no options.
  2. Large documents risk exceeding Gemini's per-request output token
     budget once you ask it to emit hundreds of structured items at once.

FIX: split the raw text into sections by their own heading line (e.g.
"Dear learner: Choose the correct answer (7 questions)."), classify each
section by its heading text, and ONLY send MCQ and True/False sections to
Gemini — each in small batches, isolated from unrelated content. Everything
else (open Q&A, fill-in-the-blank, matching, code-debug, scientific-term,
and the trailing TPK index tables, which never have a recognized heading)
is skipped entirely and never reaches the model.

WORKFLOW
  1. Extract raw text (reuses MCQExtractor.extract_from_pdf from pdf_extract.py).
  2. Split into sections by heading line; classify each as mcq / true_false / skip.
  3. Batch same-type sections (char-budget capped) and call Gemini per batch.
  4. Normalize + merge + dedupe all batch results.
  5. On ANY failure -> {"success": false, ...}, never raises, never exits
     non-zero, so QuestionExtractor.php can fall back to the legacy pipeline.

USAGE
  python gemini_question_extractor.py <file_path>

OUTPUT: single line of JSON on stdout. Diagnostics go to stderr only.
"""
import sys
import os
import io
import json
import re

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pdf_extract import MCQExtractor
except Exception as _import_err:
    MCQExtractor = None
    print(f"[gemini_extractor] could not import pdf_extract.py: {_import_err}", file=sys.stderr)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = 60
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

MAX_BATCH_CHARS = 4500  # keeps each request's output well under the model's token budget


def load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"[gemini_extractor] failed to read .env: {e}", file=sys.stderr)


def get_api_key():
    load_env_file()
    return os.environ.get("GEMINI_API_KEY")


def extract_raw_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if MCQExtractor is None:
        raise RuntimeError("pdf_extract.py text extraction unavailable")
    return MCQExtractor().extract_from_pdf(path)


# --- Section splitting -------------------------------------------------

HEADER_RE = re.compile(
    r'^(Dear\s+Learner\s*:.*|Dear\s+learner\s*:.*|Give\s+the\s+Scientific\s+Name.*|'
    r'Match\s+the\s+.*|Match\s+each\s+.*|Test\s+Code\s+Debugging.*)',
    re.IGNORECASE
)


def split_sections(raw_text: str):
    """Returns a list of (heading, body_text) tuples. Text before the first
    heading (cover pages, TPK/lesson mapping tables) is discarded."""
    lines = raw_text.split("\n")
    sections = []
    current_heading = None
    current_body = []

    for line in lines:
        if HEADER_RE.match(line.strip()):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = line.strip()
            current_body = []
        elif current_heading is not None:
            current_body.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_body)))

    return sections


def classify_section(heading: str) -> str:
    h = heading.lower()
    if "choose the correct answer" in h:
        return "mcq"
    if ('"true"' in h and '"false"' in h) or ("true" in h and "false" in h and "front" in h):
        return "true_false"
    return "skip"


def make_batches(sections, kind: str, max_chars: int = MAX_BATCH_CHARS):
    """Groups consecutive same-kind sections into batches capped by max_chars."""
    batches = []
    current = []
    current_len = 0
    for heading, body in sections:
        if classify_section(heading) != kind:
            continue
        block = f"{heading}\n{body}\n"
        if current and current_len + len(block) > max_chars:
            batches.append("\n".join(current))
            current, current_len = [], 0
        current.append(block)
        current_len += len(block)
    if current:
        batches.append("\n".join(current))
    return batches


# --- Gemini prompts ------------------------------------------------------

MCQ_PROMPT = """You are an exam-question parser. Below is raw text extracted from a PDF table containing ONLY multiple-choice questions. The text may be mixed Arabic/English and may have minor layout noise from the PDF table export.

IMPORTANT - table export quirk you must handle:
- Each MCQ block looks like: a question stem line (which often ends with a trailing single answer letter, e.g. "What does PHP stand for? b"), followed by four option lines starting with a), b), c), d).
- A stray row NUMBER from the table's "No" column sometimes lands in front of one of the OPTION lines instead of the question line (e.g. "44 b) Hypertext Preprocessor"). That number is just a row label - ignore it, and never treat an option line (a line starting with a)/b)/c)/d)) as if it were the question itself.
- The question stem is always the line BEFORE the first a)/b)/c)/d) line. The trailing single letter on that stem line (if present) is the answer, not part of the question text - strip it from the question text.

Extract EVERY multiple-choice question in this text, in order, with no omissions.

Return ONLY valid JSON (no markdown fences, no commentary):
{
  "questions": [
    {
      "type": "multiple_choice",
      "question": "string (question stem only, no trailing answer letter, no row number)",
      "options": ["string", "string", "string", "string"],
      "answer": "A" | "B" | "C" | "D" | null
    }
  ]
}

Rules:
- "answer" is the option LETTER (A/B/C/D), matching position in "options".
- If no answer is explicitly given, use null. Never guess.
- Preserve Arabic text as-is.
- Return JSON only.

RAW TEXT:
---
{raw_text}
---
"""

TF_PROMPT = """You are an exam-question parser. Below is raw text extracted from a PDF table containing ONLY True/False statements, one per line, in the format: "<row number> <statement>. <True or False>".

Extract EVERY statement in this text, in order, with no omissions.

Return ONLY valid JSON (no markdown fences, no commentary):
{
  "questions": [
    {
      "type": "true_false",
      "question": "string (the statement only, no row number, no trailing True/False)",
      "options": ["True", "False"],
      "answer": "True" | "False" | null
    }
  ]
}

Rules:
- If no answer is explicitly given, use null. Never guess.
- Preserve Arabic text as-is.
- Return JSON only.

RAW TEXT:
---
{raw_text}
---
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING", "enum": ["multiple_choice", "true_false"]},
                    "question": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "answer": {"type": "STRING"},
                },
                "required": ["type", "question", "options"],
            },
        }
    },
    "required": ["questions"],
}


import time


def salvage_partial_json(text_out: str) -> dict:
    """If the model's output got truncated mid-array, recover whatever complete
    {...} question objects appear before the cut-off point instead of losing
    the whole batch. Captures brace-matched substrings at ANY nesting depth
    (question objects are nested inside {"questions": [...]})."""
    objs = []
    stack = []
    for i, ch in enumerate(text_out):
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if not stack:
                continue
            start = stack.pop()
            chunk = text_out[start:i + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict) and "type" in obj and "question" in obj:
                    objs.append(obj)
            except Exception:
                pass
    return {"questions": objs}


def call_gemini(api_key: str, prompt_text: str, max_retries: int = 4) -> dict:
    import requests

    body = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "maxOutputTokens": 16384,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    delay = 5
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_API_URL, params={"key": api_key}, json=body, timeout=GEMINI_TIMEOUT_SECONDS)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"[gemini_extractor] 429 rate limited, retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"[gemini_extractor] batch raw output ({len(text_out)} chars): {text_out[:1500]}", file=sys.stderr)
            try:
                return json.loads(text_out)
            except json.JSONDecodeError as je:
                salvaged = salvage_partial_json(text_out)
                print(f"[gemini_extractor] JSON truncated ({je}), salvaged "
                      f"{len(salvaged['questions'])} complete item(s) from this batch", file=sys.stderr)
                return salvaged
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < max_retries - 1:
                print(f"[gemini_extractor] network error ({e.__class__.__name__}), retrying in {delay}s "
                      f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1 and "429" in str(e):
                time.sleep(delay)
                delay *= 2
                continue
            raise

    raise last_err if last_err else RuntimeError("Gemini call failed after retries")


def _norm_type(value) -> str:
    v = (str(value) if value is not None else "").strip().lower().replace("-", "_").replace(" ", "_")
    if v in ("multiple_choice", "mcq", "multiplechoice"):
        return "multiple_choice"
    if v in ("true_false", "tf", "truefalse", "true/false"):
        return "true_false"
    return ""


def normalize_questions(parsed: dict) -> list:
    out, skipped = [], 0
    for q in parsed.get("questions", []):
        qtype = _norm_type(q.get("type"))
        question = (q.get("question") or "").strip()
        options = q.get("options") or []
        answer = q.get("answer")
        if isinstance(answer, str):
            answer = answer.strip()

        if not question or not qtype:
            skipped += 1
            continue

        if qtype == "true_false":
            ans = answer.capitalize() if isinstance(answer, str) else None
            out.append({"type": "true_false", "question": question,
                        "options": ["True", "False"], "answer": ans if ans in ("True", "False") else None})
        else:
            if len(options) < 2:
                skipped += 1
                continue
            ans = answer.upper() if isinstance(answer, str) else None
            out.append({"type": "multiple_choice", "question": question,
                        "options": [str(o).strip() for o in options],
                        "answer": ans if ans in ("A", "B", "C", "D") else None})

    if skipped:
        print(f"[gemini_extractor] skipped {skipped} malformed item(s)", file=sys.stderr)
    return out


def dedupe(questions: list) -> list:
    seen, out = set(), []
    for q in questions:
        key = q["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def fail(error: str) -> None:
    print(json.dumps({"success": False, "error": error, "questions": [], "source": "gemini"}, ensure_ascii=False))


def main():
    if len(sys.argv) < 2:
        fail("No file path")
        return

    path = sys.argv[1]
    try:
        api_key = get_api_key()
        if not api_key:
            fail("GEMINI_API_KEY not configured")
            return

        raw_text = extract_raw_text(path)
        if not raw_text or not raw_text.strip():
            fail("No extractable text")
            return

        sections = split_sections(raw_text)
        mcq_batches = make_batches(sections, "mcq")
        tf_batches = make_batches(sections, "true_false")

        print(f"[gemini_extractor] {len(sections)} section(s) found; "
              f"{len(mcq_batches)} MCQ batch(es), {len(tf_batches)} True/False batch(es)", file=sys.stderr)

        all_questions = []
        for i, batch_text in enumerate(mcq_batches):
            try:
                if i > 0:
                    time.sleep(2)
                prompt = MCQ_PROMPT.replace("{raw_text}", batch_text[:MAX_BATCH_CHARS + 500])
                parsed = call_gemini(api_key, prompt)
                all_questions.extend(normalize_questions(parsed))
            except Exception as e:
                print(f"[gemini_extractor] MCQ batch failed, skipping batch: {e}", file=sys.stderr)

        for i, batch_text in enumerate(tf_batches):
            try:
                if i > 0 or mcq_batches:
                    time.sleep(2)
                prompt = TF_PROMPT.replace("{raw_text}", batch_text[:MAX_BATCH_CHARS + 500])
                parsed = call_gemini(api_key, prompt)
                all_questions.extend(normalize_questions(parsed))
            except Exception as e:
                print(f"[gemini_extractor] T/F batch failed, skipping batch: {e}", file=sys.stderr)

        all_questions = dedupe(all_questions)

        if not all_questions:
            fail("Gemini returned no usable questions")
            return

        print(json.dumps(
            {"success": True, "text": raw_text[:1000], "questions": all_questions, "source": "gemini"},
            ensure_ascii=False
        ))

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        fail(str(e))


if __name__ == "__main__":
    main()