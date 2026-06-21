"""
app.py — Flask wrapper around gemini_question_extractor.py

This file does NOT change your existing extraction logic at all. It just
gives that logic a "front door" that can answer web requests, instead of
only working when you type a command in a terminal.

Two web addresses (routes) this creates:
  GET  /health   -> just says "I'm alive", used to check the service is up
  POST /extract  -> the real one: send it a file, get back questions

HOW TO RUN THIS ON YOUR OWN COMPUTER (before we put it on the internet):
  python app.py
Then it will print something like "Running on http://127.0.0.1:5000"
Leave that window open, and we'll test it from a second window.
"""
import os
import tempfile
from flask import Flask, request, jsonify

# This imports the functions you already have and tested in
# gemini_question_extractor.py - nothing in that file needs to change.
from gemini_question_extractor import (
    extract_raw_text,
    split_sections,
    make_batches,
    MCQ_PROMPT,
    TF_PROMPT,
    call_gemini,
    normalize_questions,
    dedupe,
    get_api_key,
)
import time

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract", methods=["POST"])
def extract():
    # Step 1: make sure a file was actually sent
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded", "questions": []})

    uploaded = request.files["file"]

    # Step 2: save it to a temporary spot on disk so the existing
    # extract_raw_text() function (which expects a file path) can read it
    suffix = os.path.splitext(uploaded.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        uploaded.save(tmp.name)
        temp_path = tmp.name

    try:
        api_key = get_api_key()
        if not api_key:
            return jsonify({"success": False, "error": "GEMINI_API_KEY not configured", "questions": []})

        raw_text = extract_raw_text(temp_path)
        if not raw_text or not raw_text.strip():
            return jsonify({"success": False, "error": "No extractable text", "questions": []})

        sections = split_sections(raw_text)
        mcq_batches = make_batches(sections, "mcq")
        tf_batches = make_batches(sections, "true_false")

        all_questions = []
        for i, batch_text in enumerate(mcq_batches):
            try:
                if i > 0:
                    time.sleep(3)
                prompt = MCQ_PROMPT.replace("{raw_text}", batch_text)
                parsed = call_gemini(api_key, prompt)
                all_questions.extend(normalize_questions(parsed))
            except Exception as e:
                print(f"[app] MCQ batch {i} failed, skipping: {e}")

        for i, batch_text in enumerate(tf_batches):
            try:
                if i > 0 or mcq_batches:
                    time.sleep(3)
                prompt = TF_PROMPT.replace("{raw_text}", batch_text)
                parsed = call_gemini(api_key, prompt)
                all_questions.extend(normalize_questions(parsed))
            except Exception as e:
                print(f"[app] T/F batch {i} failed, skipping: {e}")

        all_questions = dedupe(all_questions)

        if not all_questions:
            return jsonify({"success": False, "error": "No usable questions found", "questions": []})

        return jsonify({"success": True, "questions": all_questions, "source": "gemini"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "questions": []})

    finally:
        # Step 3: clean up the temporary file
        try:
            os.remove(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    # This line only matters when testing on your own computer.
    # In production (Render), gunicorn starts this instead - see Step 2 later.
    app.run(debug=True, port=5000)
