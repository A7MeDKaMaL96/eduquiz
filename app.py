"""
app.py — Flask wrapper around gemini_question_extractor.py, ai_chat.py, and
ai_lesson_plan.py

This file does NOT change your existing extraction logic at all. It just
gives that logic a "front door" that can answer web requests, instead of
only working when you type a command in a terminal.

Web addresses (routes) this creates:
  GET  /health         -> just says "I'm alive", used to check the service is up
  POST /extract         -> send it a file, get back questions
  POST /ai/chat          -> AI Assistant general chat (teacher/student/admin)
  POST /ai/lesson-plan     -> AI Assistant lesson plan generator (teacher)

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

# AI Assistant feature modules (new)
from ai_chat import handle_chat_request, ChatValidationError
from ai_lesson_plan import handle_lesson_plan_request, LessonPlanValidationError

app = Flask(__name__)

# Shared secret used to verify requests are coming from the PHP LMS backend,
# not directly from a browser. Set AI_SERVICE_SECRET in services/.env and
# in the LMS's config.php - PHP sends it as the X-Service-Secret header on
# every /ai/* request. This does not replace the Gemini API key protection;
# it prevents the /ai/* endpoints from being called by anyone who merely
# discovers this service's URL.
AI_SERVICE_SECRET = os.environ.get("AI_SERVICE_SECRET")


def _check_service_secret():
    """Returns True if the request carries the correct shared secret, or if
    no secret is configured (local dev). In production, always set
    AI_SERVICE_SECRET so this check is enforced."""
    if not AI_SERVICE_SECRET:
        return True
    provided = request.headers.get("X-Service-Secret")
    return provided == AI_SERVICE_SECRET


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


@app.route("/ai/chat", methods=["POST"])
def ai_chat():
    """
    AI Assistant - General Chat.

    Expects JSON body:
      {
        "role": "teacher" | "student" | "admin",
        "message": "string",
        "history": [ {"sender": "user"|"assistant", "content": "string"}, ... ]  (optional)
      }

    Returns:
      200 {"success": true, "reply": "string"}
      400 {"success": false, "error": "validation message"}
      401 {"success": false, "error": "Unauthorized"}
      500 {"success": false, "error": "AI service temporarily unavailable"}
    """
    if not _check_service_secret():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid or missing JSON body"}), 400

    try:
        result = handle_chat_request(
            role=body.get("role", ""),
            message=body.get("message", ""),
            history=body.get("history") or [],
        )
    except ChatValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


@app.route("/ai/lesson-plan", methods=["POST"])
def ai_lesson_plan():
    """
    AI Assistant - Lesson Plan Generator (Teacher only; role enforcement
    also happens on the PHP side before this is ever called).

    Expects JSON body: the full structured form data (see
    ai_lesson_plan.REQUIRED_FIELDS / OPTIONAL_FIELDS for the exact keys).

    Returns:
      200 {"success": true, "content": "string"}
      400 {"success": false, "error": "validation message"}
      401 {"success": false, "error": "Unauthorized"}
      500 {"success": false, "error": "AI service temporarily unavailable"}
    """
    if not _check_service_secret():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid or missing JSON body"}), 400

    try:
        result = handle_lesson_plan_request(body)
    except LessonPlanValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


if __name__ == "__main__":
    # This line only matters when testing on your own computer.
    # In production (Render), gunicorn starts this instead - see Step 2 later.
    app.run(debug=True, port=5000)