"""
app.py — Flask wrapper around gemini_question_extractor.py, ai_chat.py,
ai_lesson_plan.py, and ai_practice.py

This file does NOT change your existing extraction logic at all.

Web addresses (routes) this creates:
  GET  /health              -> health check
  POST /extract               -> question bank extraction
  POST /ai/chat                -> AI Assistant general chat
  POST /ai/lesson-plan           -> AI Assistant lesson plan generator (teacher)
  POST /ai/practice-questions     -> AI Assistant practice questions (student)
  POST /ai/quiz-generator          -> AI Assistant quiz generator (student)
"""
import os
import tempfile
from flask import Flask, request, jsonify

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

from ai_chat import handle_chat_request, ChatValidationError
from ai_lesson_plan import handle_lesson_plan_request, LessonPlanValidationError
from ai_practice import (
    handle_practice_questions_request,
    handle_quiz_generator_request,
    PracticeValidationError,
)

app = Flask(__name__)

AI_SERVICE_SECRET = os.environ.get("AI_SERVICE_SECRET")


def _check_service_secret():
    if not AI_SERVICE_SECRET:
        return True
    provided = request.headers.get("X-Service-Secret")
    return provided == AI_SERVICE_SECRET


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded", "questions": []})

    uploaded = request.files["file"]
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
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.route("/ai/chat", methods=["POST"])
def ai_chat():
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


@app.route("/ai/practice-questions", methods=["POST"])
def ai_practice_questions():
    if not _check_service_secret():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid or missing JSON body"}), 400

    try:
        result = handle_practice_questions_request(body)
    except PracticeValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


@app.route("/ai/quiz-generator", methods=["POST"])
def ai_quiz_generator():
    if not _check_service_secret():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid or missing JSON body"}), 400

    try:
        result = handle_quiz_generator_request(body)
    except PracticeValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(debug=True, port=5000)