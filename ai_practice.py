"""
ai_practice.py
EduQuiz AI Assistant — Student tools: Practice Question Generator and
Quiz Generator.

Both tools share the same input shape (topic, difficulty, question count,
optional instructions) and both return structured JSON via Gemini's
response_schema support - this keeps output parseable and renderable as
an interactive UI rather than free text.

Practice Questions -> open-ended questions with a model answer and a short
explanation, for self-study review.

Quiz Generator -> multiple-choice questions with 4 options and the correct
option index, for self-testing with instant feedback.
"""
import sys
import json
from gemini_client import get_api_key, call_gemini_raw

MIN_QUESTION_COUNT = 1
MAX_QUESTION_COUNT = 20
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = ["topic", "difficulty", "question_count"]


class PracticeValidationError(Exception):
    """Raised when incoming form data fails validation. Callers (app.py)
    should catch this and return a 400 response."""
    pass


PRACTICE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "model_answer": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                },
                "required": ["question", "model_answer"],
            },
        }
    },
    "required": ["questions"],
}

QUIZ_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "correct_index": {"type": "INTEGER"},
                    "explanation": {"type": "STRING"},
                },
                "required": ["question", "options", "correct_index"],
            },
        }
    },
    "required": ["questions"],
}


def _validate(form_data: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if not str(form_data.get(f, "")).strip()]
    if missing:
        raise PracticeValidationError(f"Missing required field(s): {', '.join(missing)}")

    topic = str(form_data["topic"]).strip()
    if len(topic) > 255:
        raise PracticeValidationError("Topic must be 255 characters or fewer.")

    difficulty = str(form_data["difficulty"]).strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        raise PracticeValidationError("Difficulty must be one of: easy, medium, hard.")

    try:
        count = int(form_data["question_count"])
    except (ValueError, TypeError):
        raise PracticeValidationError("question_count must be an integer.")
    if count < MIN_QUESTION_COUNT or count > MAX_QUESTION_COUNT:
        raise PracticeValidationError(
            f"question_count must be between {MIN_QUESTION_COUNT} and {MAX_QUESTION_COUNT}."
        )

    instructions = str(form_data.get("additional_instructions", "") or "").strip()
    if len(instructions) > 1000:
        raise PracticeValidationError("Additional instructions must be 1000 characters or fewer.")

    return {
        "topic": topic,
        "difficulty": difficulty,
        "question_count": count,
        "additional_instructions": instructions,
    }


def _build_practice_prompt(data: dict) -> str:
    instructions_line = data["additional_instructions"] or "None"
    return f"""You are an experienced tutor creating self-study practice questions for a student.

Topic: {data['topic']}
Difficulty: {data['difficulty']}
Number of questions: {data['question_count']}
Additional instructions: {instructions_line}

Generate exactly {data['question_count']} open-ended practice questions on this topic at the specified difficulty. For each question, provide a clear, correct model answer and a short explanation that helps the student understand the reasoning, not just the answer.

Return ONLY valid JSON matching the required schema. Do not include markdown fences or commentary. Write in clear, encouraging, student-friendly English.
"""


def _build_quiz_prompt(data: dict) -> str:
    instructions_line = data["additional_instructions"] or "None"
    return f"""You are an experienced tutor creating a multiple-choice self-test quiz for a student.

Topic: {data['topic']}
Difficulty: {data['difficulty']}
Number of questions: {data['question_count']}
Additional instructions: {instructions_line}

Generate exactly {data['question_count']} multiple-choice questions on this topic at the specified difficulty. Each question must have exactly 4 options, with correct_index being the zero-based index of the correct option. Include a short explanation of why the correct answer is correct.

Return ONLY valid JSON matching the required schema. Do not include markdown fences or commentary.
"""


def _call_and_parse(prompt: str, schema: dict, log_prefix: str) -> list:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    text_out = call_gemini_raw(
        api_key,
        prompt,
        generation_config={
            "temperature": 0.6,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "maxOutputTokens": 4096,
        },
    )

    try:
        parsed = json.loads(text_out)
    except json.JSONDecodeError as e:
        print(f"[{log_prefix}] failed to parse Gemini JSON output: {e}", file=sys.stderr)
        raise RuntimeError("The AI service returned an invalid response. Please try again.")

    questions = parsed.get("questions", [])
    if not questions:
        raise RuntimeError("The AI service did not return any questions. Please try again.")

    return questions


def handle_practice_questions_request(form_data: dict) -> dict:
    data = _validate(form_data)
    try:
        questions = _call_and_parse(_build_practice_prompt(data), PRACTICE_SCHEMA, "ai_practice:practice")
        return {"success": True, "questions": questions}
    except Exception as e:
        print(f"[ai_practice] practice questions generation failed: {e}", file=sys.stderr)
        return {"success": False, "error": "AI service temporarily unavailable"}


def handle_quiz_generator_request(form_data: dict) -> dict:
    data = _validate(form_data)
    try:
        questions = _call_and_parse(_build_quiz_prompt(data), QUIZ_SCHEMA, "ai_practice:quiz")

        # Guard against malformed items before they reach the student UI.
        clean = []
        for q in questions:
            opts = q.get("options") or []
            if len(opts) < 2:
                continue
            correct_index = q.get("correct_index")
            if not isinstance(correct_index, int) or correct_index < 0 or correct_index >= len(opts):
                correct_index = None
            clean.append({
                "question": (q.get("question") or "").strip(),
                "options": [str(o).strip() for o in opts],
                "correct_index": correct_index,
                "explanation": (q.get("explanation") or "").strip(),
            })

        if not clean:
            return {"success": False, "error": "The AI service did not return any usable questions. Please try again."}

        return {"success": True, "questions": clean}
    except Exception as e:
        print(f"[ai_practice] quiz generation failed: {e}", file=sys.stderr)
        return {"success": False, "error": "AI service temporarily unavailable"}