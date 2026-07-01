"""
ai_lesson_plan.py
EduQuiz AI Assistant — Lesson Plan Generator (Teacher tool).

Takes the structured multi-step form data collected by the PHP frontend
and builds the hidden Gemini prompt server-side (the teacher never sees
the prompt, only the final generated plan - per product spec). Returns
the fully formatted lesson plan as plain text.

All formatting rules (first-person voice, no emojis, section structure,
activity timing matching total session duration) are encoded in the
prompt template below, not left to free interpretation by the model.
"""
import sys
from gemini_client import get_api_key, call_gemini_raw

REQUIRED_FIELDS = [
    "course_name",
    "lesson_title",
    "academic_year",
    "semester",
    "student_level",
    "department",
    "topic",
    "session_duration_minutes",
    "delivery_mode",
    "practical_or_theoretical",
    "learning_objectives",
    "learning_outcomes",
    "teaching_strategy",
    "teaching_method",
]

OPTIONAL_FIELDS = [
    "class_size",
    "available_resources",
    "special_notes",
    "additional_instructions",
]


class LessonPlanValidationError(Exception):
    """Raised when required form fields are missing or malformed. Callers
    (app.py) should catch this and return a 400 response."""
    pass


def _validate_form_data(form_data: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if not str(form_data.get(f, "")).strip()]
    if missing:
        raise LessonPlanValidationError(f"Missing required field(s): {', '.join(missing)}")

    try:
        duration = int(form_data["session_duration_minutes"])
        if duration <= 0 or duration > 480:
            raise ValueError
    except (ValueError, TypeError):
        raise LessonPlanValidationError("session_duration_minutes must be a positive integer (max 480)")

    cleaned = {}
    for field in REQUIRED_FIELDS:
        cleaned[field] = str(form_data[field]).strip()
    cleaned["session_duration_minutes"] = duration

    for field in OPTIONAL_FIELDS:
        value = form_data.get(field)
        cleaned[field] = str(value).strip() if value else ""

    return cleaned


def _build_prompt(data: dict) -> str:
    resources_line = data["available_resources"] or "Standard classroom resources"
    class_size_line = data["class_size"] or "Not specified"
    notes_line = data["special_notes"] or "None"
    instructions_line = data["additional_instructions"] or "None"

    return f"""You are an experienced university-level instructor writing a formal lesson plan for internal academic use. Write entirely in English, in first-person voice throughout (e.g. "I will explain...", "I will ask students to...", "I will evaluate..."). Do not use emojis or icons anywhere. Use clean, professional academic formatting with clear section headers.

LESSON CONTEXT:
- Course Name: {data['course_name']}
- Lesson Title: {data['lesson_title']}
- Academic Year: {data['academic_year']}
- Semester: {data['semester']}
- Student Level: {data['student_level']}
- Department: {data['department']}
- Topic: {data['topic']}
- Total Session Duration: {data['session_duration_minutes']} minutes
- Class Size: {class_size_line}
- Delivery Mode: {data['delivery_mode']}
- Format: {data['practical_or_theoretical']}
- Teaching Strategy: {data['teaching_strategy']}
- Teaching Method: {data['teaching_method']}
- Available Resources: {resources_line}
- Special Notes: {notes_line}
- Additional Instructions: {instructions_line}

RAW LEARNING OBJECTIVES (provided by the teacher, expand and refine these):
{data['learning_objectives']}

RAW LEARNING OUTCOMES (provided by the teacher, expand and refine these):
{data['learning_outcomes']}

Produce the lesson plan with EXACTLY these sections, in this order:

1. Lesson Information
   A short block restating course, lesson title, year, semester, level, department, topic, duration, class size, delivery mode, and format.

2. Learning Objectives
   A refined, professionally worded list expanding on the raw objectives above.

3. Teaching Strategy
   Describe the overall strategy I will use, written in first person.

4. Strategy Objective
   Explain in first person why this strategy fits this lesson and student level.

5. Learning Outcomes
   A refined, professionally worded list expanding on the raw outcomes above.

6. Activity Objective
   Explain in first person what the classroom activities are designed to achieve.

7. Classroom Activities
   Break the ENTIRE {data['session_duration_minutes']}-minute session into a sequence of activities. The sum of all activity durations MUST equal exactly {data['session_duration_minutes']} minutes - no more, no less. For EVERY activity include, as sub-fields:
   - Step-by-step procedure
   - Estimated duration (minutes)
   - Teacher actions (first person)
   - Student actions
   - Assessment method
   - Materials
   - Expected outcome

At the end, include these final sections:
- Assessment
- Reflection
- Homework
- Summary

Formatting rules:
- No emojis, no icons, no decorative symbols.
- First-person voice for all instructor actions throughout.
- Professional academic tone suitable for university documentation and accreditation review.
- Use numbered/lettered headers matching the section list above.
- Do not include any text about these instructions themselves - output only the finished lesson plan.
"""


def handle_lesson_plan_request(form_data: dict) -> dict:
    """
    Main entry point called by the /ai/lesson-plan route in app.py.

    Args:
        form_data: dict of raw form fields from the PHP multi-step wizard.

    Returns:
        {"success": True, "content": str} on success
        {"success": False, "error": str} on failure (never raises upward
        except LessonPlanValidationError, which app.py maps to HTTP 400)
    """
    data = _validate_form_data(form_data)

    api_key = get_api_key()
    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    prompt = _build_prompt(data)

    try:
        content = call_gemini_raw(
            api_key,
            prompt,
            generation_config={
                "temperature": 0.5,
                "maxOutputTokens": 8192,
            },
        )
        return {"success": True, "content": content.strip()}
    except Exception as e:
        print(f"[ai_lesson_plan] Gemini call failed: {e}", file=sys.stderr)
        return {"success": False, "error": "AI service temporarily unavailable"}