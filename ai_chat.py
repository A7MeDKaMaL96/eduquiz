"""
ai_chat.py
EduQuiz AI Assistant — General Chat feature.

Provides a stateless, role-aware conversational endpoint on top of Gemini.
The PHP layer (includes/ai/AiClient.php) owns session/history persistence;
this file's only job is: given a role, an optional conversation history,
and a new user message, return the assistant's reply text.

Kept deliberately separate from gemini_question_extractor.py's prompts and
schema - chat has no structured output requirement, so it uses
call_gemini_raw() with plain text generation, no response_schema.
"""
import sys
from gemini_client import get_api_key, call_gemini_raw

MAX_HISTORY_MESSAGES = 20  # caps prompt size / cost per request
MAX_MESSAGE_CHARS = 6000   # guards against pathological single-message input

ROLE_SYSTEM_PROMPTS = {
    "teacher": (
        "You are the built-in AI Assistant inside EduQuiz, a school quiz and "
        "learning management platform. You are speaking with a TEACHER. "
        "You can help with: programming, networking, teaching ideas, lesson "
        "explanations, assessment ideas, classroom activities, quiz design, "
        "coding help, educational questions, and general professional advice "
        "for educators. Be clear, practical, and professional. Use concise "
        "formatting (short paragraphs, bullet points where useful). Do not "
        "claim to take actions inside the platform - you can only provide "
        "information and suggestions."
    ),
    "student": (
        "You are the built-in AI Assistant inside EduQuiz, a school quiz and "
        "learning management platform. You are speaking with a STUDENT. "
        "You can help with: explaining difficult topics, summarizing lessons, "
        "answering academic questions, study techniques, and generating "
        "practice questions. Be encouraging, clear, and age-appropriate. "
        "Break down complex ideas into simple steps. Do not do the student's "
        "graded work for them outright - guide their understanding instead."
    ),
    "admin": (
        "You are the built-in AI Assistant inside EduQuiz, a school quiz and "
        "learning management platform. You are speaking with an ADMINISTRATOR. "
        "You can help with: drafting announcements, emails, reports, course "
        "descriptions, policy drafts, and other administrative documents. Be "
        "professional and concise."
    ),
}


class ChatValidationError(Exception):
    """Raised when incoming chat input fails validation. Callers (app.py)
    should catch this and return a 400 response."""
    pass


def _validate_role(role: str) -> str:
    if role not in ROLE_SYSTEM_PROMPTS:
        raise ChatValidationError(f"Unsupported role: {role}")
    return role


def _validate_message(message: str) -> str:
    if not message or not message.strip():
        raise ChatValidationError("Message cannot be empty")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ChatValidationError(f"Message exceeds maximum length of {MAX_MESSAGE_CHARS} characters")
    return message.strip()


def _build_prompt(role: str, history: list, message: str) -> str:
    """
    Builds a single text prompt combining the role's system instructions,
    a trimmed conversation history, and the new user message.

    history: list of {"sender": "user"|"assistant", "content": str} dicts,
    ordered oldest -> newest. Only the most recent MAX_HISTORY_MESSAGES are
    included to bound prompt size.
    """
    system_prompt = ROLE_SYSTEM_PROMPTS[role]

    trimmed_history = history[-MAX_HISTORY_MESSAGES:] if history else []

    conversation_lines = []
    for turn in trimmed_history:
        sender = turn.get("sender")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "Teacher/Student/Admin" if sender == "user" else "Assistant"
        conversation_lines.append(f"{label}: {content}")

    conversation_lines.append(f"User: {message}")
    conversation_text = "\n".join(conversation_lines)

    return (
        f"{system_prompt}\n\n"
        f"Below is the conversation so far. Respond only as the Assistant, "
        f"continuing naturally from the last message. Do not repeat earlier "
        f"turns in your reply.\n\n"
        f"---\n{conversation_text}\n---\n\n"
        f"Assistant:"
    )


def handle_chat_request(role: str, message: str, history: list = None) -> dict:
    """
    Main entry point called by the /ai/chat route in app.py.

    Args:
        role: "teacher" | "student" | "admin"
        message: the new user message (raw, unvalidated)
        history: optional list of prior {"sender", "content"} turns

    Returns:
        {"success": True, "reply": str} on success
        {"success": False, "error": str} on failure (never raises upward
        except ChatValidationError, which app.py maps to HTTP 400)
    """
    role = _validate_role(role)
    message = _validate_message(message)
    history = history or []

    api_key = get_api_key()
    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    prompt = _build_prompt(role, history, message)

    try:
        reply_text = call_gemini_raw(
            api_key,
            prompt,
            generation_config={
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            },
        )
        return {"success": True, "reply": reply_text.strip()}
    except Exception as e:
        print(f"[ai_chat] Gemini call failed: {e}", file=sys.stderr)
        return {"success": False, "error": "AI service temporarily unavailable"}