"""
gemini_client.py
Shared low-level Gemini API client, used by every AI feature in this
project (question extraction, AI Assistant chat, lesson plan generator,
and any future tool).

This file owns exactly two responsibilities:
  1. Resolving the Gemini API key from the environment / .env file.
  2. Making a single Gemini generateContent call, with retry/backoff on
     rate limits and network errors, and returning the raw text response.

It does NOT parse JSON, does NOT know about question schemas, and does
NOT know about lesson plan formatting. Callers own that logic. This keeps
the client reusable across every current and future AI tool without this
file ever needing to change again.

The API key never leaves this process boundary - callers only ever see
the generated text, never the key itself.
"""
import os
import sys
import time

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = 60
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "maxOutputTokens": 8192,
}


def load_env_file():
    """Loads services/.env into os.environ if not already set. Safe to call
    multiple times - existing environment variables are never overwritten."""
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
        print(f"[gemini_client] failed to read .env: {e}", file=sys.stderr)


def get_api_key():
    """Returns the configured Gemini API key, or None if not set. Never
    logs or prints the key itself."""
    load_env_file()
    return os.environ.get("GEMINI_API_KEY")


def call_gemini_raw(api_key: str, prompt_text: str, generation_config: dict = None,
                     max_retries: int = 4, timeout_seconds: int = GEMINI_TIMEOUT_SECONDS) -> str:
    """
    Makes a single Gemini generateContent request and returns the raw text
    of the model's response (data.candidates[0].content.parts[0].text).

    Args:
        api_key: Gemini API key (from get_api_key()).
        prompt_text: The full prompt to send.
        generation_config: Optional dict merged over DEFAULT_GENERATION_CONFIG.
            Pass response_mime_type / response_schema here for structured
            JSON output (see gemini_question_extractor.py for an example).
            Omit for free-form text output (chat, lesson plans).
        max_retries: Number of attempts before giving up on 429s / network errors.
        timeout_seconds: Per-request HTTP timeout.

    Raises:
        The last encountered exception if all retries are exhausted, or if
        the response shape is unexpected. Callers are responsible for
        catching this and deciding on a fallback (never silently swallowed
        here, since different tools need different failure behavior).
    """
    import requests

    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    config = dict(DEFAULT_GENERATION_CONFIG)
    if generation_config:
        config.update(generation_config)

    body = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": config,
    }

    delay = 5
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GEMINI_API_URL,
                params={"key": api_key},
                json=body,
                timeout=timeout_seconds,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"[gemini_client] 429 rate limited, retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                delay *= 2
                continue

            resp.raise_for_status()
            data = resp.json()

            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as shape_err:
                raise RuntimeError(
                    f"Unexpected Gemini response shape: {shape_err}"
                ) from shape_err

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < max_retries - 1:
                print(f"[gemini_client] network error ({e.__class__.__name__}), retrying in {delay}s "
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