import hashlib
import json
import os
from pathlib import Path
import re
import time

from config import constants


class LanguageModelError(Exception):
    """The parent of everything that can go wrong with the model."""
    pass


class ModelNotConfigured(LanguageModelError):
    """Raised where no key has been configured."""
    pass


def _read_env_file():
    env_path = Path(".env")
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k not in os.environ:
                    os.environ[k] = v


def load_settings():
    _read_env_file()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ModelNotConfigured(
            "GEMINI_API_KEY is not set; put it in .env before asking the model anything"
        )
    model_name = (
        os.environ.get("GEMINI_MODEL", "").strip() or constants.DEFAULT_LANGUAGE_MODEL
    )
    return {"api_key": api_key, "model_name": model_name}


def cache_key(model_name, prompt):
    content = f"{model_name}:{prompt}".encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def read_cache(cache_path=constants.LANGUAGE_MODEL_CACHE_PATH):
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise LanguageModelError(f"Could not read cache file at {cache_path}: {e}")


def write_cache(cache, cache_path=constants.LANGUAGE_MODEL_CACHE_PATH):
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return p


def open_model_client(api_key):
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai
        except ImportError:
            raise LanguageModelError("Neither google.genai nor google.generativeai is installed.")


def ask_model(prompt, cache_path=constants.LANGUAGE_MODEL_CACHE_PATH):
    model_name = (
        os.environ.get("GEMINI_MODEL", "").strip() or constants.DEFAULT_LANGUAGE_MODEL
    )
    key = cache_key(model_name, prompt)
    cache = read_cache(cache_path)
    if key in cache:
        return cache[key]

    settings = load_settings()
    client = open_model_client(settings["api_key"])
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=settings["model_name"], contents=prompt
            )
            answer = resp.text
        else:
            model = client.GenerativeModel(settings["model_name"])
            resp = model.generate_content(prompt)
            answer = resp.text
        if not answer or not answer.strip():
            raise LanguageModelError("The model answered with nothing at all.")
        cache[key] = answer
        write_cache(cache, cache_path)
        return answer
    except Exception as e:
        if isinstance(e, LanguageModelError):
            raise
        raise LanguageModelError(f"Language model request failed: {e}")


def ask_model_with_retry(
    prompt,
    cache_path=constants.LANGUAGE_MODEL_CACHE_PATH,
    attempts=constants.MODEL_RETRY_ATTEMPTS,
    pause_seconds=constants.MODEL_RETRY_SECONDS,
    waits=None,
):
    current_pause = pause_seconds
    for attempt in range(1, attempts + 1):
        try:
            return ask_model(prompt, cache_path)
        except ModelNotConfigured:
            raise
        except LanguageModelError as e:
            if attempt == attempts:
                raise
            if waits is not None:
                waits.append(current_pause)
            else:
                time.sleep(current_pause)
            current_pause *= 2.0


def read_answer(said, wanted_keys):
    text = said.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)

    try:
        data = json.loads(text)
    except Exception as e:
        raise LanguageModelError(f"Could not parse JSON from answer: {said[:80]}... Reason: {e}")

    if not isinstance(data, dict):
        raise LanguageModelError(f"Answer did not contain a JSON object: {said[:80]}")

    missing = [k for k in wanted_keys if k not in data]
    if missing:
        raise LanguageModelError(
            f"Answer missing required key(s) {missing}: {said[:80]}"
        )
    return data


if __name__ == "__main__":
    cache = read_cache()
    print(f"cache holds: {len(cache)} answers")
