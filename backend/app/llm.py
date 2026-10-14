"""Model adapter for Gemini / hosted OpenAI-compatible API and local Ollama."""
import asyncio
from typing import Any
import httpx
from openai import AsyncOpenAI
from .config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_GPU,
)
from .schemas import AnswerLanguage
from .prompts import build_prompts


# Backward compatibility alias
def _build_prompts(
    question: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    language: AnswerLanguage | str = AnswerLanguage.EN,
):
    return build_prompts(question, context_chunks, history, language=language)


async def call_gemini(system_prompt: str, user_prompt: str, safe_history: list[dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty in backend/.env. Please configure your Gemini API key.")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *safe_history,
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "The Gemini model returned an empty answer."


async def call_ollama(system_prompt: str, user_prompt: str, safe_history: list[dict]) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        *safe_history,
        {"role": "user", "content": user_prompt}
    ]
    # Options tuned for laptops with 4GB VRAM (GTX 1650) and 8GB System RAM
    options = {
        "temperature": 0.2,
        "num_predict": 1536,  # Allows complete, detailed answers without premature cutoff
        "num_ctx": 3072,     # Context window for chunks + output
        "num_thread": 6      # Utilizes all cores of AMD Ryzen 5 5600H
    }
    if OLLAMA_NUM_GPU >= 0:
        options["num_gpu"] = OLLAMA_NUM_GPU

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "5m",
                    "options": options
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("message", {}).get("content", "Ollama returned an empty answer.")
        except httpx.HTTPStatusError as e:
            # If GPU CUDA_Host memory fails (500), automatically retry on CPU (num_gpu=0)
            if e.response.status_code == 500:
                cpu_options = {**options, "num_gpu": 0}
                retry = await client.post(
                    f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "keep_alive": "5m",
                        "options": cpu_options
                    },
                )
                if retry.status_code == 200:
                    return retry.json().get("message", {}).get("content", "Ollama returned an empty answer.")
            err_detail = e.response.text or str(e)
            raise RuntimeError(f"Ollama server returned {e.response.status_code}: {err_detail}") from e


async def generate_answer(
    question: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    provider: str | None = None,
    answer_language: AnswerLanguage | str = AnswerLanguage.EN,
) -> dict[str, Any]:
    selected_provider = (provider or LLM_PROVIDER or "gemini").lower()
    if selected_provider in {"openai", "gemini"}:
        selected_provider = "gemini"

    resolved_lang = (
        answer_language
        if isinstance(answer_language, AnswerLanguage)
        else AnswerLanguage.from_str(str(answer_language))
    )

    system_prompt, user_prompt, safe_history = build_prompts(
        question, context_chunks, history, language=resolved_lang
    )

    if selected_provider == "gemini":
        ans = await call_gemini(system_prompt, user_prompt, safe_history)
        return {
            "provider": "gemini",
            "answer_language": resolved_lang.value,
            "answer": ans,
            "answers": {"gemini": ans}
        }

    if selected_provider == "ollama":
        try:
            ans = await call_ollama(system_prompt, user_prompt, safe_history)
            return {
                "provider": "ollama",
                "answer_language": resolved_lang.value,
                "answer": ans,
                "answers": {"ollama": ans}
            }
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}. Make sure Ollama is running and model '{OLLAMA_MODEL}' is pulled.") from e

    if selected_provider == "both":
        # Run both Gemini and Ollama concurrently with the exact same language setting
        gemini_task = call_gemini(system_prompt, user_prompt, safe_history)
        ollama_task = call_ollama(system_prompt, user_prompt, safe_history)

        gemini_res, ollama_res = await asyncio.gather(gemini_task, ollama_task, return_exceptions=True)

        gemini_text = str(gemini_res) if not isinstance(gemini_res, Exception) else f"⚠️ Gemini Error: {str(gemini_res)}"
        ollama_text = str(ollama_res) if not isinstance(ollama_res, Exception) else f"⚠️ Ollama Error: {str(ollama_res)}"

        combined_answer = (
            f"**🌟 Gemini ({OPENAI_MODEL}):**\n\n{gemini_text}\n\n"
            f"---\n\n"
            f"**🦙 Ollama ({OLLAMA_MODEL}):**\n\n{ollama_text}"
        )

        return {
            "provider": "both",
            "answer_language": resolved_lang.value,
            "answer": combined_answer,
            "answers": {
                "gemini": gemini_text,
                "ollama": ollama_text
            }
        }

    raise RuntimeError(f"Unknown LLM provider '{provider}'. Choose 'gemini', 'ollama', or 'both'.")
