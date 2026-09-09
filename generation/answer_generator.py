import logging
import time
from typing import Optional
import google.generativeai as genai

from generation.config import (
    GEMINI_API_KEY,
    GENERATION_LOG_FILE,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
)

logger = logging.getLogger("noticerag.generation.answer_generator")


class GeminiAnswerGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Generation will fail or return fallback notice.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            logger.info("Initialized Gemini client with model %s", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize Gemini client: %s", exc)
            self.model = None

    def generate_answer(self, prompt: str, max_retries: int = 3, timeout_seconds: float = 30.0) -> str:
        if not self.api_key:
            return "Error: GEMINI_API_KEY environment variable is not configured. Please set your GEMINI_API_KEY."

        if not self.model:
            self._init_client()
            if not self.model:
                return "Error: Gemini model could not be initialized."

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    request_options={"timeout": timeout_seconds},
                )
                if response and response.text:
                    return response.text.strip()
                return "No answer could be generated from the provided context."
            except Exception as exc:
                last_error = exc
                logger.warning("Gemini API attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)

        logger.error("All Gemini API attempts exhausted. Error: %s", last_error)
        return f"Error communicating with Gemini API: {last_error}"


_default_generator: Optional[GeminiAnswerGenerator] = None


def generate_answer(prompt: str) -> str:
    global _default_generator
    if _default_generator is None:
        _default_generator = GeminiAnswerGenerator()
    return _default_generator.generate_answer(prompt)
