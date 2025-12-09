from typing import Optional, Dict, Any
from .langfuse_client import langfuse


def trace_llm_call(
    name: str,
    prompt: str,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
    response_text: Optional[str] = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """
    Логируем LLM-вызов в Langfuse через новый Python SDK v3.

    Создаём observation типа "generation", в input кладём prompt,
    в output — ответ/ошибку, в metadata — метаинфу.
    """

    try:
        # создаём "generation"-обсервацию
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input={"prompt": prompt},
        ) as gen:
            gen.update(
                output=response_text or "",
                metadata={
                    "error": error,
                    "duration_ms": duration_ms,
                    **(metadata or {}),
                },
            )
    except Exception:
        # мониторинг не должен ломать бизнес-логику
        pass
