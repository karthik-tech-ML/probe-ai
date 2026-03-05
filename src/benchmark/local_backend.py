"""
Ollama backend for local inference benchmarking.

Talks to a local Ollama instance over HTTP, using the same system prompt
and context format as our Claude generator. That way we're comparing
apples to apples — same instructions, same context, different model.
"""

import time
from dataclasses import dataclass

import httpx
from loguru import logger


# same system prompt as src/rag/generator.py — we want the local model
# under identical constraints so the comparison is fair
_SYSTEM_PROMPT = """\
You are a movie knowledge assistant. Your ONLY source of information is the \
context provided below. Follow these rules strictly:

1. ONLY use facts that appear in the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
say "I don't have enough information to answer that based on the available data."
3. When citing facts (director, cast, budget, etc.), stick to exactly what the \
context says. Do not embellish or infer beyond what is written.
4. If the question asks about a movie or topic not covered in the context, say so.
5. Keep answers concise and direct. No filler.
"""

# Ollama API defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


@dataclass
class LocalResult:
    answer: str
    model: str
    latency_ms: float
    # ollama gives us token counts in the response
    prompt_eval_count: int  # input tokens processed
    eval_count: int  # output tokens generated
    eval_duration_ns: int  # time spent generating (nanoseconds)
    total_duration_ns: int  # total request time (nanoseconds)


def _build_prompt(question: str, context_texts: list[str]) -> str:
    """
    Format question + context the same way as the Claude generator.
    Numbered [Source N] blocks so the model sees the same layout.
    """
    sections = []
    for i, text in enumerate(context_texts, 1):
        sections.append(f"[Source {i}]\n{text}")
    context_block = "\n\n".join(sections)

    return f"""Context:
{context_block}

Question: {question}"""


def check_ollama(base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Quick health check — is Ollama running and reachable?"""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def list_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """Return names of models currently pulled in Ollama."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning(f"couldn't list ollama models: {e}")
        return []


def generate(
    question: str,
    context_texts: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 120.0,
) -> LocalResult:
    """
    Send a question + context to Ollama and get a grounded answer.

    Uses the /api/generate endpoint with stream=false so we get the
    full response in one shot (plus timing stats from Ollama itself).
    """
    prompt = _build_prompt(question, context_texts)

    logger.debug(f"calling ollama ({model}) with {len(context_texts)} context chunks")

    t0 = time.perf_counter()

    resp = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            # keep output focused — we don't want rambling from small models
            "options": {
                "num_predict": 512,
                "temperature": 0.1,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    latency_ms = (time.perf_counter() - t0) * 1000

    answer = data.get("response", "").strip()
    if not answer:
        answer = "(empty response from model)"

    result = LocalResult(
        answer=answer,
        model=data.get("model", model),
        latency_ms=round(latency_ms, 1),
        prompt_eval_count=data.get("prompt_eval_count", 0),
        eval_count=data.get("eval_count", 0),
        eval_duration_ns=data.get("eval_duration_ns", 0),
        total_duration_ns=data.get("total_duration_ns", 0),
    )

    logger.info(
        f"ollama responded in {latency_ms:.0f}ms "
        f"({result.prompt_eval_count}in/{result.eval_count}out tokens)"
    )

    return result
