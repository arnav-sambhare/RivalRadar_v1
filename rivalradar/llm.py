"""Groq chat calls shared by the relevance pass and discovery, within rate limits.

The API key comes from ``GROQ_API_KEY`` (env or ``.env``) and the model from
``llm.model`` in config. Callers handle failure themselves (all LLM use in
Rival Radar is optional and falls back).

Groq limits each key per minute (tokens in, and separately tokens out) and
per day (requests). The input and request budgets are reported in
``x-ratelimit-*`` response headers; the output budget is not, and a request
whose possible reply exceeds it is refused outright. Every call here goes
through one ``RateBudget`` that:
  * spaces calls at least ``llm.min_interval`` seconds apart (default 2);
  * estimates a request's tokens and, when the minute's remaining token budget
    is too small, waits for it to reset rather than being refused;
  * caps each reply at ``llm.max_output_tokens`` and keeps the replies of the
    last minute within ``llm.output_tokens_per_minute``, waiting if needed;
  * stops calling once the day's request allowance is spent;
and the Groq client itself retries 429s, honouring ``Retry-After``.
Calls are made at temperature 0 so a run's inferences are repeatable.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque

from .config import Config

log = logging.getLogger("rivalradar.llm")

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_INTERVAL = 2.0
CHARS_PER_TOKEN = 3  # conservative: real text runs ~4 chars/token
DEFAULT_MAX_OUTPUT = 600  # cap on each reply's tokens
DEFAULT_OUTPUT_PER_MINUTE = 1000  # Groq on-demand output-token limit for many models
_DURATION_PART = re.compile(r"([\d.]+)(ms|h|m|s)")


class LLMUnavailable(RuntimeError):
    """The LLM can't be called now (quota spent); callers fall back."""


def estimate_tokens(*texts: str) -> int:
    return sum(len(t) for t in texts) // CHARS_PER_TOKEN + 1


def parse_reset(value: str | None) -> float:
    """Groq reset durations like "1m26.4s", "1.17s", "250ms" -> seconds."""
    seconds = 0.0
    for number, unit in _DURATION_PART.findall(value or ""):
        seconds += float(number) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return seconds


class RateBudget:
    """What's left of the key's Groq allowance, from the latest headers."""

    def __init__(self) -> None:
        self.remaining_tokens: int | None = None
        self.tokens_reset_at = 0.0
        self.remaining_requests: int | None = None
        self.requests_reset_at = 0.0
        self.last_call = 0.0
        self.outputs: deque[tuple[float, int]] = deque()  # (when, output tokens) in the last minute

    def _output_wait(self, now: float, max_output: int, per_minute: int) -> float:
        """Seconds until the last minute's replies leave room for ``max_output`` more."""
        while self.outputs and now - self.outputs[0][0] >= 60:
            self.outputs.popleft()
        used = sum(n for _, n in self.outputs)
        wait = 0.0
        for when, n in self.outputs:
            if used + max_output <= per_minute:
                break
            used -= n
            wait = when + 60 - now
        return max(wait, 0.0)

    def record_output(self, tokens: int) -> None:
        self.outputs.append((time.monotonic(), tokens))

    def wait(self, needed_tokens: int, interval: float,
             max_output: int = 0, output_per_minute: int | None = None) -> None:
        """Sleep until a call fits the budgets; raise if the day's quota is gone."""
        now = time.monotonic()
        if self.remaining_requests == 0 and now < self.requests_reset_at:
            raise LLMUnavailable(
                f"Groq daily request quota used up (resets in {self.requests_reset_at - now:.0f}s)")
        delay = self.last_call + interval - now
        if (self.remaining_tokens is not None and self.remaining_tokens < needed_tokens
                and now < self.tokens_reset_at):
            delay = max(delay, self.tokens_reset_at - now + 0.5)
            log.info("llm: waiting %.1fs for the Groq token budget to reset", delay)
        if output_per_minute:
            out_wait = self._output_wait(now, max_output, output_per_minute)
            if out_wait > 0 and out_wait > delay:
                delay = out_wait + 0.5
                log.info("llm: waiting %.1fs for the Groq output-token budget", delay)
        if delay > 0:
            time.sleep(delay)

    def update(self, headers) -> None:
        now = time.monotonic()
        self.last_call = now

        def as_int(name: str) -> int | None:
            value = headers.get(name)
            try:
                return int(value) if value is not None else None
            except ValueError:
                return None

        tokens = as_int("x-ratelimit-remaining-tokens")
        if tokens is not None:
            self.remaining_tokens = tokens
            self.tokens_reset_at = now + parse_reset(headers.get("x-ratelimit-reset-tokens"))
        requests = as_int("x-ratelimit-remaining-requests")
        if requests is not None:
            self.remaining_requests = requests
            self.requests_reset_at = now + parse_reset(headers.get("x-ratelimit-reset-requests"))


_budget = RateBudget()
_clients: dict = {}


def complete_json(config: Config, system: str, user: str) -> str:
    """One chat completion in JSON mode; returns the raw JSON text."""
    from groq import Groq  # imported lazily so fallbacks work without it

    key = config.secrets.require("groq_api_key")
    client = _clients.get(key)
    if client is None:
        client = _clients[key] = Groq(api_key=key, max_retries=3)  # retries 429 per Retry-After

    llm = config.llm
    max_output = int(llm.get("max_output_tokens", DEFAULT_MAX_OUTPUT))
    _budget.wait(estimate_tokens(system, user) + max_output,
                 float(llm.get("min_interval", DEFAULT_INTERVAL)),
                 max_output, int(llm.get("output_tokens_per_minute", DEFAULT_OUTPUT_PER_MINUTE)))
    raw = client.chat.completions.with_raw_response.create(
        model=llm.get("model", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=max_output,
    )
    _budget.update(raw.headers)
    completion = raw.parse()
    usage = getattr(completion, "usage", None)
    _budget.record_output(int(getattr(usage, "completion_tokens", 0) or max_output))
    return completion.choices[0].message.content or ""
