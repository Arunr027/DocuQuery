
import hashlib
import json

from redis import Redis

from app.core.redis_client import redis_client


CACHE_TTL_SECONDS = 300  # 5 minutes
CACHE_PREFIX = "docuquery:hybrid:"


def make_cache_key(question: str, limit: int) -> str:
    """Create a consistent cache key for a search request."""
    normalized_question = " ".join(question.casefold().split())
    raw_key = f"{normalized_question}|{limit}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    return f"{CACHE_PREFIX}{digest}"


def get_cached_results(
    question: str,
    limit: int,
) -> list[dict] | None:
    """Return cached search results, or None on a cache miss."""
    key = make_cache_key(question, limit)
import hashlib
import json

import redis

from app.core.redis_client import redis_client


CACHE_TTL_SECONDS = 300
CACHE_PREFIX = "docuquery:hybrid:"


def make_cache_key(question: str, limit: int) -> str:
    """Create a consistent cache key for a search request."""
    normalized_question = " ".join(question.casefold().split())
    raw_key = f"{normalized_question}|{limit}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    return f"{CACHE_PREFIX}{digest}"


def get_cached_results(
    question: str,
    limit: int,
) -> list[dict] | None:
    """Return cached results, or None if unavailable."""
    try:
        cached_value = redis_client.get(make_cache_key(question, limit))

        if cached_value is None:
            return None

        return json.loads(cached_value)

    except (redis.RedisError, json.JSONDecodeError) as exc:
        print(f"Redis cache read failed: {exc}")
        return None


def cache_results(
    question: str,
    limit: int,
    results: list[dict],
) -> None:
    """Save search results without failing the main request."""
    try:
        redis_client.setex(
            make_cache_key(question, limit),
            CACHE_TTL_SECONDS,
            json.dumps(results),
        )
    except (redis.RedisError, TypeError, ValueError) as exc:
        print(f"Redis cache write failed: {exc}")


def clear_search_cache() -> int:
    """Delete all hybrid-search cache entries."""
    try:
        keys = list(redis_client.scan_iter(match=f"{CACHE_PREFIX}*"))

        if not keys:
            return 0

        return redis_client.delete(*keys)

    except redis.RedisError as exc:
        print(f"Redis cache clearing failed: {exc}")
        return 0

    cached_value = redis_client.get(key)

    if cached_value is None:
        return None

    return json.loads(cached_value)


def cache_results(
    question: str,
    limit: int,
    results: list[dict],
) -> None:
    """Save search results in Redis for five minutes."""
    key = make_cache_key(question, limit)

    redis_client.setex(
        key,
        CACHE_TTL_SECONDS,
        json.dumps(results),
    )


def clear_search_cache() -> int:
    """Delete all hybrid-search cache entries."""
    keys = list(redis_client.scan_iter(match=f"{CACHE_PREFIX}*"))

    if not keys:
        return 0

    return redis_client.delete(*keys)
