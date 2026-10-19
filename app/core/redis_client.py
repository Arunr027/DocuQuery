
import redis
from redis import Redis

from app.core.config import settings


def get_redis_client() -> Redis:
    """Create a Redis client using application settings."""
    return redis.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


redis_client = get_redis_client()
