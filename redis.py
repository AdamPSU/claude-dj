"""Basic connection example."""

import os

import redis

r = redis.Redis(
    host="sugar-daylit-corn-40583.db.redis.io",
    port=18497,
    decode_responses=True,
    username="default",
    password=os.environ["REDIS_PASSWORD"],  # set in shell/.env; never hardcode
)

success = r.set("foo", "bar")
# True

result = r.get("foo")
print(result)
# >>> bar
