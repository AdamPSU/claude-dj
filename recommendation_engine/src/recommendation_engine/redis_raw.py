from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Iterable

from . import config


@dataclass(frozen=True)
class RawRedisConfig:
    host: str
    port: int
    username: str | None
    password: str | None
    socket_timeout_seconds: float
    socket_connect_timeout_seconds: float

    @classmethod
    def from_env(cls) -> RawRedisConfig:
        config.load_dotenv()
        return cls(
            host=config.getenv("REDIS_HOST", "localhost") or "localhost",
            port=int(config.getenv("REDIS_PORT", "6379") or "6379"),
            username=config.getenv("REDIS_USERNAME") or None,
            password=config.getenv("REDIS_PASSWORD") or None,
            socket_timeout_seconds=float(config.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "10") or "10"),
            socket_connect_timeout_seconds=float(config.getenv("REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS", "10") or "10"),
        )


class RawRedisClient:
    """Minimal RESP2 client for Redis Cloud binary vector reads/writes."""

    def __init__(self, raw_config: RawRedisConfig | None = None) -> None:
        self.config = raw_config or RawRedisConfig.from_env()

    def hset(self, key: str, mapping: dict[str, Any]) -> Any:
        parts: list[Any] = ["HSET", key]
        for field, value in mapping.items():
            parts.extend([field, value])
        return self.execute_command(*parts)

    def hmget(self, key: str, *fields: str) -> list[Any]:
        return list(self.execute_command("HMGET", key, *fields) or [])

    def hget(self, key: str, field: str) -> Any:
        return self.execute_command("HGET", key, field)

    def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        if ex is None:
            return self.execute_command("SET", key, value)
        return self.execute_command("SET", key, value, "EX", ex)

    def expire(self, key: str, seconds: int) -> Any:
        return self.execute_command("EXPIRE", key, seconds)

    def scan_iter(self, match: str) -> Iterable[Any]:
        yield from self.execute_command("KEYS", match) or []

    def execute_command(self, *parts: Any) -> Any:
        last_error: OSError | TimeoutError | None = None
        for _ in range(3):
            try:
                return self._execute_command_once(*parts)
            except (OSError, TimeoutError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Redis command failed before execution")

    def _execute_command_once(self, *parts: Any) -> Any:
        with socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.socket_connect_timeout_seconds,
        ) as sock:
            sock.settimeout(self.config.socket_timeout_seconds)
            sock.sendall(self._command(*self._hello_parts()))
            self._read_response(sock)
            sock.sendall(self._command(*parts))
            return self._read_response(sock)

    def _hello_parts(self) -> tuple[Any, ...]:
        if self.config.username or self.config.password:
            return ("HELLO", "2", "AUTH", self.config.username or "default", self.config.password or "")
        return ("HELLO", "2")

    def _command(self, *parts: Any) -> bytes:
        return b"*" + str(len(parts)).encode() + b"\r\n" + b"".join(self._bulk(part) for part in parts)

    def _bulk(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            data = value
        else:
            data = str(value).encode()
        return b"$" + str(len(data)).encode() + b"\r\n" + data + b"\r\n"

    def _read_response(self, sock: socket.socket) -> Any:
        line = self._read_line(sock)
        prefix = line[:1]
        body = line[1:-2]
        if prefix == b"+":
            return body
        if prefix == b":":
            return int(body)
        if prefix == b"-":
            raise RuntimeError(body.decode(errors="replace"))
        if prefix == b"$":
            length = int(body)
            if length < 0:
                return None
            payload = self._read_exact(sock, length + 2)
            return payload[:-2]
        if prefix == b"*":
            return [self._read_response(sock) for _ in range(int(body))]
        raise RuntimeError(f"unsupported Redis response prefix: {prefix!r}")

    def _read_line(self, sock: socket.socket) -> bytes:
        line = b""
        while not line.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                raise RuntimeError("Redis socket closed")
            line += chunk
        return line

    def _read_exact(self, sock: socket.socket, length: int) -> bytes:
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Redis socket closed")
            data += chunk
        return data


def get_raw_redis_client() -> RawRedisClient:
    return RawRedisClient()
