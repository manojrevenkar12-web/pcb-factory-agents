"""Running ASGI apps in a background thread: tests, demos and the evaluation harness."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn
from fastapi import FastAPI


class Server:
    """Runs an ASGI app in a background thread (tests, demos, the evaluation harness)."""

    def __init__(self, asgi: FastAPI, port: int, host: str = "127.0.0.1"):
        self.config = uvicorn.Config(asgi, host=host, port=port, log_level="warning", access_log=False)
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = f"http://{host}:{port}"

    def __enter__(self) -> Server:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
