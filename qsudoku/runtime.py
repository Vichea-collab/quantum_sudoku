"""Run heavy Qiskit/Aer work on a thread with a large stack.

macOS gives secondary threads a ~512 KB stack. Streamlit runs scripts on such a thread, and
Qiskit/Aer (Rust + C++ + deep import chains) can overflow it and segfault. Hopping to our own
thread with a big stack avoids that. Outside Streamlit it is harmless.
"""
from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")
STACK_BYTES = 256 * 1024 * 1024


def run_big_stack(fn: Callable[..., T], *args, **kwargs) -> T:
    box: dict = {}

    def work() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:              # re-raised in the caller's thread
            box["error"] = exc

    previous = threading.stack_size()
    threading.stack_size(STACK_BYTES)
    try:
        th = threading.Thread(target=work, name="big-stack-worker")
        th.start()
    finally:
        threading.stack_size(previous)
    th.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
