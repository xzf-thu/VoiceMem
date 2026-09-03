"""Persistent logging for the web demo without changing existing print calls."""
from __future__ import annotations

import atexit
import os
import platform
import re
import sys
import threading
from datetime import datetime
from pathlib import Path


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _Tee:
    def __init__(self, console, logfile, label: str, lock: threading.RLock):
        self.console = console
        self.logfile = logfile
        self.label = label
        self.lock = lock
        self._line_start = True
        self.encoding = getattr(console, "encoding", "utf-8")
        self.errors = getattr(console, "errors", "replace")

    def write(self, value) -> int:
        text = str(value)
        with self.lock:
            self.console.write(text)
            clean = _ANSI.sub("", text).replace("\r", "\n")
            for part in clean.splitlines(keepends=True):
                if self._line_start:
                    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    self.logfile.write(f"{stamp} [{self.label}] ")
                self.logfile.write(part)
                self._line_start = part.endswith(("\n", "\r"))
            self.logfile.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.console.fileno()

    def writable(self) -> bool:
        return True


def setup_file_logging(root: Path, requested: str = "") -> Path:
    """Mirror stdout/stderr to one timestamped UTF-8 file and return its path."""
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = root / path
    else:
        log_dir = Path(os.environ.get("VOICEMEM_LOG_DIR", root / "results" / "logs"))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"voicemem-{stamp}-{os.getpid()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logfile = path.open("a", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, logfile, "stdout", lock)
    sys.stderr = _Tee(original_stderr, logfile, "stderr", lock)

    def close() -> None:
        with lock:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            logfile.flush()
            logfile.close()

    atexit.register(close)

    print(f"[log] 文件：{path}", flush=True)
    print(f"[log] Python={platform.python_version()} pid={os.getpid()} cwd={Path.cwd()}",
          flush=True)
    return path
