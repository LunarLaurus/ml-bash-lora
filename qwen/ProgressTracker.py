import json
import time
from pathlib import Path
from threading import Lock


class ProgressTracker:
    """
    Tracks model call statistics in a JSON file.
    Supports:
    - total calls
    - total prompts
    - total tokens
    - average latency
    - last update timestamp
    """

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.lock = Lock()
        # initialize stats
        self.stats = {
            "total_calls": 0,
            "total_prompts": 0,
            "total_tokens": 0,
            "average_latency_s": 0.0,
            "last_update": None,
        }
        if self.output_file.exists():
            try:
                with self.output_file.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.stats.update(loaded)
            except Exception:
                pass

    def update(self, num_prompts: int, tokens_used: int, batch_latency: float):
        with self.lock:
            prev_total_calls = self.stats["total_calls"]
            prev_avg = self.stats["average_latency_s"]
            self.stats["total_calls"] += 1
            self.stats["total_prompts"] += num_prompts
            self.stats["total_tokens"] += tokens_used
            # incremental average update
            self.stats["average_latency_s"] = (
                prev_avg * prev_total_calls + batch_latency
            ) / max(1, self.stats["total_calls"])
            self.stats["last_update"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            # write out immediately
            try:
                with self.output_file.open("w", encoding="utf-8") as f:
                    json.dump(self.stats, f, indent=2)
            except Exception:
                pass

    def snapshot(self):
        """Return a copy of current stats"""
        with self.lock:
            return dict(self.stats)
