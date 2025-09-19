"""Kafka reader for STM streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str = "stm-stream"


class KafkaReader:
    def __init__(self, config: KafkaConfig) -> None:
        self.config = config

    def poll(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError("Kafka polling not yet implemented")
