from dataclasses import dataclass


@dataclass
class Task:
    id: int
    description: str
    status: str = "pending"
    result: str | None = None
