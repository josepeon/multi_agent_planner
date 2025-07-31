from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    id: int
    description: str
    status: str = "pending"
    result: Optional[str] = None