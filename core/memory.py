"""
Memory Module

Provides persistent key-value storage for agents to maintain state across sessions.
"""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional


class Memory:
    """Persistent key-value store with JSON file backing."""
    
    data: Dict[str, Any]
    filepath: Optional[str]
    
    def __init__(self, filepath: Optional[str] = None) -> None:
        self.data = {}
        self.filepath = filepath
        if filepath and os.path.exists(filepath):
            self._load()

    def _load(self) -> None:
        """Load data from the backing JSON file."""
        try:
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to load memory file: {e}")
            self.data = {}

    def _save(self) -> None:
        """Persist data to the backing JSON file."""
        if self.filepath:
            try:
                with open(self.filepath, 'w') as f:
                    def serialize(obj: Any) -> Any:
                        if hasattr(obj, '__dataclass_fields__'):
                            return asdict(obj)
                        if isinstance(obj, list):
                            return [serialize(v) for v in obj]
                        if isinstance(obj, dict):
                            return {k: serialize(v) for k, v in obj.items()}
                        return obj

                    serializable_data = {k: serialize(v) for k, v in self.data.items()}
                    json.dump(serializable_data, f, indent=2)
            except Exception as e:
                print(f"Failed to save memory file: {e}")

    def set(self, key: str, value: Any) -> None:
        """Set a key-value pair in memory."""
        def serialize(obj: Any) -> Any:
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            if isinstance(obj, list):
                return [serialize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        self.data[key] = serialize(value)
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory by key."""
        return self.data.get(key, default)

    def delete(self, key: str) -> None:
        """Delete a key from memory."""
        if key in self.data:
            del self.data[key]
            self._save()

    def clear(self) -> None:
        """Clear all data from memory."""
        self.data.clear()
        self._save()