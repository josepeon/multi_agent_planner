import json
import os
from dataclasses import asdict

class Memory:
    def __init__(self, filepath=None):
        self.data = {}
        self.filepath = filepath
        if filepath and os.path.exists(filepath):
            self._load()

    def _load(self):
        try:
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to load memory file: {e}")
            self.data = {}

    def _save(self):
        if self.filepath:
            try:
                with open(self.filepath, 'w') as f:
                    def serialize(obj):
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

    def set(self, key, value):
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            if isinstance(obj, list):
                return [serialize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        self.data[key] = serialize(value)
        self._save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def delete(self, key):
        if key in self.data:
            del self.data[key]
            self._save()

    def clear(self):
        self.data.clear()
        self._save()