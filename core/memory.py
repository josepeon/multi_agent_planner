import json
import os

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
                    json.dump(self.data, f, indent=2)
            except Exception as e:
                print(f"Failed to save memory file: {e}")

    def set(self, key, value):
        self.data[key] = value
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