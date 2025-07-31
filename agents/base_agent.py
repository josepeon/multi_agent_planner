from core.memory import Memory

class BaseAgent:
    def __init__(self, name, memory_filepath=None):
        self.name = name
        self.memory = Memory(memory_filepath)

    def run(self, input_data):
        raise NotImplementedError("Each agent must implement its own 'run' method.")
