from core.memory import Memory
from core.task_schema import Task

class BaseAgent:
    def __init__(self, name, memory_filepath=None):
        self.name = name
        self.memory = Memory(memory_filepath)

    def run(self, task: Task):
        raise NotImplementedError("Each agent must implement its own 'run' method for processing a Task object.")
